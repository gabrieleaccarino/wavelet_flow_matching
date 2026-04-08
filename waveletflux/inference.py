"""
run_inference.py
----------------
Distributed ensemble inference for WaveletFlowModule (and data-space variants).

Usage — single device (MPS / CPU):
    python run_inference.py --config configs/trl2d_wavelet.yaml

Usage — multi-GPU with torchrun:
    torchrun --nproc_per_node=4 run_inference.py --config configs/trl2d_wavelet.yaml

The script:
  • Splits trajectories evenly across ranks.
  • Each rank generates `n_ensemble // world_size` ensemble members per trajectory
    (ensemble members are concatenated at gather time on rank 0).
  • Computes RMSE, ensemble spread, spread-skill ratio, and CRPS per field/step.
  • Saves raw predictions to a single pickle on rank 0.
"""

import argparse
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Optional

import fsspec
import numpy as np
import torch
import torch.distributed as dist
import yaml

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WaveletFlow ensemble inference")
    p.add_argument("--config", required=True, help="Path to YAML experiment config")
    p.add_argument(
        "--override",
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help="Override config keys, e.g. --override n_ensemble=32 num_sampling_steps=50",
    )
    return p.parse_args()


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply dot-notation or flat KEY=VALUE overrides to cfg in-place."""
    for kv in overrides:
        k, v = kv.split("=", 1)
        # Try to cast to int / float / bool before falling back to str
        for cast in (int, float, lambda x: {"true": True, "false": False}[x.lower()]):
            try:
                v = cast(v)
                break
            except (ValueError, KeyError):
                pass
        keys = k.split(".")
        d = cfg
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = v
    return cfg


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def init_distributed() -> tuple[int, int, torch.device]:
    """
    Initialise process group when launched with torchrun.
    Falls back to single-process (rank=0, world=1) otherwise.
    """
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        device     = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    elif torch.backends.mps.is_available():
        rank, world_size = 0, 1
        device = torch.device("mps")
    else:
        rank, world_size = 0, 1
        device = torch.device("cpu")

    return rank, world_size, device


def is_main(rank: int) -> bool:
    return rank == 0


def barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


# ---------------------------------------------------------------------------
# Trajectory splitting
# ---------------------------------------------------------------------------

def get_local_indices(n_traj: int, rank: int, world_size: int) -> list[int]:
    """Return the trajectory indices assigned to this rank."""
    indices = list(range(n_traj))
    # Spread indices round-robin so load is balanced even if n_traj % world_size != 0
    return indices[rank::world_size]

# ---------------------------------------------------------------------------
# Gather helpers
# ---------------------------------------------------------------------------

def gather_tensor_to_rank0(tensor: torch.Tensor, rank: int, world_size: int) -> Optional[torch.Tensor]:
    """Gather a variable-length 1-D CPU byte tensor from all ranks to rank 0."""
    if world_size == 1:
        return tensor

    # Serialise to bytes so shapes need not be identical
    buf  = pickle.dumps(tensor)
    size = torch.tensor([len(buf)], dtype=torch.long)

    # Share sizes
    sizes = [torch.zeros(1, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(sizes, size)

    if is_main(rank):
        max_size = max(s.item() for s in sizes)
    else:
        max_size = 0
    max_size_t = torch.tensor([max_size], dtype=torch.long)
    dist.broadcast(max_size_t, src=0)
    max_size = max_size_t.item()

    padded = torch.zeros(max_size, dtype=torch.uint8)
    padded[: len(buf)] = torch.frombuffer(buf, dtype=torch.uint8)

    gathered = [torch.zeros(max_size, dtype=torch.uint8) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    if is_main(rank):
        return [pickle.loads(g[: sizes[i].item()].numpy().tobytes()) for i, g in enumerate(gathered)]
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Load + patch config ──────────────────────────────────────────────────
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, args.override)

    project_root  = cfg["project_root"]
    sys.path.insert(0, project_root)

    # Late imports so project_root is on sys.path
    from waveletflux.utils.well_data import get_well_dataset, get_well_inputs
    from waveletflux.modules.wavelet_fm import WaveletFlowModule

    # ── Distributed init ─────────────────────────────────────────────────────
    rank, world_size, device = init_distributed()

    if is_main(rank):
        print(f"\n{'═'*64}")
        print(f" Inference Pipeline ")
        print(f"{'═'*64}")
        print(f"  Config      : {args.config}")
        print(f"  Dataset     : {cfg['well_dataset_name']}")
        print(f"  Checkpoint  : {cfg['ckpt_name']}")
        print(f"  N_ensemble  : {cfg['n_ensemble']}")
        print(f"  World size  : {world_size}  |  Device : {device}")
        print(f"{'─'*64}\n")

    # ── Paths ────────────────────────────────────────────────────────────────
    well_base_path    = cfg["well_base_path"]
    well_dataset_name = cfg["well_dataset_name"]
    ckpt_name         = cfg["ckpt_name"]
    ckpt_path         = os.path.join(project_root, "experiments", well_dataset_name, "ckpts", ckpt_name)
    # comment: this is managed by the normalizer inside the model, so no need to load it separately here
    #stats_path        = os.path.join(well_base_path, well_dataset_name, "stats.yaml")

    ckpt_base_name = re.split(r"_\d{8}_", ckpt_name)[0]
    # Handle checkpoints whose names don't contain a date stamp
    if ckpt_base_name == ckpt_name:
        ckpt_base_name = Path(ckpt_name).stem

    # ── Load model (every rank needs its own copy) ───────────────────────────
    model = WaveletFlowModule.from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)

    if is_main(rank):
        print(f"  Model loaded OK on rank {rank}\n")

    barrier(world_size)

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = get_well_dataset(
        path      = well_base_path,
        physics   = well_dataset_name,
        split     = cfg.get("split", "test"),
        steps     = cfg.get("steps", -1),
        transform = cfg.get("transform", ["log_scalars"]),
        augment   = cfg.get("augment", []),
    )

    n_steps_per_traj = dataset.metadata.n_steps_per_trajectory
    N_TRAJ           = len(n_steps_per_traj)
    N_STEPS_INPUT    = cfg["n_steps_input"]
    NUM_SAMPLING     = cfg["num_sampling_steps"]
    N_ENSEMBLE       = cfg["n_ensemble"]

    # How many ensemble members each rank generates
    n_ens_local = max(1, N_ENSEMBLE // world_size)
    # Rank 0 gets any remainder
    if is_main(rank):
        n_ens_local += N_ENSEMBLE - (n_ens_local * world_size)

    # Trajectory indices for this rank
    local_traj_indices = get_local_indices(N_TRAJ, rank, world_size)

    if is_main(rank):
        print(f"  N_TRAJ={N_TRAJ}  N_STEPS_INPUT={N_STEPS_INPUT}  "
              f"N_ROLLOUT varies per traj  NUM_SAMPLING={NUM_SAMPLING}")
        print(f"  Each rank generates {n_ens_local} ensemble members/traj")
        print(f"  Rank 0 trajectories: {local_traj_indices[:5]}{'...' if len(local_traj_indices) > 5 else ''}\n")

    # ── Inference loop ───────────────────────────────────────────────────────
    local_preds   = {}   # traj_idx → (n_ens_local, T, H, W, C) CPU tensor
    local_targets = {}   # traj_idx → (T, H, W, C)              CPU tensor

    t0 = time.perf_counter()

    with torch.inference_mode():
        for i, traj_idx in enumerate(local_traj_indices):
            n_rollout = n_steps_per_traj[traj_idx] - N_STEPS_INPUT

            if is_main(rank):
                print(f"  [rank {rank}] traj {traj_idx}  ({i+1}/{len(local_traj_indices)})  "
                      f"rollout_steps={n_rollout}")

            item          = dataset[traj_idx]
            x_full, label = get_well_inputs(item, device=device)  # (T,H,W,C), (num_cond,)

            seed      = x_full[:N_STEPS_INPUT]              # (T_in, H, W, C)
            seed_exp  = seed.unsqueeze(0).expand(n_ens_local, -1, -1, -1, -1)
            label_exp = (
                label.unsqueeze(0).expand(n_ens_local, -1)
                if label is not None else None
            )

            _, preds_phys = model.rollout(
                seed             = seed_exp,
                scalar_conds     = label_exp,
                rollout_steps    = n_rollout,
                n_steps_input    = N_STEPS_INPUT,
                n_ensemble       = 1,               # 1 member per batch element
                normalized_input = False,
                verbose          = cfg.get("verbose", False),
            )
            # preds_phys: (n_ens_local, 1, n_rollout, H, W, C)  → squeeze forecast dim
            preds_phys = preds_phys.squeeze(1)   # (n_ens_local, n_rollout, H, W, C)

            local_preds[traj_idx]   = preds_phys.cpu()
            local_targets[traj_idx] = x_full[N_STEPS_INPUT:].cpu()   # ground truth rollout

            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    if is_main(rank):
        print(f"\n  Rank {rank} finished inference in {elapsed:.1f}s\n")

    barrier(world_size)

    # ── Gather across ranks ───────────────────────────────────────────────────
    # Each rank may have different trajectory indices and different n_ens_local.
    # Strategy: serialise local dicts and gather on rank 0.
    if world_size > 1:
        gathered_preds   = gather_tensor_to_rank0(local_preds,   rank, world_size)
        gathered_targets = gather_tensor_to_rank0(local_targets, rank, world_size)
    else:
        gathered_preds   = [local_preds]
        gathered_targets = [local_targets]

    # ── Rank 0: merge, save ─────────────────────────────────
    if is_main(rank):
        print("  Merging results across ranks ...")

        # Merge per-traj dicts from all ranks
        merged_preds   = {}   # traj_idx → list of (n_ens_local, T, H, W, C) tensors
        merged_targets = {}   # traj_idx → (T, H, W, C) tensor

        for rank_preds, rank_targets in zip(gathered_preds, gathered_targets):
            for idx, arr in rank_preds.items():
                merged_preds.setdefault(idx, []).append(arr)
            for idx, arr in rank_targets.items():
                merged_targets[idx] = arr

        # Sort by trajectory index and concatenate ensemble members
        all_indices = sorted(merged_preds.keys())
        preds_list  = []
        tgts_list   = []

        for idx in all_indices:
            cat = torch.cat(merged_preds[idx], dim=0)   # (N_ENS, T, H, W, C)
            preds_list.append(cat)
            tgts_list.append(merged_targets[idx])

        # Pad to uniform rollout length (take minimum) if trajectories differ in length
        min_T = min(p.shape[1] for p in preds_list)
        preds_arr  = torch.stack([p[:, :min_T] for p in preds_list], dim=0).numpy()  # (N_TRAJ, N_ENS, T, H, W, C)
        targets_arr = torch.stack([t[:min_T]   for t in tgts_list],  dim=0).numpy()  # (N_TRAJ,      , T, H, W, C)

        print(f"  preds shape  : {preds_arr.shape}")
        print(f"  targets shape: {targets_arr.shape}")

        # ── Save ─────────────────────────────────────────────────────────────
        save_dir = Path(project_root) / "experiments" / well_dataset_name / "results"
        save_dir.mkdir(parents=True, exist_ok=True)

        fname = (
            f"{ckpt_base_name}"
            f"_nsampling{NUM_SAMPLING}"
            f"_nens{N_ENSEMBLE}"
            f"_traj.pkl"
        )
        save_path = save_dir / fname

        payload = {
            "preds":        preds_arr,           # (N_TRAJ, N_ENS, T, H, W, C)
            "targets":      targets_arr,         # (N_TRAJ,      , T, H, W, C)
            "config":       cfg,
            "traj_indices": all_indices,
        }

        with open(save_path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"  Saved → {save_path}\n")
        print(f"{'═'*64}")
        print(f"  Done.")
        print(f"{'═'*64}\n")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()