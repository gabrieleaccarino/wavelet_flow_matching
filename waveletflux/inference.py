"""
inference.py
------------
Distributed ensemble inference for WaveletFlowModule (and data-space variants).

Parallelism strategy
--------------------
  Every rank processes ALL trajectories but generates only
  (N_ENSEMBLE // world_size) ensemble members each.  After inference,
  each rank writes its shard to the shared filesystem (Lustre on Derecho).
  Rank 0 waits for all ranks to finish (barrier), then reads every shard,
  concatenates along the ensemble axis, and produces the final outputs.

  This avoids any GPU-to-GPU communication for the data gather, which would
  require allocating world_size x shard_size bytes on each GPU simultaneously
  and cause OOM on large fields.

Outputs (all written by rank 0)
--------------------------------
  <stem>_traj.pkl       full predictions  (N_TRAJ, N_ENS, T, H, W, C)
  <stem>_nens1_traj.pkl single member     (N_TRAJ,    1,  T, H, W, C)  for plotting
  <stem>_stats.csv      per-(model, traj, field, step) scalar metrics
  <stem>_psd.csv        isotropic PSD in long format

Usage — single device (MPS / CPU)
    python inference.py --config configs/trl2d/inference_trl2d.yaml

Usage — multi-GPU with torchrun
    torchrun --nproc_per_node=4 inference.py --config configs/trl2d/inference_trl2d.yaml

CLI overrides
    torchrun --nproc_per_node=4 inference.py \\
        --config configs/trl2d/inference_trl2d.yaml \\
        --override n_ensemble=16 num_sampling_steps=50 verbose=true
"""

import argparse
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WaveletFlow ensemble inference")
    p.add_argument("--config", required=True, help="Path to YAML experiment config")
    p.add_argument(
        "--override",
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help="Override any config key, e.g. --override n_ensemble=32 verbose=true",
    )
    return p.parse_args()


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply flat or dot-notation KEY=VALUE pairs to cfg in-place."""
    for kv in overrides:
        k, v = kv.split("=", 1)
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


# ─────────────────────────────────────────────────────────────────────────────
# Distributed helpers
# ─────────────────────────────────────────────────────────────────────────────

def init_distributed() -> tuple[int, int, torch.device]:
    """
    Initialise the NCCL process group when launched with torchrun.
    Falls back gracefully to single-process on MPS or CPU.
    """
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        # Passing device_id tells NCCL exactly which GPU owns this process,
        # eliminating the "devices used by this process are currently unknown" warning.
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
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


def barrier(world_size: int, device: torch.device | None = None) -> None:
    if world_size > 1:
        device_ids = [device.index] if device is not None and device.type == "cuda" else None
        dist.barrier(device_ids=device_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Filesystem-based shard gather
# ─────────────────────────────────────────────────────────────────────────────
# GPU collectives (all_gather / dist.gather) for prediction tensors would
# require every rank to allocate world_size receive buffers simultaneously
# on-GPU (e.g. 4 x 7.7 GiB = 30 GiB for a 4-GPU job on trl2d) → OOM.
#
# Instead, each rank serialises its shard to Lustre and rank 0 reads them
# back after a barrier.  Zero GPU memory is used for communication.

def _shard_path(shard_dir: Path, rank: int) -> Path:
    return shard_dir / f"shard_rank{rank:02d}.pkl"


def write_shard(obj: object, shard_dir: Path, rank: int) -> Path:
    path = _shard_path(shard_dir, rank)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def read_all_shards(shard_dir: Path, world_size: int) -> list:
    shards = []
    for r in range(world_size):
        with open(_shard_path(shard_dir, r), "rb") as fh:
            shards.append(pickle.load(fh))
    return shards


def remove_shards(shard_dir: Path, world_size: int) -> None:
    for r in range(world_size):
        _shard_path(shard_dir, r).unlink(missing_ok=True)
    try:
        shard_dir.rmdir()   # succeeds only if the directory is now empty
    except OSError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, args.override or [])

    project_root = cfg["project_root"]
    sys.path.insert(0, project_root)

    # Late imports — project_root must be on sys.path first
    from waveletflux.utils.well_data import get_well_dataset, get_well_inputs
    from waveletflux.utils.eval import build_stats_df, build_psd_df
    from waveletflux.modules.wavelet_fm import WaveletFlowModule

    # ── Distributed init ──────────────────────────────────────────────────────
    rank, world_size, device = init_distributed()

    # ── Experiment identifiers ────────────────────────────────────────────────
    well_base_path    = cfg["well_base_path"]
    well_dataset_name = cfg["well_dataset_name"]
    ckpt_name         = cfg["ckpt_name"]
    N_STEPS_INPUT     = cfg["n_steps_input"]
    NUM_SAMPLING      = cfg["num_sampling_steps"]
    N_ENSEMBLE        = cfg["n_ensemble"]
    ODE_METHOD        = cfg.get("ode_method", "euler")       
    CLAMP_SIGMA       = cfg.get("clamp_sigma", 5.0)    

    # Strip date stamp (e.g. _20260406_) for cleaner save names.
    # Fall back to the full stem when no date stamp is present.
    ckpt_base_name = re.split(r"_\d{8}_", ckpt_name)[0]
    if ckpt_base_name == ckpt_name:
        ckpt_base_name = Path(ckpt_name).stem

    ckpt_path = os.path.join(
        project_root, "experiments", well_dataset_name,
        ckpt_base_name, "artifacts/checkpoints", ckpt_name,
    )

    save_dir  = Path(project_root) / "experiments" / well_dataset_name / "results"
    shard_dir = save_dir / "_shards"

    # Rank 0 creates output directories; others wait
    if is_main(rank):
        save_dir.mkdir(parents=True, exist_ok=True)
        shard_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'═'*64}")
        print(f"  Inference Pipeline")
        print(f"{'═'*64}")
        print(f"  Config      : {args.config}")
        print(f"  Dataset     : {well_dataset_name}")
        print(f"  Checkpoint  : {ckpt_name}")
        print(f"  N_ensemble  : {N_ENSEMBLE}")
        print(f"  World size  : {world_size}  |  Device : {device}")
        print(f"{'─'*64}\n")

    barrier(world_size, device)

    # ── Model — every rank loads its own copy ─────────────────────────────────
    model = WaveletFlowModule.from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)
    print(f"  [rank {rank}] model loaded on {device}")
    barrier(world_size, device)

    # ── Dataset ───────────────────────────────────────────────────────────────
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

    # ── Ensemble split ────────────────────────────────────────────────────────
    # Every rank handles ALL trajectories.
    # Ensemble members are divided across ranks; any remainder is distributed
    # one extra member to the first (n_ens_remainder) ranks.
    n_ens_base      = N_ENSEMBLE // world_size
    n_ens_remainder = N_ENSEMBLE  % world_size
    n_ens_local     = n_ens_base + (1 if rank < n_ens_remainder else 0)

    if is_main(rank):
        print(f"  N_TRAJ        : {N_TRAJ}")
        print(f"  N_STEPS_INPUT : {N_STEPS_INPUT}")
        print(f"  NUM_SAMPLING  : {NUM_SAMPLING}")
        print(f"  ODE method  : {ODE_METHOD}")
        print(f"  Clamp sigma : {CLAMP_SIGMA}")
        print(f"  N_ENSEMBLE    : {N_ENSEMBLE}  "
              f"({n_ens_local} member(s) on this rank)\n")

    # ── Inference loop — all ranks, all trajectories ──────────────────────────
    # local_preds  : {traj_idx: Tensor(n_ens_local, T, H, W, C)} on CPU
    # local_targets: {traj_idx: Tensor(T, H, W, C)}              on CPU
    local_preds   = {}
    local_targets = {}

    t0 = time.perf_counter()

    with torch.inference_mode():
        for traj_idx in range(N_TRAJ):
            n_rollout = n_steps_per_traj[traj_idx] - N_STEPS_INPUT

            if is_main(rank):
                pct = (traj_idx + 1) / N_TRAJ * 100
                print(f"  [{pct:5.1f}%] traj {traj_idx:>3d}/{N_TRAJ}  "
                      f"rollout_steps={n_rollout}")

            item          = dataset[traj_idx]
            x_full, label = get_well_inputs(item, device=device)   # (T,H,W,C), (C_cond,)

            seed      = x_full[:N_STEPS_INPUT]
            # .contiguous() avoids expand() creating non-contiguous views
            # that some internal ops may not accept
            seed_exp  = seed.unsqueeze(0).expand(n_ens_local, -1, -1, -1, -1).contiguous()
            label_exp = (
                label.unsqueeze(0).expand(n_ens_local, -1).contiguous()
                if label is not None else None
            )

            _, preds_phys = model.rollout(
                seed             = seed_exp,
                scalar_conds     = label_exp,
                rollout_steps    = n_rollout,
                n_steps_input    = N_STEPS_INPUT,
                n_ensemble       = 1,       # 1 member per batch element
                normalized_input = False,
                verbose          = cfg.get("verbose", False),
                ode_method       = ODE_METHOD,           # NEW
                clamp_sigma      = CLAMP_SIGMA,          # NEW, 0 means off
            )
            # (n_ens_local, 1, T, H, W, C) → (n_ens_local, T, H, W, C)
            preds_phys = preds_phys.squeeze(1)

            local_preds[traj_idx]   = preds_phys.cpu()
            local_targets[traj_idx] = x_full[N_STEPS_INPUT:].cpu()

            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

    elapsed = time.perf_counter() - t0
    print(f"  [rank {rank}] inference done in {elapsed:.1f}s  "
          f"({n_ens_local} member(s) x {N_TRAJ} trajectories)")

    # ── Write shard to shared filesystem ─────────────────────────────────────
    path = write_shard({"preds": local_preds, "targets": local_targets}, shard_dir, rank)
    print(f"  [rank {rank}] shard written → {path}")

    # All ranks must finish writing before rank 0 reads
    barrier(world_size, device)

    # Non-main ranks are done
    if not is_main(rank):
        if world_size > 1:
            dist.destroy_process_group()
        return

    # ── Rank 0: read shards ───────────────────────────────────────────────────
    print(f"\n  Reading {world_size} shard(s) from disk …")
    shards = read_all_shards(shard_dir, world_size)
    remove_shards(shard_dir, world_size)

    # ── Merge: cat ensemble shards per trajectory ─────────────────────────────
    merged_preds   = {}   # traj_idx → list[(n_ens_local, T, H, W, C)]
    merged_targets = {}   # traj_idx → (T, H, W, C)   identical across ranks, keep one

    for shard in shards:
        for idx, arr in shard["preds"].items():
            merged_preds.setdefault(idx, []).append(arr)
        for idx, arr in shard["targets"].items():
            if idx not in merged_targets:
                merged_targets[idx] = arr

    all_indices = sorted(merged_preds.keys())

    # Clip all trajectories to the shortest rollout (handles variable-length datasets)
    min_T = min(
        torch.cat(merged_preds[idx], dim=0).shape[1]
        for idx in all_indices
    )

    preds_list = []
    tgts_list  = []
    for idx in all_indices:
        cat = torch.cat(merged_preds[idx], dim=0)   # (N_ENS, T, H, W, C)
        preds_list.append(cat[:, :min_T])
        tgts_list.append(merged_targets[idx][:min_T])

    preds_arr   = torch.stack(preds_list, dim=0).numpy()  # (N_TRAJ, N_ENS, T, H, W, C)
    targets_arr = torch.stack(tgts_list,  dim=0).numpy()  # (N_TRAJ,        T, H, W, C)

    print(f"  preds shape   : {preds_arr.shape}")
    print(f"  targets shape : {targets_arr.shape}")

    assert preds_arr.shape[1] == N_ENSEMBLE, (
        f"Expected {N_ENSEMBLE} ensemble members, got {preds_arr.shape[1]}. "
        f"Check n_ensemble vs world_size."
    )

    # ── Field names ───────────────────────────────────────────────────────────
    #n_fields    = preds_arr.shape[-1]

    meta_fields = dataset.metadata.field_names
    field_names = [item for sublist in meta_fields.values() for item in sublist]
    print(f"  field names   : {field_names}")
    
    #field_names = list(
    #    getattr(dataset.metadata, "field_names", None)
    #    or [f"field_{i}" for i in range(n_fields)]
    #)

    stem = f"{ckpt_base_name}_nsampling{NUM_SAMPLING}_nens{N_ENSEMBLE}"

    # ── 1. Full pickle ────────────────────────────────────────────────────────
    full_pkl = save_dir / f"{stem}_traj.pkl"
    with open(full_pkl, "wb") as fh:
        pickle.dump(
            {
                "preds":        preds_arr,      # (N_TRAJ, N_ENS, T, H, W, C)
                "targets":      targets_arr,    # (N_TRAJ,        T, H, W, C)
                "config":       cfg,
                "traj_indices": all_indices,
                "field_names":  field_names,
            },
            fh, protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"\n  [1/4] Full pickle        → {full_pkl}")

    # ── 2. Single-member pickle (small download for plotting) ─────────────────
    single_pkl = save_dir / f"{ckpt_base_name}_nsampling{NUM_SAMPLING}_nens1_traj.pkl"
    with open(single_pkl, "wb") as fh:
        pickle.dump(
            {
                "preds":        preds_arr[:, :1],   # (N_TRAJ, 1, T, H, W, C)
                "targets":      targets_arr,
                "config":       cfg,
                "traj_indices": all_indices,
                "field_names":  field_names,
            },
            fh, protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"  [2/4] Single-member pkl  → {single_pkl}")

    # ── 3 & 4. Stats and PSD DataFrames ──────────────────────────────────────
    # build_stats_df / build_psd_df expect:
    #   truth       : (N_TRAJ, Ti + T, H, W, C)  full trajectory including context
    #   predictions : {name: (N_TRAJ, N_ENS, T, H, W, C)}
    #   Ti          : context frames to skip when aligning truth to the rollout
    #
    # targets_arr starts at step Ti (rollout only), so we prepend the seed window
    # by re-fetching it from the dataset on CPU — cheap, no GPU involved.
    print(f"\n  Rebuilding full truth tensor (Ti={N_STEPS_INPUT}) ...")

    seeds_list = []
    for idx in all_indices:
        item       = dataset[idx]
        x_full, _  = get_well_inputs(item, device=torch.device("cpu"))
        seeds_list.append(x_full.numpy()[:N_STEPS_INPUT + min_T])  # (Ti + T, H, W, C)

    truth_traj  = torch.from_numpy(np.stack(seeds_list, axis=0))   # (N_TRAJ, Ti+T, H, W, C)
    preds_torch = torch.from_numpy(preds_arr)                      # (N_TRAJ, N_ENS, T, H, W, C)
    predictions = {ckpt_base_name: preds_torch}

    print(f"  Building stats DataFrame ...")
    stats_df = build_stats_df(
        truth       = truth_traj,
        predictions = predictions,
        field_names = field_names,
        Ti          = N_STEPS_INPUT,
        lead_times  = None,
        stride      = 1,
    )
    stats_csv = save_dir / f"{stem}_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"  [3/4] Stats CSV          → {stats_csv}  ({len(stats_df):,} rows)")

    print(f"  Building PSD DataFrame ...")
    psd_df = build_psd_df(
        truth         = truth_traj,
        predictions   = predictions,
        field_names   = field_names,
        Ti            = N_STEPS_INPUT,
        lead_times    = None,
        stride        = 1,
        min_bin_count = 3,
    )
    psd_csv = save_dir / f"{stem}_psd.csv"
    psd_df.to_csv(psd_csv, index=False)
    print(f"  [4/4] PSD CSV            → {psd_csv}  ({len(psd_df):,} rows)")

    print(f"\n{'═'*64}")
    print(f"  Done.  All outputs in:")
    print(f"  {save_dir}")
    print(f"{'═'*64}\n")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
