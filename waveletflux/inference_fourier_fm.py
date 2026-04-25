"""
inference.py
------------
Distributed ensemble inference for FourierFlowModule (and data-space variants).

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

Trajectory subsampling
----------------------
  N_TRAJ_PER_FILE controls how many trajectories are drawn from each HDF5 file.
  Set to 1 to evaluate one trajectory per unique scalar-condition combination
  (cheapest), or to -1 to use every trajectory in each file (full evaluation).

  Examples
    n_traj_per_file: 1    →  9 trajectories for a 9-file Rayleigh-Bénard test set
    n_traj_per_file: 3    →  27 trajectories
    n_traj_per_file: -1   →  all trajectories (45 for 9 files × 5 traj/file)

  CLI override:
    torchrun ... inference.py --config ... --override n_traj_per_file=3

Outputs (all written by rank 0)
--------------------------------
  <stem>_traj.pkl       full predictions  (N_TRAJ, N_ENS, T, H, W, C)
  <stem>_nens1_traj.pkl single member     (N_TRAJ,    1,  T, H, W, C)  for plotting
  <stem>_stats.csv      per-(model, traj, field, step) scalar metrics
                        + one column per scalar condition (e.g. Rayleigh, Prandtl)
  <stem>_psd.csv        isotropic PSD in long format
                        + one column per scalar condition

Usage — single device (MPS / CPU)
    python inference.py --config configs/trl2d/inference_trl2d.yaml

Usage — multi-GPU with torchrun
    torchrun --nproc_per_node=4 inference.py --config configs/trl2d/inference_trl2d.yaml

CLI overrides
    torchrun --nproc_per_node=4 inference.py \\
        --config configs/trl2d/inference_trl2d.yaml \\
        --override n_ensemble=16 num_sampling_steps=50 n_traj_per_file=1 verbose=true
"""

from the_well.data.normalization import ZScoreNormalization
import torch.distributed as dist
from pathlib import Path
import numpy as np
import argparse
import pickle
import torch
import yaml
import time
import sys
import os
import re

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# !! TRAJECTORY SUBSAMPLING MACRO !!
# ─────────────────────────────────────────────────────────────────────────────
# Number of trajectories to evaluate per HDF5 file.
#   1   → one trajectory per unique scalar-condition (fastest, recommended for sweeps)
#   N   → first N trajectories from each file
#  -1   → all trajectories in each file (full evaluation)
#
# This default is overridden by the YAML key `n_traj_per_file` or the CLI flag
# --override n_traj_per_file=N, so you never need to edit this file directly.
N_TRAJ_PER_FILE_DEFAULT: int = 1
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FourierFlow ensemble inference")
    p.add_argument("--config", required=True, help="Path to YAML experiment config")
    p.add_argument(
        "--override",
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help="Override any config key, e.g. --override n_ensemble=32 n_traj_per_file=1",
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

def _shard_path(shard_dir: Path, rank: int) -> Path:
    return shard_dir / f"shard_rank{rank:02d}.pkl"


def write_shard(obj: object, shard_dir: Path, rank: int) -> Path:
    path = _shard_path(shard_dir, rank)
    # Defensive mkdir: handles Lustre metadata lag on non-zero ranks
    # even after the barrier, and any other timing edge-case.
    path.parent.mkdir(parents=True, exist_ok=True)
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
        shard_dir.rmdir()
    except OSError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory index helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_traj_index(
    n_traj_per_file:   list[int],
    steps_per_file:    list[int],
    traj_per_file_cap: int,
) -> tuple[list[int], list[int]]:
    """Return (dataset_indices, n_steps) for the selected trajectories.

    Parameters
    ----------
    n_traj_per_file   : list of trajectory counts per file  (from dataset)
    steps_per_file    : list of step counts per file        (from metadata)
    traj_per_file_cap : how many trajectories to take from each file;
                        -1 means take all.

    Returns
    -------
    traj_indices   : flat list of integer indices into the dataset
    n_steps_list   : corresponding number of steps for each selected trajectory
    """
    traj_indices = []
    n_steps_list = []
    offset = 0
    for n_traj, n_steps in zip(n_traj_per_file, steps_per_file):
        cap = n_traj if traj_per_file_cap == -1 else min(traj_per_file_cap, n_traj)
        for local_i in range(cap):
            traj_indices.append(offset + local_i)
            n_steps_list.append(n_steps)
        offset += n_traj
    return traj_indices, n_steps_list


# ─────────────────────────────────────────────────────────────────────────────
# Label helpers
# ─────────────────────────────────────────────────────────────────────────────

def label_to_dict(label: torch.Tensor | None, scalar_names: list[str]) -> dict[str, float]:
    """Convert a 1-D label tensor to a {name: float} dict.

    If scalar_names is shorter than the tensor (or empty), unnamed entries are
    stored as 'cond_0', 'cond_1', … so no information is ever silently dropped.
    """
    if label is None:
        return {}
    values = label.detach().cpu().float().tolist()
    names = list(scalar_names) + [
        f"cond_{i}" for i in range(len(scalar_names), len(values))
    ]
    return {n: v for n, v in zip(names, values)}


def build_labels_df(
    labels_dict: dict[int, dict[str, float]],
    traj_indices: list[int],
) -> pd.DataFrame:
    """Build a tidy DataFrame with one row per trajectory and one column per
    scalar condition.  ``traj`` matches the key used in stats / psd DataFrames
    so it can be directly merged on that column.

    Example output:
        traj  Rayleigh  Prandtl
           0   23.0259      1.0
           1    9.2103      0.5
    """
    rows = [{"traj": idx, **labels_dict.get(idx, {})} for idx in traj_indices]
    return pd.DataFrame(rows)


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
    from waveletflux.modules.fourier_fm import FourierFlowModule

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
    # Trajectory cap: YAML > CLI override > module-level default
    N_TRAJ_PER_FILE   = cfg.get("n_traj_per_file", N_TRAJ_PER_FILE_DEFAULT)

    ckpt_base_name = re.split(r"_\d{8}_", ckpt_name)[0]
    if ckpt_base_name == ckpt_name:
        ckpt_base_name = Path(ckpt_name).stem

    ckpt_path = os.path.join(
        project_root, "experiments", well_dataset_name,
        ckpt_base_name, "artifacts/checkpoints", ckpt_name,
    )

    # stem is unique per (checkpoint, sampling-steps, ensemble-size) — used both
    # for output filenames and for the temporary shard directory so that
    # concurrent inference runs never share the same shard folder.
    stem = f"{ckpt_base_name}_nsampling{NUM_SAMPLING}_nens{N_ENSEMBLE}"

    save_dir = Path(project_root) / "experiments" / well_dataset_name / "results"

    # Embed the PBS job ID (or PID as fallback) so that concurrent jobs with
    # identical configs never share the same shard directory.
    job_id = (
        os.environ.get("PBS_JOBID")               # PBS / Derecho production
        or os.environ.get("TORCHELASTIC_RUN_ID")  # torchrun >= 2.0
        or os.environ.get("MASTER_PORT")          # torchrun fallback, same on all ranks
        or str(os.getpid())                       # world_size=1 only (single process, safe)
    )
    shard_dir = save_dir / f"_shards_{stem}_{job_id}"

    # Only rank 0 creates directories.  A concurrent mkdir from every rank is
    # a known Lustre metadata race: the kernel on one node may not see a
    # directory created by another node's kernel even milliseconds later.
    if is_main(rank):
        save_dir.mkdir(parents=True, exist_ok=True)
        shard_dir.mkdir(parents=True, exist_ok=True)

    # GPU barrier: rank 0 has finished the mkdir before anyone proceeds.
    barrier(world_size, device)

    # Non-zero ranks: issue their own mkdir to force a fresh Lustre metadata
    # lookup.  This is a no-op once the directory is visible, but it prevents
    # the "No such file or directory" error on slow-propagating mounts.
    if not is_main(rank):
        shard_dir.mkdir(parents=True, exist_ok=True)

    if is_main(rank):
        print(f"\n{'═'*64}")
        print(f"  Inference Pipeline")
        print(f"{'═'*64}")
        print(f"  Config          : {args.config}")
        print(f"  Dataset         : {well_dataset_name}")
        print(f"  Checkpoint      : {ckpt_name}")
        print(f"  N_ensemble      : {N_ENSEMBLE}")
        print(f"  N_traj_per_file : {N_TRAJ_PER_FILE}  (-1 = all)")
        print(f"  World size      : {world_size}  |  Device : {device}")
        print(f"{'─'*64}\n")

    barrier(world_size, device)

    # ── Model — every rank loads its own copy ─────────────────────────────────
    model = FourierFlowModule.from_checkpoint(ckpt_path, map_location=device)
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

    # ── Trajectory index construction ─────────────────────────────────────────
    # dataset.metadata.n_steps_per_trajectory is indexed by FILE, not trajectory.
    # dataset.n_trajectories_per_file gives the per-file trajectory counts.
    # build_traj_index flattens these into a per-trajectory index list,
    # respecting the N_TRAJ_PER_FILE cap.
    steps_per_file  = dataset.metadata.n_steps_per_trajectory  # one entry per file
    n_traj_per_file = dataset.n_trajectories_per_file           # one entry per file

    traj_indices, n_steps_per_traj = build_traj_index(
        n_traj_per_file   = n_traj_per_file,
        steps_per_file    = steps_per_file,
        traj_per_file_cap = N_TRAJ_PER_FILE,
    )
    N_TRAJ = len(traj_indices)

    # ── Scalar condition names ────────────────────────────────────────────────
    scalar_names: list[str] = list(
        getattr(dataset.metadata, "constant_scalar_names", []) or []
    )

    # ── Ensemble split ────────────────────────────────────────────────────────
    n_ens_base      = N_ENSEMBLE // world_size
    n_ens_remainder = N_ENSEMBLE  % world_size
    n_ens_local     = n_ens_base + (1 if rank < n_ens_remainder else 0)

    if is_main(rank):
        print(f"  Scalar conditions : {scalar_names or '(none)'}")
        print(f"  N_TRAJ            : {N_TRAJ}  "
              f"({N_TRAJ_PER_FILE} traj/file × {len(steps_per_file)} files)")
        print(f"  N_STEPS_INPUT     : {N_STEPS_INPUT}")
        print(f"  NUM_SAMPLING      : {NUM_SAMPLING}")
        print(f"  ODE method        : {ODE_METHOD}")
        print(f"  Clamp sigma       : {CLAMP_SIGMA}")
        print(f"  N_ENSEMBLE        : {N_ENSEMBLE}  "
              f"({n_ens_local} member(s) on this rank)\n")

    # ── Inference loop ────────────────────────────────────────────────────────
    # Keyed by position i (0..N_TRAJ-1), NOT by raw dataset index.
    # This keeps labels aligned with the 0-based traj column in stats/psd CSVs.
    local_preds   = {}
    local_targets = {}
    local_labels  = {}

    t0 = time.perf_counter()

    with torch.inference_mode():
        for i, (traj_idx, n_steps) in enumerate(zip(traj_indices, n_steps_per_traj)):
            n_rollout = n_steps - N_STEPS_INPUT

            if is_main(rank):
                pct = (i + 1) / N_TRAJ * 100
                print(f"  [{pct:5.1f}%] traj {i:>3d}/{N_TRAJ}  "
                      f"(dataset_idx={traj_idx})  rollout_steps={n_rollout}")

            item          = dataset[traj_idx]
            x_full, label = get_well_inputs(item, device=device)   # (T,H,W,C), (C_cond,)

            local_labels[i] = label_to_dict(label, scalar_names)

            seed      = x_full[:N_STEPS_INPUT]
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
                n_ensemble       = 1,
                normalized_input = False,
                verbose          = cfg.get("verbose", False),
                ode_method       = ODE_METHOD,
                clamp_sigma      = CLAMP_SIGMA,
            )
            # (n_ens_local, 1, T, H, W, C) → (n_ens_local, T, H, W, C)
            preds_phys = preds_phys.squeeze(1)

            local_preds[i]   = preds_phys.cpu()
            local_targets[i] = x_full[N_STEPS_INPUT:].cpu()

            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

    elapsed = time.perf_counter() - t0
    print(f"  [rank {rank}] inference done in {elapsed:.1f}s  "
          f"({n_ens_local} member(s) x {N_TRAJ} trajectories)")

    # ── Write shard to shared filesystem ─────────────────────────────────────
    path = write_shard(
        {"preds": local_preds, "targets": local_targets, "labels": local_labels},
        shard_dir, rank,
    )
    print(f"  [rank {rank}] shard written → {path}")

    barrier(world_size, device)

    if not is_main(rank):
        if world_size > 1:
            dist.destroy_process_group()
        return

    # ── Rank 0: read shards ───────────────────────────────────────────────────
    print(f"\n  Reading {world_size} shard(s) from disk …")
    shards = read_all_shards(shard_dir, world_size)
    remove_shards(shard_dir, world_size)

    # ── Merge: cat ensemble shards per trajectory ─────────────────────────────
    merged_preds   = {}
    merged_targets = {}
    # Labels are identical across all ranks (same dataset view) — take shard 0.
    merged_labels: dict[int, dict[str, float]] = shards[0]["labels"]

    for shard in shards:
        for idx, arr in shard["preds"].items():
            merged_preds.setdefault(idx, []).append(arr)
        for idx, arr in shard["targets"].items():
            if idx not in merged_targets:
                merged_targets[idx] = arr

    all_indices = sorted(merged_preds.keys())

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
    meta_fields = dataset.metadata.field_names
    field_names = [item for sublist in meta_fields.values() for item in sublist]
    if is_main(rank):
        print(f"  field names   : {field_names}")

    # ── Labels DataFrame ──────────────────────────────────────────────────────
    # Tidy table: one row per trajectory, one column per scalar condition.
    # Keyed on "traj" (0-based position) to match build_stats_df / build_psd_df.
    labels_df = build_labels_df(merged_labels, all_indices)
    if is_main(rank) and not labels_df.empty:
        print(f"  scalar conditions sample:\n{labels_df.head()}\n")

    # ── 1. Full pickle ────────────────────────────────────────────────────────
    full_pkl = save_dir / f"{stem}_traj.pkl"
    with open(full_pkl, "wb") as fh:
        pickle.dump(
            {
                "preds":           preds_arr,      # (N_TRAJ, N_ENS, T, H, W, C)
                "targets":         targets_arr,    # (N_TRAJ,        T, H, W, C)
                "config":          cfg,
                "traj_indices":    all_indices,    # 0-based position indices
                "dataset_indices": traj_indices,   # raw dataset indices for provenance
                "field_names":     field_names,
                "labels":          merged_labels,
                "scalar_names":    scalar_names,
            },
            fh, protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"\n  [1/4] Full pickle        → {full_pkl}")

    # ── 2. Single-member pickle (small download for plotting) ─────────────────
    single_pkl = save_dir / f"{ckpt_base_name}_nsampling{NUM_SAMPLING}_nens1_traj.pkl"
    with open(single_pkl, "wb") as fh:
        pickle.dump(
            {
                "preds":           preds_arr[:, :1],   # (N_TRAJ, 1, T, H, W, C)
                "targets":         targets_arr,
                "config":          cfg,
                "traj_indices":    all_indices,
                "dataset_indices": traj_indices,
                "field_names":     field_names,
                "labels":          merged_labels,
                "scalar_names":    scalar_names,
            },
            fh, protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"  [2/4] Single-member pkl  → {single_pkl}")

    # ── 3 & 4. Stats and PSD DataFrames ──────────────────────────────────────
    print(f"\n  Rebuilding full truth tensor (Ti={N_STEPS_INPUT}) ...")

    seeds_list = []
    for i, traj_idx in zip(all_indices, traj_indices):
        item      = dataset[traj_idx]
        x_full, _ = get_well_inputs(item, device=torch.device("cpu"))
        seeds_list.append(x_full.numpy()[:N_STEPS_INPUT + min_T])

    truth_traj  = torch.from_numpy(np.stack(seeds_list, axis=0))
    preds_torch = torch.from_numpy(preds_arr)
    predictions = {ckpt_base_name: preds_torch}

    print(f"  Building stats DataFrame ...")
    stats_df = build_stats_df(
        truth       = truth_traj,
        predictions = predictions,
        field_names = field_names,
        Ti          = N_STEPS_INPUT,
        lead_times  = None,
    )
    if not labels_df.empty:
        stats_df = stats_df.merge(labels_df, on="traj", how="left")

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
        min_bin_count = 3,
    )
    if not labels_df.empty:
        psd_df = psd_df.merge(labels_df, on="traj", how="left")

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
