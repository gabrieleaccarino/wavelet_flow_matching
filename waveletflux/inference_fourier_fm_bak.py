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

Performance reporting
---------------------
  Inference-only timing is measured with a barrier *before* the loop and
  another barrier *after* it, so the reported wall-clock time is the max
  across all ranks (i.e. the true end-to-end latency, not a local average).
  DataFrame construction, file I/O, and shard merging are excluded.

  Metrics written to <stem>_perf.yaml (rank 0 only):
    wall_clock_s          true wall-clock inference time  (max across ranks)
    throughput_steps_s    total rollout-steps generated globally / wall_clock_s
    throughput_frames_s   total frames  (step × H × W)  generated globally / wall_clock_s
    gpu_peak_mem_GiB_*    per-rank peak GPU memory allocated during inference

Outputs (all written by rank 0)
--------------------------------
  <stem>_traj.pkl       full predictions  (N_TRAJ, N_ENS, T, H, W, C)
  <stem>_nens1_traj.pkl single member     (N_TRAJ,    1,  T, H, W, C)  for plotting
  <stem>_stats.csv      per-(model, traj, field, step) scalar metrics
                        + one column per scalar condition (e.g. Rayleigh, Prandtl)
  <stem>_psd.csv        isotropic PSD in long format
                        + one column per scalar condition
  <stem>_perf.yaml      inference-only performance counters (see above)

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
# Performance reporting
# ─────────────────────────────────────────────────────────────────────────────

def report_perf(
    wall_s:      float,
    total_steps: int,
    H:           int,
    W:           int,
    world_size:  int,
    device:      torch.device,
    save_path:   Path,
    rank:        int,
) -> dict:
    """
    Reduce timing/memory stats across ranks and write a YAML report (rank 0).

    wall_s and total_steps are LOCAL values; the following reductions happen:
      - wall_clock_s        : MAX across ranks  (true end-to-end latency)
      - total_steps_global  : SUM across ranks  (all ensemble members × rollout steps)
      - gpu_peak_mem_GiB    : gathered per-rank (reported individually)

    Parameters
    ----------
    wall_s       : local wall-clock time from perf_counter (seconds)
    total_steps  : n_ens_local × sum(n_rollout per trajectory) on this rank
    H, W         : spatial dimensions of the field (for frame throughput)
    world_size   : number of distributed ranks
    device       : local torch device
    save_path    : where to write the YAML report (rank 0 only)
    rank         : this rank's index
    """
    # Peak GPU memory on this rank (bytes → GiB); 0 on CPU/MPS
    peak_gib = (
        torch.cuda.max_memory_allocated(device) / 2**30
        if device.type == "cuda" else 0.0
    )

    if world_size > 1:
        # Max wall time across ranks — reflects slowest rank (true latency)
        t_tensor = torch.tensor(wall_s, dtype=torch.float64, device=device)
        dist.all_reduce(t_tensor, op=dist.ReduceOp.MAX)
        wall_max_s = t_tensor.item()

        # Sum steps across ranks — total work done by the whole job
        s_tensor = torch.tensor(float(total_steps), dtype=torch.float64, device=device)
        dist.all_reduce(s_tensor, op=dist.ReduceOp.SUM)
        total_steps_g = int(s_tensor.item())

        # Per-rank peak GPU memory (for load-balance diagnostics)
        mem_tensor = torch.tensor(peak_gib, dtype=torch.float64, device=device)
        mem_all = [torch.zeros_like(mem_tensor) for _ in range(world_size)]
        dist.all_gather(mem_all, mem_tensor)
        peak_per_rank = {f"rank_{r}": round(m.item(), 3) for r, m in enumerate(mem_all)}
    else:
        wall_max_s    = wall_s
        total_steps_g = total_steps
        peak_per_rank = {"rank_0": round(peak_gib, 3)}

    total_frames_g = total_steps_g * H * W
    report = {
        "wall_clock_s":        round(wall_max_s, 2),
        "throughput_steps_s":  round(total_steps_g / wall_max_s, 1) if wall_max_s > 0 else 0,
        "throughput_frames_s": round(total_frames_g / wall_max_s, 1) if wall_max_s > 0 else 0,
        "total_steps_global":  total_steps_g,
        "total_frames_global": total_frames_g,
        "spatial_size_HW":     H * W,
        "gpu_peak_mem_GiB":    peak_per_rank,
    }

    if rank == 0:
        with open(save_path, "w") as fh:
            yaml.dump(report, fh, default_flow_style=False, sort_keys=False)
        print(
            f"\n  ── Inference timing ───────────────────────────────────────\n"
            f"  Wall clock (max across ranks) : {report['wall_clock_s']:.1f} s\n"
            f"  Throughput                    : "
            f"{report['throughput_steps_s']:,.0f} steps/s  |  "
            f"{report['throughput_frames_s']:,.0f} frames/s\n"
            f"  Total steps generated (global): {report['total_steps_global']:,}\n"
            f"  Spatial size H×W              : {report['spatial_size_HW']:,}\n"
            f"  Peak GPU memory (allocated)   : {peak_per_rank}\n"
            f"  Perf report → {save_path}\n"
            f"  ───────────────────────────────────────────────────────────\n"
        )
    return report


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

    # ── Reset GPU peak-memory stats before the clock starts ───────────────────
    # Do this BEFORE the barrier so the reset is not counted in inference time.
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    # Running counters for the perf report — accumulated inside the loop so
    # they are always correct even if an exception is raised mid-way.
    total_steps = 0   # n_ens_local × Σ n_rollout across trajectories
    H = W = 1         # spatial dimensions, set on the first trajectory

    # ── Synchronise all ranks before starting the clock ───────────────────────
    # This ensures rank-0 setup overhead (prints, mkdir, etc.) is excluded from
    # the timing, and that all ranks start counting at the same moment.
    barrier(world_size, device)
    t_start = time.perf_counter()

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

            # ── Accumulate perf counters per trajectory ───────────────────────
            # Done immediately after rollout so the count is correct even if
            # a later trajectory raises an exception.
            H, W         = x_full.shape[-3], x_full.shape[-2]
            total_steps += n_ens_local * n_rollout

            local_preds[i]   = preds_phys.cpu()
            local_targets[i] = x_full[N_STEPS_INPUT:].cpu()

            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

    # ── Stop clock — synchronise CUDA kernels before reading perf_counter ─────
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_s = time.perf_counter() - t_start

    # ── Barrier before reduction so wall_max reflects the slowest rank ────────
    barrier(world_size, device)

    # ── Gather and report performance across all ranks ────────────────────────
    perf_yaml_path = save_dir / f"{stem}_perf.yaml"
    report_perf(wall_s, total_steps, H, W, world_size, device, perf_yaml_path, rank)

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
    print(f"\n  [1/5] Full pickle        → {full_pkl}")

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
    print(f"  [2/5] Single-member pkl  → {single_pkl}")

    # ── 3. Performance report (already written by report_perf) ────────────────
    print(f"  [3/5] Perf report        → {perf_yaml_path}")

    # ── 4 & 5. Stats and PSD DataFrames ──────────────────────────────────────
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
    print(f"  [4/5] Stats CSV          → {stats_csv}  ({len(stats_df):,} rows)")

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
    print(f"  [5/5] PSD CSV            → {psd_csv}  ({len(psd_df):,} rows)")

    print(f"\n{'═'*64}")
    print(f"  Done.  All outputs in:")
    print(f"  {save_dir}")
    print(f"{'═'*64}\n")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()