"""
inference.py
------------
Distributed ensemble inference for WaveletFlowModule (and data-space variants),
with integrated physics-based evaluation.

Everything in the original script is preserved byte-for-byte except where
marked `# [PHYSICS]` or `# [SINGLE-CSV]`.

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

  CLI override:
    torchrun ... inference.py --config ... --override n_traj_per_file=3

Seeding
-------
  `seed` (int, default 0) and `run_tag` (str, default "seed<seed>") are plain
  config keys, overridable from the CLI:

    torchrun ... inference.py --config ... --override seed=3

  Each DDP rank derives a disjoint RNG stream from (seed, rank), so the four
  GPUs draw independent noise for their ensemble shards, and two jobs with
  different seeds share no stream. The seed is stamped both into the output
  filenames (via run_tag) and into a `seed` column of the metrics CSV, so
  CSVs from repeated runs can simply be concatenated for uncertainty
  quantification.

Performance reporting
---------------------
  Inference-only timing is measured with a barrier *before* the loop and
  another barrier *after* it, so the reported wall-clock time is the max
  across all ranks (i.e. the true end-to-end latency, not a local average).
  DataFrame construction, file I/O, and shard merging are excluded.

Outputs (all written by rank 0)
--------------------------------
  <stem>_traj.pkl          full predictions  (N_TRAJ, N_ENS, T, H, W, C)
  <stem>_nens1_traj.pkl    single member     (N_TRAJ,    1,  T, H, W, C)
  <stem>_perf.yaml         inference-only performance counters
  <stem>_metrics.csv       THE single metrics file:
                             rows with field != "_global_":
                               ML stats (rmse, vrmse, crps, spread, coverage,
                               w1, rmse_p/lp/c per band, ...) merged with
                               per-field physics metrics (mass_bias,
                               gradsq_bias, neg_frac_pred, ...) on
                               (model, traj, field, step); derived channels
                               (e.g. vorticity) appear as ordinary field rows
                               when physics.derived_ml is set.
                             rows with field == "_global_":
                               coupled diagnostics (ke_bias, enstrophy_bias,
                               massflux_bias, continuity/divnorm ratio, ...),
                               one row per (model, traj, step). Kept as
                               separate rows, NOT broadcast across fields, so
                               later mean/sem aggregation is not duplicated.
                             every row also carries: seed, dataset,
                               n_ensemble, num_sampling_steps, ode_method,
                               clamp_sigma, and the scalar conditions.
  <stem>_psd.csv           OPTIONAL, off by default (compute_psd=true to get it)

Usage — single device (MPS / CPU)
    python inference.py --config configs/trl2d/inference_trl2d.yaml

Usage — multi-GPU with torchrun
    torchrun --nproc_per_node=4 inference.py --config configs/trl2d/inference_trl2d.yaml

CLI overrides
    torchrun --nproc_per_node=4 inference.py \\
        --config configs/trl2d/inference_trl2d.yaml \\
        --override n_ensemble=16 num_sampling_steps=50 seed=2 verbose=true

[PHYSICS] CLI overrides
    --override physics.enabled=true physics.stride=2 physics.max_members=8
"""

from the_well.data.normalization import ZScoreNormalization
import torch.distributed as dist
from pathlib import Path
import numpy as np
import argparse
import pickle
import random
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
N_TRAJ_PER_FILE_DEFAULT: int = 1
# ─────────────────────────────────────────────────────────────────────────────


# [SINGLE-CSV] label for the global-diagnostics rows in the merged CSV
GLOBAL_FIELD_LABEL = "_global_"
MERGE_KEYS = ["model", "traj", "field", "step"]


# [PHYSICS] ───────────────────────────────────────────────────────────────────
# Physics-evaluation defaults. Override from YAML under a `physics:` block or
# with --override physics.<key>=<value>.
PHYSICS_DEFAULTS = {
    "enabled":         True,
    "global_metrics":  None,     # None -> physics.RECOMMENDED[dataset]["global"]
    "field_metrics":   None,     # None -> physics.RECOMMENDED[dataset]["field"]
    "continuity":      None,     # None -> physics.RECOMMENDED[dataset]["continuity"]
    "stride":          1,        # evaluate every n-th lead time
    "max_members":     None,     # cap the ensemble used for physics
    "derived_ml":      [],       # derived channels through build_stats_df
                                 # e.g. ["vorticity"]
    "scalars_are_log": True,     # `transform: [log_scalars]` stores ln(x)
}
# [SINGLE-CSV] removed: "derived_psd" and "make_figures" — this script writes
# exactly one CSV and no figures/tables.
# ─────────────────────────────────────────────────────────────────────────────


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
        help="Override any config key, e.g. --override n_ensemble=32 seed=3",
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


def seed_everything(base_seed: int, rank: int) -> int:
    """Deterministic per-rank seeding. Each DDP rank gets a disjoint stream,
    otherwise all GPUs would draw identical noise for their ensemble shards;
    the base-seed multiplier keeps streams of different seeds disjoint too."""
    seed = int(base_seed) * 100_003 + int(rank) * 1_009
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


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
    """
    peak_gib = (
        torch.cuda.max_memory_allocated(device) / 2**30
        if device.type == "cuda" else 0.0
    )

    if world_size > 1:
        t_tensor = torch.tensor(wall_s, dtype=torch.float64, device=device)
        dist.all_reduce(t_tensor, op=dist.ReduceOp.MAX)
        wall_max_s = t_tensor.item()

        s_tensor = torch.tensor(float(total_steps), dtype=torch.float64, device=device)
        dist.all_reduce(s_tensor, op=dist.ReduceOp.SUM)
        total_steps_g = int(s_tensor.item())

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
    """Return (dataset_indices, n_steps) for the selected trajectories."""
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
    """Convert a 1-D label tensor to a {name: float} dict."""
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
    """One row per trajectory, one column per scalar condition, keyed on `traj`."""
    rows = [{"traj": idx, **labels_dict.get(idx, {})} for idx in traj_indices]
    return pd.DataFrame(rows)


# [PHYSICS] ═══════════════════════════════════════════════════════════════════
# Physics-evaluation helpers
# ═════════════════════════════════════════════════════════════════════════════

def make_grid(dataset_name: str, Nx: int, Ny: int, item: dict | None = None):
    """
    Build the physical Grid for a dataset at the ACTUAL stored resolution.
    (unchanged from the original file)
    """
    from wavelet_flow_matching.utils.physics import Grid

    if item is not None:
        for key in ("space_grid", "coords", "grid", "x"):
            g = item.get(key) if isinstance(item, dict) else None
            if g is None:
                continue
            g = torch.as_tensor(g).squeeze()
            if g.ndim == 3 and g.shape[-1] == 2 and g.shape[:2] == (Nx, Ny):
                xs = g[:, 0, 0].double()
                ys = g[0, :, 1].double()
                periodic_y = dataset_name == "active_matter"
                print(f"  [PHYSICS] grid from item['{key}']: "
                      f"x in [{float(xs.min()):.3g}, {float(xs.max()):.3g}], "
                      f"y in [{float(ys.min()):.3g}, {float(ys.max()):.3g}]")
                return Grid.from_coords(xs, ys, periodic_x=True,
                                        periodic_y=periodic_y)

    if dataset_name == "active_matter":
        return Grid.uniform(10.0, Nx, 10.0, Ny, periodic_x=True, periodic_y=True)
    if dataset_name.startswith("rayleigh_benard"):
        if dataset_name.endswith("uniform"):
            return Grid.uniform(4.0, Nx, 1.0, Ny, periodic_x=True, periodic_y=False)
        return Grid.chebyshev_y(4.0, Nx, 1.0, Ny)
    if dataset_name.startswith("turbulent_radiative_layer"):
        return Grid.uniform(1.0, Nx, 3.0, Ny, x0=-0.5, y0=-1.0,
                            periodic_x=True, periodic_y=False)
    raise KeyError(
        f"no grid recipe for '{dataset_name}'. Add one to make_grid(), or pass "
        f"coordinates so Grid.from_coords can be used.")


def physical_params(
    labels_df:       pd.DataFrame,
    dataset_name:    str,
    scalars_are_log: bool,
) -> dict[int, dict[str, float]]:
    """
    Turn the scalar-condition table into the physical parameters the physics
    metrics need, keyed by trajectory. (unchanged from the original file)
    """
    out: dict[int, dict[str, float]] = {}
    if labels_df.empty:
        return out

    def _val(row, *names):
        for n in names:
            for col in row.index:
                if col.lower() == n.lower():
                    v = float(row[col])
                    return float(np.exp(v)) if scalars_are_log else v
        return None

    for _, row in labels_df.iterrows():
        traj = int(row["traj"])
        p: dict[str, float] = {}

        if dataset_name.startswith("rayleigh_benard"):
            Ra = _val(row, "Rayleigh", "rayleigh", "Ra")
            Pr = _val(row, "Prandtl", "prandtl", "Pr")
            if Ra is not None and Pr is not None:
                if Ra <= 0 or Pr <= 0:
                    print(f"  [PHYSICS] traj {traj}: Ra={Ra:g}, Pr={Pr:g} are not "
                          f"positive -- check physics.scalars_are_log. Skipping "
                          f"Nu / eps / Re for this trajectory.")
                else:
                    p["Ra"], p["Pr"] = Ra, Pr
                    p["kappa"] = (Ra * Pr) ** -0.5
                    p["nu"]    = (Ra / Pr) ** -0.5
                    p["Ly"], p["delta_b"] = 1.0, 1.0

        elif dataset_name.startswith("turbulent_radiative_layer"):
            tc = _val(row, "t_cool", "tcool", "cooling_time")
            if tc is not None and tc > 0:
                p["t_cool"] = tc
            elif tc is not None:
                print(f"  [PHYSICS] traj {traj}: t_cool={tc:g} is not positive -- "
                      f"check physics.scalars_are_log. Using t_cool=1.")
            p["t_cold"] = 10.0

        elif dataset_name == "active_matter":
            for key, names in (("alpha", ("alpha",)),
                               ("beta", ("beta",)),
                               ("zeta", ("zeta",))):
                v = _val(row, *names)
                if v is not None:
                    p[key] = v

        out[traj] = p
    return out


def check_denormalised(truth: torch.Tensor, field_names: list[str], spec) -> bool:
    """
    Loud guard against running the physics metrics on z-scored fields.
    (unchanged from the original file)
    """
    print("\n  [PHYSICS] denormalisation check")
    t = truth.double()
    stats = {}
    for i, name in enumerate(field_names):
        c = t[..., i]
        stats[name] = (float(c.mean()), float(c.std()),
                       float(c.min()), float(c.max()))

    ok = True
    for name in spec.positive_fields:
        if name in stats and stats[name][2] < 0:
            print(f"    FAIL  '{name}' has min={stats[name][2]:.4g} < 0 but must "
                  f"be strictly positive -> fields are NORMALISED.")
            ok = False

    if ok and all(abs(m) < 0.1 and abs(s - 1.0) < 0.2
                  for m, s, _, _ in stats.values()):
        print("    WARN  every channel has mean~0 and std~1; this looks z-scored.")
        ok = False

    for name, (m, s, lo, hi) in stats.items():
        print(f"    {name:<18s} mean={m:>11.4g}  std={s:>10.4g}  "
              f"range=[{lo:>10.4g}, {hi:>10.4g}]")

    print("    -> " + ("fields look PHYSICAL, proceeding."
                       if ok else
                       "REFUSING to compute physics metrics on normalised fields."))
    return ok


# [SINGLE-CSV] ────────────────────────────────────────────────────────────────
# run_physics_evaluation now RETURNS DataFrames instead of writing CSVs,
# LaTeX tables, or figures. The single merged CSV is written in main().
# ─────────────────────────────────────────────────────────────────────────────

def run_physics_evaluation(
    truth_traj:    torch.Tensor,          # (N_TRAJ, Ti+T, Nx, Ny, C)
    preds_torch:   torch.Tensor,          # (N_TRAJ, N_ENS, T, Nx, Ny, C)
    field_names:   list[str],
    dataset_name:  str,
    model_name:    str,
    Ti:            int,
    labels_df:     pd.DataFrame,
    pcfg:          dict,
    item:          dict | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Streaming physics evaluation, one trajectory at a time.

    Returns
    -------
    (field_df, global_df, derived_stats_df) — any of which may be None.
      field_df         : per-(model, traj, field, step) physics metrics
      global_df        : per-(model, traj, step) coupled diagnostics
      derived_stats_df : ML stats (build_stats_df) on derived channels only
                         (e.g. vorticity), same schema as the main stats_df
    """
    from wavelet_flow_matching.utils.eval import build_stats_df
    from wavelet_flow_matching.utils import physics as P

    Nx, Ny = truth_traj.shape[2], truth_traj.shape[3]
    grid = make_grid(dataset_name, Nx, Ny, item=item)
    spec = P.spec_for(dataset_name).with_grid(grid)
    rec  = P.recommended_for(dataset_name)

    g_metrics = pcfg["global_metrics"] if pcfg["global_metrics"] is not None \
        else rec["global"]
    f_metrics = pcfg["field_metrics"] if pcfg["field_metrics"] is not None \
        else rec["field"]
    do_cont = pcfg["continuity"] if pcfg["continuity"] is not None \
        else rec["continuity"]

    # ── role resolution ──────────────────────────────────────────────────────
    idx = spec.resolve(field_names)
    print(f"\n  [PHYSICS] dataset : {dataset_name}")
    print(f"  [PHYSICS] fields  : {field_names}")
    print(f"  [PHYSICS] roles   : {idx}")
    print(f"  [PHYSICS] global  : {list(g_metrics)}"
          f"{' + continuity' if do_cont else ''}")
    print(f"  [PHYSICS] field   : {list(f_metrics)}")
    if "ux" not in idx or "uy" not in idx:
        print("  [PHYSICS] ABORT: velocity roles unresolved. Use "
              f"spec.with_roles(ux=..., uy=...) to map onto {field_names}.")
        return None, None, None

    # ── denormalisation guard ────────────────────────────────────────────────
    if not check_denormalised(truth_traj[:1, Ti:Ti + 4], field_names, spec):
        return None, None, None

    # ── physical parameters per trajectory ───────────────────────────────────
    params = physical_params(labels_df, dataset_name, pcfg["scalars_are_log"])
    if params:
        k0 = sorted(params)[0]
        print(f"  [PHYSICS] params[{k0}] = "
              f"{ {k: float(f'{v:.6g}') for k, v in params[k0].items()} }")
        if pcfg["scalars_are_log"]:
            print("  [PHYSICS] scalar conditions EXPONENTIATED "
                  "(transform: log_scalars stores ln(x)). Set "
                  "physics.scalars_are_log=false if yours are not logged.")

    # ── lead times and ensemble cap ──────────────────────────────────────────
    T_avail = min(truth_traj.shape[1] - Ti, preds_torch.shape[2])
    lead_times = list(range(0, T_avail, max(1, int(pcfg["stride"]))))
    n_ens = preds_torch.shape[1]
    m_cap = min(int(pcfg["max_members"] or n_ens), n_ens)
    print(f"  [PHYSICS] lead times: {len(lead_times)} of {T_avail} "
          f"(stride={pcfg['stride']}) | members: {m_cap} of {n_ens}")

    field_dfs, global_dfs, ml_dfs = [], [], []
    t0 = time.perf_counter()

    for t in range(truth_traj.shape[0]):
        tr = truth_traj[t:t + 1]
        pr = {model_name: preds_torch[t:t + 1, :m_cap]}
        par = {0: params.get(t, {})}

        # Layer 3 -- global coupled diagnostics
        if g_metrics or do_cont:
            gdf = P.build_physics_global_df(
                tr, pr, field_names, spec, Ti=Ti, lead_times=lead_times,
                params=par, metrics=g_metrics, continuity=do_cont)
            gdf["traj"] = t
            global_dfs.append(gdf)

        # Layer 1 + 2 -- derived channels and per-field metrics
        if pcfg["derived_ml"] or f_metrics:
            derived = list(pcfg["derived_ml"])
            if derived:
                tr_d, pr_d, names_d = P.derive_fields(
                    tr, pr, field_names, spec, derived=derived, params=par)
            else:
                tr_d, pr_d, names_d = tr, pr, list(field_names)

            if derived:
                mdf = build_stats_df(tr_d, pr_d, names_d, Ti=Ti,
                                     lead_times=lead_times)
                mdf["traj"] = t
                ml_dfs.append(mdf[mdf["field"].isin(derived)])

            if f_metrics:
                fdf = P.build_physics_field_df(
                    tr_d, pr_d, names_d, spec, Ti=Ti, lead_times=lead_times,
                    params=par, metrics=f_metrics,
                    fields=[f for f in names_d if f != "pressure"])
                fdf["traj"] = t
                field_dfs.append(fdf)

            if derived:
                del tr_d, pr_d

        print(f"    [PHYSICS] traj {t + 1}/{truth_traj.shape[0]} "
              f"({time.perf_counter() - t0:.0f}s)")

    field_df  = pd.concat(field_dfs,  ignore_index=True) if field_dfs  else None
    global_df = pd.concat(global_dfs, ignore_index=True) if global_dfs else None
    ml_df     = pd.concat(ml_dfs,     ignore_index=True) if ml_dfs     else None
    print(f"  [PHYSICS] evaluation time: {time.perf_counter() - t0:.0f}s")
    return field_df, global_df, ml_df


# [SINGLE-CSV] ────────────────────────────────────────────────────────────────

def merge_to_single_csv(
    stats_df:    pd.DataFrame,
    field_df:    pd.DataFrame | None,
    global_df:   pd.DataFrame | None,
    derived_df:  pd.DataFrame | None,
    labels_df:   pd.DataFrame,
    run_meta:    dict,
    save_path:   Path,
) -> pd.DataFrame:
    """
    Assemble THE one CSV.

      1. stats rows (raw fields) + derived-channel stats rows are concatenated
         (same schema, disjoint `field` values);
      2. per-field physics metrics are merged in on (model, traj, field, step);
      3. global diagnostics are appended as rows with field == "_global_"
         (NOT broadcast across fields — keeps later sem over trajectories honest);
      4. scalar conditions are merged on `traj`;
      5. run metadata (seed first) is stamped on every row.
    """
    parts = [stats_df]
    if derived_df is not None and not derived_df.empty:
        parts.append(derived_df)
    per_field = pd.concat(parts, ignore_index=True, sort=False)

    if field_df is not None and not field_df.empty:
        per_field = per_field.merge(
            field_df, on=MERGE_KEYS, how="outer", suffixes=("", "_dup"))
        dup = [c for c in per_field.columns if c.endswith("_dup")]
        if dup:
            per_field = per_field.drop(columns=dup)

    frames = [per_field]
    if global_df is not None and not global_df.empty:
        gdf = global_df.copy()
        gdf["field"] = GLOBAL_FIELD_LABEL
        frames.append(gdf)

    out = pd.concat(frames, ignore_index=True, sort=False)

    if not labels_df.empty:
        out = out.merge(labels_df, on="traj", how="left")

    for k, v in run_meta.items():
        out[k] = v

    meta_cols   = list(run_meta.keys())
    label_cols  = [c for c in labels_df.columns if c != "traj"]
    metric_cols = sorted(c for c in out.columns
                         if c not in MERGE_KEYS + meta_cols + label_cols)
    out = out[MERGE_KEYS + meta_cols + label_cols + metric_cols]

    out.to_csv(save_path, index=False)
    n_g = int((out["field"] == GLOBAL_FIELD_LABEL).sum())
    print(f"  [SINGLE-CSV] → {save_path}")
    print(f"  [SINGLE-CSV]   {len(out):,} rows "
          f"({len(out) - n_g:,} per-field, {n_g:,} global) × "
          f"{out.shape[1]} cols")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, args.override or [])

    # [PHYSICS] merge physics config with defaults
    pcfg = {**PHYSICS_DEFAULTS, **(cfg.get("physics") or {})}

    project_root = cfg["project_root"]
    sys.path.insert(0, project_root)

    # Late imports — project_root must be on sys.path first
    from wavelet_flow_matching.utils.well_data import get_well_dataset, get_well_inputs
    from wavelet_flow_matching.utils.eval import build_stats_df, build_psd_df
    from wavelet_flow_matching.modules.wavelet_fm import WaveletFlowModule

    # ── Distributed init ──────────────────────────────────────────────────────
    rank, world_size, device = init_distributed()

    # [SINGLE-CSV] cfg is a dict — cfg["seed"], not cfg.seed
    BASE_SEED = int(cfg.get("seed", 0))
    RUN_TAG   = str(cfg.get("run_tag", f"seed{BASE_SEED}"))
    effective_seed = seed_everything(BASE_SEED, rank)
    print(f"  [rank {rank}] base_seed={BASE_SEED} effective_seed={effective_seed}")

    # ── Experiment identifiers ────────────────────────────────────────────────
    well_base_path    = cfg["well_base_path"]
    well_dataset_name = cfg["well_dataset_name"]
    ckpt_name         = cfg["ckpt_name"]
    N_STEPS_INPUT     = cfg["n_steps_input"]
    NUM_SAMPLING      = cfg["num_sampling_steps"]
    N_ENSEMBLE        = cfg["n_ensemble"]   # M: ensemble members (independent realizations) per initial condition
    ODE_METHOD        = cfg.get("ode_method", "euler")
    CLAMP_SIGMA       = cfg.get("clamp_sigma", 5.0)
    N_TRAJ_PER_FILE   = cfg.get("n_traj_per_file", N_TRAJ_PER_FILE_DEFAULT)

    ckpt_base_name = re.split(r"_\d{8}_", ckpt_name)[0]
    if ckpt_base_name == ckpt_name:
        ckpt_base_name = Path(ckpt_name).stem

    ckpt_path = os.path.join(
        project_root, "experiments", well_dataset_name,
        ckpt_base_name, "artifacts/checkpoints", ckpt_name,
    )

    stem = f"{ckpt_base_name}_nsampling{NUM_SAMPLING}_nens{N_ENSEMBLE}_{RUN_TAG}"

    save_dir = Path(project_root) / "experiments" / well_dataset_name / "results"

    job_id = (
        os.environ.get("PBS_JOBID")
        or os.environ.get("TORCHELASTIC_RUN_ID")
        or os.environ.get("MASTER_PORT")
        or str(os.getpid())
    )
    shard_dir = save_dir / f"_shards_{stem}_{job_id}"

    if is_main(rank):
        save_dir.mkdir(parents=True, exist_ok=True)
        shard_dir.mkdir(parents=True, exist_ok=True)

    barrier(world_size, device)

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
        print(f"  Seed / tag      : {BASE_SEED} / {RUN_TAG}")
        print(f"  World size      : {world_size}  |  Device : {device}")
        print(f"  Physics metrics : {pcfg['enabled']}  "
              f"(stride={pcfg['stride']}, max_members={pcfg['max_members']})")
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

    # ── Trajectory index construction ─────────────────────────────────────────
    steps_per_file  = dataset.metadata.n_steps_per_trajectory
    n_traj_per_file = dataset.n_trajectories_per_file

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
    local_preds   = {}
    local_targets = {}
    local_labels  = {}

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    total_steps = 0
    H = W = 1

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

            seed_frames = x_full[:N_STEPS_INPUT]
            seed_exp  = seed_frames.unsqueeze(0).expand(
                n_ens_local, -1, -1, -1, -1).contiguous()
            label_exp = (
                label.unsqueeze(0).expand(n_ens_local, -1).contiguous()
                if label is not None else None
            )

            _, preds_phys = model.rollout(
                seed             = seed_exp,
                scalar_conds     = label_exp,
                rollout_steps    = n_rollout,
                n_steps_input    = N_STEPS_INPUT,
                n_ensemble       = 1,   # members are already stacked along the batch dim (n_ens_local copies of the seed)
                normalized_input = False,
                verbose          = cfg.get("verbose", False),
                ode_method       = ODE_METHOD,
                clamp_sigma      = CLAMP_SIGMA,
            )
            preds_phys = preds_phys.squeeze(1)

            H, W         = x_full.shape[-3], x_full.shape[-2]
            total_steps += n_ens_local * n_rollout

            local_preds[i]   = preds_phys.cpu()
            local_targets[i] = x_full[N_STEPS_INPUT:].cpu()

            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "mps":
                torch.mps.empty_cache()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_s = time.perf_counter() - t_start

    barrier(world_size, device)

    perf_yaml_path = save_dir / f"{stem}_perf.yaml"
    report_perf(wall_s, total_steps, H, W, world_size, device, perf_yaml_path, rank)

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

    merged_preds   = {}
    merged_targets = {}
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

    preds_arr   = torch.stack(preds_list, dim=0).numpy()
    targets_arr = torch.stack(tgts_list,  dim=0).numpy()

    print(f"  preds shape   : {preds_arr.shape}")
    print(f"  targets shape : {targets_arr.shape}")

    assert preds_arr.shape[1] == N_ENSEMBLE, (
        f"Expected {N_ENSEMBLE} ensemble members, got {preds_arr.shape[1]}. "
        f"Check n_ensemble vs world_size."
    )

    # ── Field names ───────────────────────────────────────────────────────────
    meta_fields = dataset.metadata.field_names
    field_names = [item for sublist in meta_fields.values() for item in sublist]
    print(f"  field names   : {field_names}")

    # ── Labels DataFrame ──────────────────────────────────────────────────────
    labels_df = build_labels_df(merged_labels, all_indices)
    if not labels_df.empty:
        print(f"  scalar conditions sample:\n{labels_df.head()}\n")

    # ── 1. Full pickle ────────────────────────────────────────────────────────
    full_pkl = save_dir / f"{stem}_traj.pkl"
    with open(full_pkl, "wb") as fh:
        pickle.dump(
            {
                "preds":           preds_arr,
                "targets":         targets_arr,
                "config":          cfg,
                "seed":            BASE_SEED,
                "traj_indices":    all_indices,
                "dataset_indices": traj_indices,
                "field_names":     field_names,
                "labels":          merged_labels,
                "scalar_names":    scalar_names,
            },
            fh, protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"\n  [1/4] Full pickle        → {full_pkl}")

    # ── 2. Single-member pickle ───────────────────────────────────────────────
    # [SINGLE-CSV] run_tag added so seeded runs never overwrite each other
    single_pkl = save_dir / (
        f"{ckpt_base_name}_nsampling{NUM_SAMPLING}_nens1_{RUN_TAG}_traj.pkl")
    with open(single_pkl, "wb") as fh:
        pickle.dump(
            {
                "preds":           preds_arr[:, :1],
                "targets":         targets_arr,
                "config":          cfg,
                "seed":            BASE_SEED,
                "traj_indices":    all_indices,
                "dataset_indices": traj_indices,
                "field_names":     field_names,
                "labels":          merged_labels,
                "scalar_names":    scalar_names,
            },
            fh, protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"  [2/4] Single-member pkl  → {single_pkl}")

    # ── 3. Performance report ─────────────────────────────────────────────────
    print(f"  [3/4] Perf report        → {perf_yaml_path}")

    # ── 4. Single metrics CSV: ML stats + physics, one file ──────────────────
    print(f"\n  Rebuilding full truth tensor (Ti={N_STEPS_INPUT}) ...")

    seeds_list = []
    last_item = None
    for i, traj_idx in zip(all_indices, traj_indices):
        last_item = dataset[traj_idx]
        x_full, _ = get_well_inputs(last_item, device=torch.device("cpu"))
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

    # [SINGLE-CSV] physics evaluation returns DataFrames; nothing written yet
    field_df = global_df = derived_df = None
    if pcfg["enabled"]:
        print(f"\n{'─'*64}")
        print(f"  Physics evaluation")
        print(f"{'─'*64}")
        try:
            field_df, global_df, derived_df = run_physics_evaluation(
                truth_traj   = truth_traj,
                preds_torch  = preds_torch,
                field_names  = field_names,
                dataset_name = well_dataset_name,
                model_name   = ckpt_base_name,
                Ti           = N_STEPS_INPUT,
                labels_df    = labels_df,
                pcfg         = pcfg,
                item         = last_item,
            )
        except Exception as exc:
            import traceback
            print(f"  [PHYSICS] FAILED: {exc}")
            traceback.print_exc()
            print("  [PHYSICS] continuing: the CSV will contain ML stats only.")
    else:
        print(f"  Physics evaluation : disabled "
              f"(--override physics.enabled=true)")

    run_meta = {
        "seed":               BASE_SEED,
        "dataset":            well_dataset_name,
        "n_ensemble":         N_ENSEMBLE,
        "num_sampling_steps": NUM_SAMPLING,
        "ode_method":         ODE_METHOD,
        "clamp_sigma":        CLAMP_SIGMA,
    }
    metrics_csv = save_dir / f"{stem}_metrics.csv"
    merge_to_single_csv(
        stats_df, field_df, global_df, derived_df,
        labels_df, run_meta, metrics_csv)
    print(f"  [4/4] Metrics CSV        → {metrics_csv}")

    # ── Optional: full long-format PSD table (OFF by default) ────────────────
    # [SINGLE-CSV] default flipped to False so the run produces exactly one
    # CSV. The per-band spectral errors (rmse_p/lp/c_*) are in the metrics CSV
    # regardless — only the full k-resolved PSD table is skipped.
    if cfg.get("compute_psd", False):
        print(f"  Building PSD DataFrame (compute_psd=true) ...")
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
        print(f"  PSD CSV (optional)       → {psd_csv}  ({len(psd_df):,} rows)")

    print(f"\n{'═'*64}")
    print(f"  Done.  All outputs in:")
    print(f"  {save_dir}")
    print(f"{'═'*64}\n")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
