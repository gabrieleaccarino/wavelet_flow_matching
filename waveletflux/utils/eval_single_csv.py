"""
eval_single_csv.py — one CSV per inference run: ML stats + physics metrics.

Drop this file in `waveletflux/utils/` next to `eval.py`, `physics.py`,
`physics_report.py`.

It replaces the old physics block in `inference_wavelet_fm.py` that wrote
`*_physics_global.csv`, `*_physics_fields.csv`, `*_physfields_stats.csv`,
the LaTeX table and the figures. This module writes NOTHING except one CSV.

Output contract
---------------
One long-format CSV, one row per (model, traj, field, step):

  * rows with field in {density, pressure, ux, uy, vorticity, ...}:
        ML stats from `build_stats_df` (rmse, nrmse, vrmse, crps, spread,
        coverage_*, w1, rmse_p/lp/c per band, ...)
        + per-field physics metrics from `build_physics_field_df`
        (mass_bias, gradsq_bias, w1_pdf, neg_frac_pred, ...)
  * rows with field == "_global_":
        coupled diagnostics from `build_physics_global_df`
        (ke_bias, enstrophy_bias, massflux_bias, divnorm_ratio /
        continuity_ratio, nusselt_bias, ...).
        Kept as separate rows (NOT broadcast across fields) so that any
        later mean/sem aggregation is not artificially duplicated.

Every row also carries: seed, dataset, n_ensemble, num_sampling_steps,
ode_method, clamp_sigma — so CSVs from different seeded runs can simply be
concatenated for the uncertainty analysis you will do afterwards.

Columns absent for a given row type are NaN. That is intentional: it keeps
a single tidy file that `physics_report.summarize` can still consume —
    summarize(df[df.field != "_global_"], metrics, by=("model","field"))
    summarize(df[df.field == "_global_"], metrics, by=("model",))
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import torch

MERGE_KEYS = ["model", "traj", "field", "step"]
GLOBAL_FIELD_LABEL = "_global_"


def evaluate_to_single_csv(
    truth:        torch.Tensor,               # (N_TRAJ, STEPS, H, W, C)
    predictions:  Mapping[str, torch.Tensor], # name -> (N_TRAJ, N_ENS, STEPS, H, W, C)
    field_names:  Sequence[str],
    spec,                                     # physics.DatasetSpec, grid attached
    params:       Mapping[int, dict],         # per-traj physical params
    save_path:    str | Path,
    seed:         int,
    *,
    dataset_name: str = "",
    Ti:           int = 0,
    lead_times:   list[int] | None = None,
    include_derived_channels: bool = True,    # vorticity/KE rows through build_stats_df
    skip_fields_physics: tuple[str, ...] = ("pressure",),
    emd_fields:   set[str] | None = None,
    run_meta:     Mapping[str, object] | None = None,
    verbose:      bool = True,
) -> pd.DataFrame:
    """
    Compute everything and write exactly ONE csv at `save_path`.

    `run_meta` is stamped onto every row, e.g.
        {"n_ensemble": 8, "num_sampling_steps": 50,
         "ode_method": "euler", "clamp_sigma": 5.0}
    """
    from waveletflux.utils.eval import build_stats_df          # ML stats
    from waveletflux.utils import physics as P                 # physics metrics

    t0 = time.perf_counter()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_traj = truth.shape[0]
    stats_parts, field_parts, global_parts = [], [], []

    for t in range(n_traj):
        tr  = truth[t : t + 1]
        pr  = {k: v[t : t + 1] for k, v in predictions.items()}
        par = {0: params.get(t, {})} if params else None

        # ── Layer 1: derived channels appended to the field axis ─────────────
        if include_derived_channels:
            tr_d, pr_d, names_d = P.derive_fields(
                tr, pr, list(field_names), spec, params=par)
        else:
            tr_d, pr_d, names_d = tr, pr, list(field_names)

        # ── ML stats on raw + derived channels ───────────────────────────────
        sdf = build_stats_df(
            tr_d, pr_d, names_d, Ti=Ti, lead_times=lead_times,
            emd_fields=emd_fields)
        sdf["traj"] = t
        stats_parts.append(sdf)

        # ── Layer 2: per-field physics metrics ───────────────────────────────
        fdf = P.build_physics_field_df(
            tr_d, pr_d, names_d, spec, Ti=Ti, lead_times=lead_times,
            params=par,
            fields=[f for f in names_d if f not in skip_fields_physics])
        fdf["traj"] = t
        field_parts.append(fdf)

        # ── Layer 3: global coupled diagnostics ──────────────────────────────
        gdf = P.build_physics_global_df(
            tr, pr, list(field_names), spec, Ti=Ti, lead_times=lead_times,
            params=par)
        gdf["traj"] = t
        global_parts.append(gdf)

        del tr_d, pr_d
        if verbose:
            print(f"    [EVAL] traj {t + 1}/{n_traj} done "
                  f"({time.perf_counter() - t0:.0f}s elapsed)", flush=True)

    stats_df  = pd.concat(stats_parts,  ignore_index=True)
    field_df  = pd.concat(field_parts,  ignore_index=True)
    global_df = pd.concat(global_parts, ignore_index=True)

    # ── Merge per-field physics INTO the stats rows ──────────────────────────
    # Outer merge keeps stats-only fields (e.g. pressure) and physics-only
    # combinations, if any. `_dup` suffix guards accidental column clashes.
    per_field = stats_df.merge(
        field_df, on=MERGE_KEYS, how="outer", suffixes=("", "_dup"))
    dup = [c for c in per_field.columns if c.endswith("_dup")]
    if dup:
        per_field = per_field.drop(columns=dup)

    # ── Global metrics as their own rows, field="_global_" ───────────────────
    global_df = global_df.copy()
    global_df["field"] = GLOBAL_FIELD_LABEL

    out = pd.concat([per_field, global_df], ignore_index=True, sort=False)

    # ── Stamp run metadata (seed first-class) ────────────────────────────────
    out["seed"] = int(seed)
    if dataset_name:
        out["dataset"] = dataset_name
    for k, v in (run_meta or {}).items():
        out[k] = v

    # stable column order: keys, metadata, then metrics alphabetically
    meta = ["seed"] + (["dataset"] if dataset_name else []) \
         + list((run_meta or {}).keys())
    metric_cols = sorted(c for c in out.columns if c not in MERGE_KEYS + meta)
    out = out[MERGE_KEYS + meta + metric_cols]

    out.to_csv(save_path, index=False)
    if verbose:
        n_g = (out["field"] == GLOBAL_FIELD_LABEL).sum()
        print(f"  [EVAL] single CSV → {save_path}  "
              f"({len(out):,} rows: {len(out) - n_g:,} per-field, "
              f"{n_g:,} global; {out.shape[1]} cols; "
              f"{time.perf_counter() - t0:.0f}s)", flush=True)
    return out
