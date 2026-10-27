"""Regression check against the SUBMITTED paper.

Rebuilds the submission's tables from the single-run ``*_stats.csv`` files with
the same builders used for the camera-ready, and compares them cell by cell
(3 decimals) with the values typeset in the paper
(``paper_reference_values.json``, extracted from main_camera_ready.tex).
All cells matching proves that the camera-ready (3-seed) tables use exactly
the metric definitions of the submission; only the data (1 run -> 3 seeds)
changes.

Two quirks of the submission are reproduced explicitly here, and NOT carried
into the camera-ready, which uses inclusive horizon bounds everywhere:
  * Table 1 (and the submission's Table S9, since removed) were computed by the figures notebook, whose horizon bins
    use exclusive upper bounds (lead time 32 falls in "33:64"); rebuilt with
    those legacy bins, tolerance 1e-3.
  * The submission's FM_fourier is the 32-mode model (Table S4).
  * WNO diverges on TRL: the paper prints inf where the CSV has no finite rows.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import catalog as C
from . import tables as T
from .paths import CHECKS, ensure_artifact_dirs

REFERENCE = Path(__file__).with_name("paper_reference_values.json")


def _key_of(label):
    m = re.match(r"WFM[ -](1|3)(?:-scale)? (haar|db\d)", label)
    if m:
        return C.wfm_key(m.group(1) + "scale", m.group(2))
    if "pixel" in label:
        return "FM_data_space_v4.0"
    if "fourier" in label:
        return "FourierFlow_32modes_v4.0"
    return {"WDNO": "WDNO_db2", "WNO": "WNO_3scales_db2", "FNO": "FNO", "TFNO": "TFNO"}[label]


def paper_reproduction(save=True) -> pd.DataFrame:
    """Return one row per checked cell (paper value, rebuilt value, match)."""
    ref = json.loads(REFERENCE.read_text())
    df, _ = C.load(source="stats", verbose=False)

    one = lambda b, d=df, **kw: T.across_seeds(b, d, **kw)[0]
    legacy = df.copy()
    edges = {p: [ub for ub, _ in e] for p, e in C.HORIZON_EDGES.items()}
    legacy["horizon"] = [C.HORIZONS[p][next((i for i, ub in enumerate(edges[p]) if s < ub), 2)]
                         for p, s in zip(legacy["pde"], legacy["step"])]
    tol = {"tab:avg_sc_rmse_gen_wavelets": 1e-3 + 1e-9}
    built = {
        "tab:avg_sc_rmse_gen_wavelets": one(T.selection_table, d=legacy, models=C.ROWS_GEN, time_agg="horizon"),
        "tab:ablation_trl_crps_tcool_main": one(T.param_table, pde="turbulent_radiative_layer_2D",
                                                param="tcool", models=C.ROWS_GEN).iloc[:, :3],
        "tab:ablation_rb_crps_ra_main": one(T.param_table, pde="rayleigh_benard", param="Rayleigh",
                                            models=C.ROWS_GEN),
        "si-tab:appendix_vrmse": one(T.horizon_table, metric="vrmse", models=C.ROWS_ALL),
        "tab:appendix_crps": one(T.horizon_table, metric="crps", models=C.ROWS_GEN),
        "tab:appendix_spectral": one(T.bands_table, models=C.ROWS_GEN, time_agg="horizon"),
    }
    rows = []
    for lab, tab in ref.items():
        got = built[lab]
        for label, vals in tab.items():
            k = _key_of(label)
            mine = got.loc[k].to_numpy(float) if k in got.index else np.full(len(vals), np.nan)
            for j, v in enumerate(vals):
                v, g = float(v), mine[j]
                if np.isinf(v):
                    match = not np.isfinite(g) or g > 1e6          # diverged rollout
                elif lab in tol:
                    match = bool(np.isfinite(g) and abs(round(g, 3) - v) <= tol[lab])
                else:
                    match = bool(np.isfinite(g) and round(g, 3) == round(v, 3))
                rows.append(dict(table=lab, row=label, col=j, paper=v, rebuilt=g, match=bool(match)))
    out = pd.DataFrame(rows)
    if save:
        ensure_artifact_dirs()
        out.to_csv(CHECKS / "paper_reproduction.csv", index=False)
    return out


def summary(out: pd.DataFrame) -> pd.DataFrame:
    return out.groupby("table")["match"].agg(cells="size", matched="sum")
