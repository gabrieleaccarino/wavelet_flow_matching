#!/usr/bin/env python
"""Regression check: rebuild the SUBMITTED paper's tables from the single-run
``*_stats.csv`` files with the same builders used for the camera-ready, and
compare cell-by-cell (3 decimals) with the values typeset in the paper
(``scripts/paper_reference_values.json``, extracted from main_camera_ready.tex).

Passing proves that the camera-ready (3-seed) tables use exactly the metric
definitions of the submission; only the data (1 run -> 3 seeds) changes.

Two quirks of the submission are reproduced explicitly (and NOT carried into
the camera-ready, which uses the inclusive horizon bounds everywhere):
  * Table 1 (and S9) were computed by figures.ipynb, whose horizon bins use
    exclusive upper bounds (lead time 32 falls in "33:64"); all appendix
    tables use inclusive bounds. Rebuilt here with the legacy bins, tol 1e-3.
  * WNO diverges on TRL; the paper prints inf where the CSV has no finite rows.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from waveletflux.analysis import catalog as C
from waveletflux.analysis import tables as T
from waveletflux.analysis.paths import CHECKS, ensure_artifact_dirs


def key_of(label):
    m = re.match(r"WFM[ -](1|3)(?:-scale)? (haar|db\d)", label)
    if m:
        return C.wfm_key(m.group(1) + "scale", m.group(2))
    if "pixel" in label: return "FM_data_space_v4.0"
    if "fourier" in label: return "FourierFlow_16modes_v4.0"
    return {"WDNO": "WDNO_db2", "WNO": "WNO_3scales_db2", "FNO": "FNO", "TFNO": "TFNO"}[label]


def main():
    ensure_artifact_dirs()
    # The submitted paper's FM_fourier column is the 32-mode model (Table S4:
    # modes (32, 32)); the camera-ready 3-seed runs use the 16-mode model.
    C.MODEL_REGISTRY["FourierFlow_16modes_v4.0"]["stats_key"] = "FourierFlow_32modes_v4.0"
    ref = json.loads((Path(__file__).parent / "paper_reference_values.json").read_text())
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
    rows, ok = [], True
    for lab, tab in ref.items():
        got = built[lab]
        for label, vals in tab.items():
            k = key_of(label)
            mine = got.loc[k].to_numpy(float) if k in got.index else np.full(len(vals), np.nan)
            for j, v in enumerate(vals):
                v = float(v)
                g = mine[j]
                if np.isinf(v):
                    match = not np.isfinite(g) or g > 1e6          # diverged rollout
                elif lab in tol:
                    match = np.isfinite(g) and abs(round(g, 3) - v) <= tol[lab]
                else:
                    match = np.isfinite(g) and round(g, 3) == round(v, 3)
                rows.append(dict(table=lab, row=label, col=j, paper=v, rebuilt=g, match=bool(match)))
    out = pd.DataFrame(rows)
    out.to_csv(CHECKS / "paper_reproduction.csv", index=False)
    summ = out.groupby("table")["match"].agg(cells="size", matched="sum")
    print(summ.to_string())
    bad = out[~out["match"]]
    if len(bad):
        print(f"\n{len(bad)} mismatching cells:")
        print(bad.to_string(index=False))
        sys.exit(1)
    print(f"\nall {len(out)} cells reproduced exactly.")


if __name__ == "__main__":
    main()
