"""Initial-condition perturbation study (rebuttal, reviewer jBaL Q3).

WFM 3-scale (haar) on TRL; Gaussian noise of std eta * sigma_c added to the
initial conditions before the autoregressive rollout. Runs are produced by
``wavelet_flow_matching/inference_wavelet_fm_ic_pert.py`` (``scripts/inference_ic_pert.pbs``)
and renamed to ``WFM_3scale_haar_<pde>_ic0p<eta>_seed<S>_metrics.csv``.

Aggregation: mean over (trajectories, fields, steps in window) within each
(eta, seed), then mean ± std across seeds.
"""
from __future__ import annotations

import glob
import re

import pandas as pd

from . import catalog as C
from .paths import metrics_dir

PDE = "turbulent_radiative_layer_2D"
_ETA_RX = re.compile(r"_ic0p(\d+)_")
MODEL = "WFM_3scale_haar"


def _eta(token: str) -> float:
    # 0 -> 0.0, 001 -> 0.001, 01 -> 0.01, 1 -> 0.1
    return 0.0 if token == "0" else float("0." + token)


def load_ic(pde=PDE) -> pd.DataFrame:
    files = sorted(glob.glob(str(metrics_dir(pde) / f"{MODEL}_{pde}_ic0p*_seed*_metrics.csv")))
    if not files:
        raise FileNotFoundError(f"no IC-perturbation files in {metrics_dir(pde)}")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        eta = _eta(_ETA_RX.search(f).group(1))
        if "ic_noise" in d.columns:
            assert abs(d["ic_noise"].iloc[0] - eta) < 1e-12, (f, d["ic_noise"].iloc[0], eta)
        d["ic_noise"] = eta
        d["pde"] = pde
        frames.append(d[d["field"] != C.GLOBAL_FIELD])
    df = C.add_features(pd.concat(frames, ignore_index=True))
    return df


def _per_seed(df, col, horizon=None):
    sub = df if horizon is None else df[df["horizon"] == horizon]
    return sub.groupby(["ic_noise", "seed"])[col].mean()


def ic_tables(df):
    """Returns {"crps": (mean, std), "spectral": (mean, std)} indexed by eta.
    CRPS per rollout horizon; spectral coherence RMSE per band (step mean)."""
    out = {}
    cols = {}
    for h in C.HORIZONS[PDE]:
        cols[h] = _per_seed(df, "crps", h)
    crps = pd.DataFrame(cols)
    out["crps"] = (crps.groupby("ic_noise").mean(), crps.groupby("ic_noise").std(ddof=1))
    spec = pd.DataFrame({C.BAND_COLS[c]: _per_seed(df, c) for c in C.BAND_COLS})
    out["spectral"] = (spec.groupby("ic_noise").mean(), spec.groupby("ic_noise").std(ddof=1))
    out["n_seeds"] = df.groupby("ic_noise")["seed"].nunique()
    return out
