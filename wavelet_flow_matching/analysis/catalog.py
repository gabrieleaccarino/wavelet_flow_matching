"""Model registry, PDE metadata and metric-file loading.

Every results table and analysis in the paper is built from the per-run metric
CSVs in ``experiments/<pde>/metrics/``:

* ``<MODEL>_<pde>_seed<S>_metrics.csv`` — one file per inference seed
  (42, 84, 168). Used for every model that has seeds: all WFM variants and all
  generative baselines (FM_pixel, FM_fourier, WDNO, LatentFM).
* ``<MODEL>_<pde>_stats.csv`` — single run. Used only for the deterministic
  baselines (FNO, TFNO, WNO), which are deterministic at inference time and
  were trained once, and for the paper-reproduction check.

Rows with ``field == "_global_"`` hold the field-independent physics
diagnostics (kinetic energy etc.); the per-field ML metrics (CRPS, VRMSE,
spectral coherence) are NaN there. ``load()`` keeps them separate.

MODEL is the name used in the paper (WFM_1scale_haar, FM_pixel, FM_fourier,
WDNO, LatentFM, WNO, FNO, TFNO, ...); metric files written by the inference
scripts are renamed to this scheme before analysis.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import metrics_dir

GLOBAL_FIELD = "_global_"
SEEDS = (42, 84, 168)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def _rgb(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"

WAVELETS = ("haar", "db2", "db4", "db6")


def wfm_key(scale: str, wavelet: str) -> str:
    """Model key for a WFM variant, e.g. wfm_key('3scale', 'db2') -> 'WFM_3scale_db2'."""
    return f"WFM_{'1' if scale.startswith('1') else '3'}scale_{wavelet}"


MODEL_REGISTRY: dict[str, dict] = {}
for _w in WAVELETS:
    MODEL_REGISTRY[wfm_key("1scale", _w)] = dict(
        label=f"WFM-1 {_w}", group="wfm-1scale", color=_rgb(98, 124, 107), source="metrics")
for _w in WAVELETS:
    MODEL_REGISTRY[wfm_key("3scale", _w)] = dict(
        label=f"WFM-3 {_w}", group="wfm-3scale", color=_rgb(117, 176, 138), source="metrics")
MODEL_REGISTRY.update({
    "FM_pixel":   dict(label=r"FM$_\text{pixel}$",   group="generative", color=_rgb(176, 176, 176), source="metrics"),
    "FM_fourier": dict(label=r"FM$_\text{fourier}$", group="generative", color=_rgb(143, 94, 106),  source="metrics"),  # 32 modes
    "WDNO":       dict(label="WDNO",                 group="generative", color=_rgb(191, 134, 148), source="metrics"),  # db2, 5 samples
    # Latent-space FM baseline (rebuttal): DC-AE f32c64 of Rozet et al. 2025
    # + flow matching in latent space. Trained and evaluated on Rayleigh-Benard only.
    "LatentFM":   dict(label=r"LatentFM$_{\div 64}$", group="generative", color=_rgb(120, 140, 190), source="metrics",
                       pdes=("rayleigh_benard",)),
    "WNO":        dict(label="WNO",  group="deterministic", color=_rgb(176, 136, 84),  source="stats"),
    "FNO":        dict(label="FNO",  group="deterministic", color=_rgb(208, 169, 110), source="stats"),
    "TFNO":       dict(label="TFNO", group="deterministic", color=_rgb(236, 227, 163), source="stats"),
})

# Synthetic "selected wavelet" rows (one per WFM scale; per-PDE wavelet below).
WFM1_BEST, WFM3_BEST = "WFM_1scale_best", "WFM_3scale_best"
MODEL_REGISTRY[WFM1_BEST] = dict(label="WFM 1-scale", group="wfm-best", color=_rgb(98, 124, 107), source=None)
MODEL_REGISTRY[WFM3_BEST] = dict(label="WFM 3-scale", group="wfm-best", color=_rgb(117, 176, 138), source=None)
LATENTFM = "LatentFM"

LABELS = {m: d["label"] for m, d in MODEL_REGISTRY.items()}
COLORS = {m: d["color"] for m, d in MODEL_REGISTRY.items()}


def _by_group(*groups):
    return [m for m, d in MODEL_REGISTRY.items() if d["group"] in groups]


WFM_1SCALE = _by_group("wfm-1scale")
WFM_3SCALE = _by_group("wfm-3scale")
WFM_ALL = WFM_1SCALE + WFM_3SCALE
GEN_MODELS = _by_group("generative")
DET_MODELS = _by_group("deterministic")
BEST = [WFM1_BEST, WFM3_BEST]

# Row orders and \midrule blocks used by the tables.
ROWS_ALL = WFM_ALL + GEN_MODELS + DET_MODELS
ROWS_GEN = WFM_ALL + GEN_MODELS
GROUPS_ALL = [WFM_1SCALE, WFM_3SCALE, GEN_MODELS, DET_MODELS]
GROUPS_GEN = [WFM_1SCALE, WFM_3SCALE, GEN_MODELS]
ROWS_BEST_GEN = BEST + GEN_MODELS
GROUPS_BEST_GEN = [BEST, GEN_MODELS]

# Selected wavelet per PDE (Table S11). The tables module re-derives this from
# the data and refuses to build if the two disagree.
BEST_WFM_PER_PDE = {
    "turbulent_radiative_layer_2D": {"1scale": "haar", "3scale": "haar"},
    "rayleigh_benard":              {"1scale": "db4",  "3scale": "db4"},
    "active_matter":                {"1scale": "haar", "3scale": "db2"},
}

# ---------------------------------------------------------------------------
# PDEs
# ---------------------------------------------------------------------------
PDES = ["turbulent_radiative_layer_2D", "rayleigh_benard", "active_matter"]
PDE_LABELS = {
    "turbulent_radiative_layer_2D": r"Turbulent Radiative Layer 2D",
    "rayleigh_benard":              r"Rayleigh--B\'{e}nard",
    "active_matter":                "Active Matter",
}
PDE_SHORT = {"turbulent_radiative_layer_2D": "TRL", "rayleigh_benard": "RB", "active_matter": "AM"}

# Rollout horizons: (inclusive upper bound on the 1-based lead time, label).
# Inclusive bounds reproduce the paper's Tables S12/S13 exactly.
HORIZON_EDGES = {
    "turbulent_radiative_layer_2D": [(32, "01:32"), (64, "33:64"), (97,  "65:97")],
    "rayleigh_benard":              [(32, "01:32"), (64, "33:64"), (196, "65:196")],
    "active_matter":                [(16, "01:16"), (32, "17:32"), (77,  "33:77")],
}
HORIZONS = {p: [lbl for _, lbl in e] for p, e in HORIZON_EDGES.items()}

BANDS = ["Low", "Mid", "High"]
BAND_COLS = {"rmse_c_low": "Low", "rmse_c_mid": "Mid", "rmse_c_high": "High"}

# Physical-parameter columns. RB stores ln(Ra) and ln(Pr) (log-transformed
# conditioning); we map them back to the nominal values for table headers.
PARAM_LABELS = {
    "tcool":    lambda v: f"{v:g}",
    "Rayleigh": lambda v: rf"$10^{{{int(round(v / np.log(10)))}}}$",
    "Prandtl":  lambda v: rf"$10^{{{int(round(v / np.log(10)))}}}$" if round(v / np.log(10)) != 0 else "$1$",
}


def _horizon_of(pde: str, step1: int) -> str:
    for ub, lbl in HORIZON_EDGES[pde]:
        if step1 <= ub:
            return lbl
    return HORIZON_EDGES[pde][-1][1]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
_SEED_RX = re.compile(r"seed[_\-]?(\d+)", re.I)


def files_for(model: str, pde: str, suffix: str) -> list[Path]:
    """Metric files for one (model, pde): ``<model>_<pde>_seed<S>_metrics.csv``
    (suffix "metrics") or ``<model>_<pde>_stats.csv`` (suffix "stats").
    IC-perturbation runs (``_ic<eta>_``) are not matched."""
    pat = f"{model}_{pde}_seed*_metrics.csv" if suffix == "metrics" else f"{model}_{pde}_stats.csv"
    return [Path(f) for f in sorted(glob.glob(str(metrics_dir(pde) / pat)))]


def _read(fp: Path, model: str, pde: str) -> pd.DataFrame:
    df = pd.read_csv(fp)
    df["model"] = model
    df["pde"] = pde
    if not ("seed" in df.columns and df["seed"].notna().any()):
        m = _SEED_RX.search(fp.stem)
        df["seed"] = int(m.group(1)) if m else 0      # 0 == single run
    df["seed"] = df["seed"].fillna(0).astype(int)
    df["source"] = fp.name
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """1-based lead time (`step`) and its rollout horizon label."""
    df = df.copy()
    df["step"] = df["step"].astype(int) + 1
    df["horizon"] = [_horizon_of(p, s) for p, s in zip(df["pde"], df["step"])]
    return df


def load(models=None, pdes=PDES, source: str = "auto", verbose: bool = True):
    """Load metric rows for `models` x `pdes`.

    source:
      "auto"    -> each model's registry source (seeds where available,
                   single-run stats for the deterministic baselines);
      "metrics" -> per-seed files only;
      "stats"   -> single-run files only (paper-reproduction check).

    Returns (fields_df, global_df): per-field ML metrics, and the
    `_global_` physics-diagnostic rows (empty for stats files).
    """
    models = list(models) if models is not None else ROWS_ALL
    frames, missing = [], []
    for pde in pdes:
        for model in models:
            if pde not in MODEL_REGISTRY[model].get("pdes", PDES):
                continue
            src = MODEL_REGISTRY[model]["source"] if source == "auto" else source
            fs = files_for(model, pde, src)
            if not fs:
                missing.append((pde, model, src))
                continue
            frames.extend(_read(f, model, pde) for f in fs)
    if not frames:
        raise FileNotFoundError("no metric files found — check WAVELET_FLOW_MATCHING_ROOT")
    full = pd.concat(frames, ignore_index=True)
    key = ["model", "pde", "seed", "traj", "field", "step"]
    dup = full.duplicated(subset=key)
    if dup.any():
        raise ValueError(f"{dup.sum()} duplicated (model,pde,seed,traj,field,step) rows: "
                         f"{full.loc[dup, 'source'].unique()[:5]}")
    full = add_features(full)
    is_g = full["field"] == GLOBAL_FIELD
    fields, glob_ = full[~is_g].reset_index(drop=True), full[is_g].reset_index(drop=True)
    if verbose:
        cov = full.groupby(["pde", "model"])["seed"].unique()
        print(f"loaded {len(full):,} rows from {full['source'].nunique()} files "
              f"({len(fields):,} per-field, {len(glob_):,} global)")
        for pde, model, src in missing:
            print(f"  [missing] {PDE_SHORT[pde]:3s} {model} ({src})")
        bad = cov[cov.apply(lambda s: 0 not in s and sorted(s) != list(SEEDS))]
        for (pde, model), s in bad.items():
            print(f"  [seeds]   {PDE_SHORT[pde]:3s} {model}: {sorted(s)}")
    return fields, glob_


