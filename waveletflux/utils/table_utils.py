"""
table_utils.py — supporting code for the WaveletFlow results tables.

Import once from the notebook:

    from table_utils import *
    configure(project_root="/path/to/waveletflux")

Two flavours are produced for every results table:

  * "paper" — one run, ALL models, reproduces the numbers already in the paper.
              Reads  *_stats.csv   (SINGLE_SUFFIX).
  * "seeds" — mean ± std across seeds, only the BEST model per PDE (rebuttal).
              Reads  *_metrics.csv  (SEEDS_SUFFIX).

The only thing that changes between the two is (a) which files are loaded and
(b) which rows are kept — the table builders are identical.

Paper table map (see the notebook section headers):
  VRMSE / horizons ....... Table S11
  CRPS  / horizons ....... Table S12
  Spectral RMSE / bands .. Table S13 (and S9)
  Wavelet selection ...... Table 1  / S10
  CRPS ablations ......... Table 3a/S14 (tcool), 3b/S16 (Ra), S17 (Pr)
  Spectral ablations ..... Table S15 (tcool), S18 (Ra/Pr)
Timing tables (2, S19, S20) come from separate profiling, not the metric CSVs.
"""
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ===========================================================================
# 1 · Configuration  (edit these from the notebook via configure(...))
# ===========================================================================
PROJECT_ROOT  = "/Users/gabrieleaccarino/Workspace/waveletflux"
METRICS_GLOB  = "{root}/experiments/{pde}/metrics/{model}*_{suffix}.csv"
SINGLE_SUFFIX = "stats"    # paper-reproduction source  (single run per model)
SEEDS_SUFFIX  = "metrics"  # across-seed source          (one file per seed)
SINGLE_SEED   = None       # None -> use the whole single-run frame as-is
AGG_MODE      = "mean"     # within-experiment reduction: "mean" | "median"
STD_DDOF      = 1          # across-seed std: 1 = sample, 0 = population
DEC           = 3          # decimals shown in every table


def configure(**kw):
    """Override any config value by name, e.g. configure(project_root=..., dec=4)."""
    alias = {"project_root": "PROJECT_ROOT", "single_suffix": "SINGLE_SUFFIX",
             "seeds_suffix": "SEEDS_SUFFIX", "single_seed": "SINGLE_SEED",
             "agg_mode": "AGG_MODE", "std_ddof": "STD_DDOF", "dec": "DEC",
             "metrics_glob": "METRICS_GLOB"}
    g = globals()
    for k, v in kw.items():
        key = alias.get(k, k.upper())
        if key not in g:
            raise KeyError(f"unknown config option: {k}")
        g[key] = v


# ===========================================================================
# 2 · Model registry
# ===========================================================================
def _rgb(r, g, b):
    return f"#{r:02x}{g:02x}{b:02x}"

MODEL_REGISTRY = {
    "WaveletFlow_1scale_haar_v4.0":  dict(label="WFM-1 haar", group="wfm-1scale", color=_rgb(98, 124, 107)),
    "WaveletFlow_1scale_db2_v4.0":   dict(label="WFM-1 db2",  group="wfm-1scale", color=_rgb(98, 124, 107)),
    "WaveletFlow_1scale_db4_v4.0":   dict(label="WFM-1 db4",  group="wfm-1scale", color=_rgb(98, 124, 107)),
    "WaveletFlow_1scale_db6_v4.0":   dict(label="WFM-1 db6",  group="wfm-1scale", color=_rgb(98, 124, 107)),
    "WaveletFlow_3scales_haar_v4.0": dict(label="WFM-3 haar", group="wfm-3scale", color=_rgb(117, 176, 138)),
    "WaveletFlow_3scales_db2_v4.0":  dict(label="WFM-3 db2",  group="wfm-3scale", color=_rgb(117, 176, 138)),
    "WaveletFlow_3scales_db4_v4.0":  dict(label="WFM-3 db4",  group="wfm-3scale", color=_rgb(117, 176, 138)),
    "WaveletFlow_3scales_db6_v4.0":  dict(label="WFM-3 db6",  group="wfm-3scale", color=_rgb(117, 176, 138)),
    "FM_data_space_v4.0":            dict(label=r"FM$_\text{pixel}$",   group="generative",    color=_rgb(176, 176, 176)),
    "FourierFlow_16modes_v4.0":      dict(label=r"FM$_\text{fourier}$", group="generative",    color=_rgb(143, 94, 106)),
    "WDNO_db2":                      dict(label="WDNO",                 group="generative",    color=_rgb(191, 134, 148)),
    "WNO_3scales_db2":               dict(label="WNO",  group="deterministic", color=_rgb(176, 136, 84)),
    "FNO":                           dict(label="FNO",  group="deterministic", color=_rgb(208, 169, 110)),
    "TFNO":                          dict(label="TFNO", group="deterministic", color=_rgb(236, 227, 163)),
}

# Two synthetic "best model" rows used only by the seeds/rebuttal tables.
# Their per-PDE value is pulled from that PDE's selected wavelet (see §3).
WFM1_BEST, WFM3_BEST = "WFM_1scale_best", "WFM_3scale_best"
MODEL_REGISTRY[WFM1_BEST] = dict(label="WFM 1-scale", group="wfm-1scale", color=_rgb(98, 124, 107))
MODEL_REGISTRY[WFM3_BEST] = dict(label="WFM 3-scale", group="wfm-3scale", color=_rgb(117, 176, 138))

# Extra model, metrics-only (across-seed runs), shown only in the dedicated
# Rayleigh–Bénard table. Its own group keeps it out of the standard tables.
# If your files are named differently, change the key to match the filename
# prefix (load globs "{key}*_{suffix}.csv").
LATENTFM = r"LatentFM"
MODEL_REGISTRY[LATENTFM] = dict(label=r"$\mathrm{LatentFM}\,\mathrm{÷}_{64}$", group="latent", color=_rgb(120, 140, 190))

LABELS  = {m: d["label"] for m, d in MODEL_REGISTRY.items()}
COLORS  = {m: d["color"] for m, d in MODEL_REGISTRY.items()}

def _by_group(*groups):
    return [m for m, d in MODEL_REGISTRY.items() if d["group"] in groups
            and m not in (WFM1_BEST, WFM3_BEST)]

WFM_1SCALE = _by_group("wfm-1scale")
WFM_3SCALE = _by_group("wfm-3scale")
WFM_ALL    = WFM_1SCALE + WFM_3SCALE
GEN_MODELS = _by_group("generative")
DET_MODELS = _by_group("deterministic")

# Row order + group blocks (used to draw \midrule between model families).
ALL_ROWS         = WFM_ALL + GEN_MODELS + DET_MODELS
ALL_ROWS_GEN     = WFM_ALL + GEN_MODELS
BEST_ROWS        = [WFM1_BEST, WFM3_BEST] + GEN_MODELS + DET_MODELS
BEST_ROWS_GEN    = [WFM1_BEST, WFM3_BEST] + GEN_MODELS

GROUPS_ALL      = [WFM_1SCALE + WFM_3SCALE, GEN_MODELS, DET_MODELS]
GROUPS_ALL_GEN  = [WFM_ALL, GEN_MODELS]
GROUPS_BEST     = [[WFM1_BEST, WFM3_BEST], GEN_MODELS, DET_MODELS]
GROUPS_BEST_GEN = [[WFM1_BEST, WFM3_BEST], GEN_MODELS]


# ===========================================================================
# 3 · Best wavelet per PDE  (Table S10 — the "pick the best model" mechanism)
# ===========================================================================
# Edit this as new multi-seed runs land. Keys are PDE names; values give the
# selected wavelet for the 1-scale and 3-scale WFM variants.
BEST_WFM_PER_PDE = {
    "turbulent_radiative_layer_2D": {"1scale": "haar", "3scale": "haar"},
    "rayleigh_benard":              {"1scale": "db4",  "3scale": "db4"},
    "active_matter":                {"1scale": "haar", "3scale": "db2"},
}

def wfm_key(scale, wavelet):
    """Model key for a WFM variant, e.g. wfm_key('3scale','db2')."""
    stem = "1scale" if scale == "1scale" else "3scales"
    return f"WaveletFlow_{stem}_{wavelet}_v4.0"


# ===========================================================================
# 4 · PDE metadata: labels, physical parameters, rollout horizons, bands
# ===========================================================================
PDES = ["turbulent_radiative_layer_2D", "rayleigh_benard", "active_matter"]

PDE_LABELS = {
    "turbulent_radiative_layer_2D": r"Turb. Rad. Layer 2D",
    "rayleigh_benard":              r"Rayleigh--B\'{e}nard",
    "active_matter":                "Active Matter",
}
PDE_PARAMS = {
    "turbulent_radiative_layer_2D": ["tcool"],
    "rayleigh_benard":              ["Rayleigh", "Prandtl"],
    "active_matter":                ["zeta", "alpha"],
}

# Rollout-step ranges shown as columns; (upper_bound, label) drive the binning.
HORIZON_EDGES = {
    "turbulent_radiative_layer_2D": [(32, "01:32"), (64, "33:64"), (97,  "65:97")],
    "rayleigh_benard":              [(32, "01:32"), (64, "33:64"), (196, "65:196")],
    "active_matter":                [(16, "01:16"), (32, "17:32"), (77,  "33:77")],
}
HORIZONS = {pde: [lbl for _, lbl in edges] for pde, edges in HORIZON_EDGES.items()}

def _horizon_of(pde, step):
    for ub, lbl in HORIZON_EDGES[pde]:
        if step <= ub:
            return lbl
    return HORIZON_EDGES[pde][-1][1]

BANDS     = ["Low", "Mid", "High"]
BAND_COLS = {"rmse_c_low": "Low", "rmse_c_mid": "Mid", "rmse_c_high": "High"}


# ===========================================================================
# 5 · Loading  (stats OR metrics; one file per seed for metrics)
# ===========================================================================
_SEED_RX = [re.compile(r"seed[_\-]?(\d+)", re.I),
            re.compile(r"(?:^|_)s(\d+)(?:_|$)", re.I),
            re.compile(r"rng[_\-]?(\d+)", re.I)]

def _parse_seed(stem, df):
    """Seed for one file: explicit 'seed' column first, then filename token."""
    if "seed" in df.columns and df["seed"].notna().any():
        vals = df["seed"].dropna().unique()
        return int(vals[0]) if len(vals) == 1 else None
    for rx in _SEED_RX:
        m = rx.search(stem)
        if m:
            return int(m.group(1))
    return None

def load(suffix, models=ALL_ROWS, pdes=PDES, verbose=True):
    """Load every {model}*_{suffix}.csv across `pdes`, tagging model/pde/seed.

    Missing models are reported and skipped (they simply won't appear in the
    tables) — this is expected while some multi-seed runs are still coming in.
    """
    frames, seen = [], {p: set() for p in pdes}
    for pde in pdes:
        for model in models:
            pat = METRICS_GLOB.format(root=PROJECT_ROOT, pde=pde, model=model, suffix=suffix)
            for fp in sorted(glob.glob(pat)):
                df = pd.read_csv(fp)
                stem = Path(fp).stem
                seed = _parse_seed(stem, df)
                if "model" in df.columns:
                    df["model"] = model
                else:
                    df.insert(0, "model", model)
                df["pde"] = pde
                if not ("seed" in df.columns and df["seed"].notna().any()):
                    df["seed"] = seed
                df["source"] = stem
                frames.append(df)
                seen[pde].add(model)

    if not frames:
        raise FileNotFoundError(
            f"No *_{suffix}.csv under {PROJECT_ROOT}. Check PROJECT_ROOT / METRICS_GLOB.")

    full = pd.concat(frames, ignore_index=True)
    full = full.drop_duplicates(
        subset=[c for c in full.columns if c not in {"source"}])
    full = add_features(full)

    if verbose:
        seeds = sorted(full["seed"].dropna().unique().astype(int).tolist())
        print(f"[{suffix}] {full.shape[0]:,} rows · seeds={seeds or '(single run)'}")
        for pde in pdes:
            missing = [LABELS[m] for m in models if m not in seen[pde]]
            if missing:
                print(f"    {pde}: missing {', '.join(missing)}")
    return full


# ===========================================================================
# 6 · Feature engineering
# ===========================================================================
def add_features(df):
    df = df.copy()
    df["step"] = df["step"] + 1
    df["horizon"] = [_horizon_of(p, s) for p, s in zip(df["pde"], df["step"])]
    df["seed"] = df["seed"].fillna(0).astype(int) if "seed" in df.columns else 0
    return df


# ===========================================================================
# 7 · Active-frame plumbing  (builders read the frame set here)
# ===========================================================================
_ACTIVE = None

def set_frame(df):
    global _ACTIVE
    _ACTIVE = df
    return df

def _df():
    if _ACTIVE is None:
        raise RuntimeError("No active frame — call set_frame(...) first.")
    return _ACTIVE

def _reduce(x):
    return np.nanmedian(x) if AGG_MODE == "median" else np.nanmean(x)


# ===========================================================================
# 8 · Best-model view  (per-PDE: keep selected wavelet + baselines, relabel)
# ===========================================================================
def best_view(df, include_det=True, extra=()):
    """Filter to the best WFM variant per PDE (relabelled to generic 'ours'
    rows) plus the baseline models. Used for the across-seed rebuttal tables.
    `extra` adds further model keys to keep (e.g. LatentFM for the RB table)."""
    baselines = GEN_MODELS + (DET_MODELS if include_det else []) + list(extra)
    out = []
    for pde, sub in df.groupby("pde", sort=False):
        sel = BEST_WFM_PER_PDE.get(pde)
        if sel:
            k1, k3 = wfm_key("1scale", sel["1scale"]), wfm_key("3scale", sel["3scale"])
            sub = sub[sub["model"].isin([k1, k3] + baselines)].copy()
            sub.loc[sub["model"] == k1, "model"] = WFM1_BEST
            sub.loc[sub["model"] == k3, "model"] = WFM3_BEST
        out.append(sub)
    return pd.concat(out, ignore_index=True)


# ===========================================================================
# 9 · Table builders  (all return a numeric DataFrame; rows = model labels)
# ===========================================================================
def _pivot_pde_cols(per_pde, models):
    """Concatenate per-PDE frames side by side, ordering rows by `models`.
    Rows stay keyed by model *key* (labels are applied only at print time);
    columns are keyed by (pde, sub)."""
    frames = []
    for pde, sub in per_pde:
        sub = sub.reindex([m for m in models if m in sub.index])
        sub.columns = pd.MultiIndex.from_tuples([(pde, c) for c in sub.columns],
                                                names=["pde", "sub"])
        frames.append(sub)
    out = pd.concat(frames, axis=1)
    out.index.name = "Model"
    return out

def build_horizon_table(metric, models):
    """VRMSE / CRPS by rollout horizon.  Model × (pde, horizon).  [S11 / S12]"""
    df = _df()
    per_pde = []
    for pde in PDES:
        sub = (df[(df["pde"] == pde) & df["model"].isin(models)]
               .groupby(["model", "horizon"])[metric].agg(_reduce).unstack("horizon")
               .reindex(columns=HORIZONS[pde]))
        per_pde.append((pde, sub))
    return _pivot_pde_cols(per_pde, models)

def build_bands_table(models):
    """Spectral coherence RMSE by frequency band.  Model × (pde, band).  [S13]"""
    df = _df()
    per_pde = []
    for pde in PDES:
        sub = (df[(df["pde"] == pde) & df["model"].isin(models)]
               .groupby("model")[list(BAND_COLS)].agg(_reduce).rename(columns=BAND_COLS).reindex(columns=BANDS))
        per_pde.append((pde, sub))
    return _pivot_pde_cols(per_pde, models)

def build_selection_table(models):
    """Band-averaged spectral RMSE, one value per PDE.  Model × pde.  [Table 1]
    Columns are keyed by the raw PDE name (same as build_bands_table); the nice
    label is applied only when displaying / writing LaTeX."""
    df = _df()
    cols = {}
    for pde in PDES:
        s = (df[(df["pde"] == pde) & df["model"].isin(models)]
             .groupby("model")[list(BAND_COLS)].agg(_reduce).mean(axis=1))
        cols[pde] = s.reindex([m for m in models if m in s.index])
    out = pd.DataFrame(cols)
    out.index.name = "Model"
    return out

def select_wavelets():
    """Per PDE and WFM scale, the wavelet with the lowest band-averaged spectral
    RMSE in the active frame. Returns {(pde, scale): (model_key, value)}.
    This single function drives Table 1's bolding, Table S10, and the check
    against BEST_WFM_PER_PDE — so the three can never disagree."""
    df = _df()
    out = {}
    for pde in PDES:
        for scale, keys in (("1scale", WFM_1SCALE), ("3scale", WFM_3SCALE)):
            s = (df[(df["pde"] == pde) & df["model"].isin(keys)]
                 .groupby("model")[list(BAND_COLS)].agg(_reduce).mean(axis=1))
            if not s.empty:
                out[(pde, scale)] = (s.idxmin(), round(float(s.min()), DEC))
    return out

def build_s10():
    """Table S10 — the selected wavelet per PDE and WFM scale (from the data)."""
    rows = []
    for (pde, scale), (key, val) in select_wavelets().items():
        rows.append({"PDE": PDE_LABELS[pde], "WFM": scale.replace("scale", "-scale"),
                     "wavelet": key.split("_")[2], "value": val})
    return pd.DataFrame(rows)

def selection_mismatch():
    """Where the data-derived selection differs from the hard-coded
    BEST_WFM_PER_PDE (used by the seeds tables). Empty list == consistent."""
    out = []
    for (pde, scale), (key, _) in select_wavelets().items():
        wl = key.split("_")[2]
        want = BEST_WFM_PER_PDE.get(pde, {}).get(scale)
        if want and wl != want:
            out.append(f"{pde}/{scale}: data={wl} vs hard-coded={want}")
    return out

def compute_best_wavelets(df):
    """Back-compat wrapper: derive Table S10 from an explicit frame."""
    saved = _ACTIVE
    try:
        set_frame(df)
        return build_s10()
    finally:
        set_frame(saved)

def build_crps_ablation(pde, param, metric="crps", models=None):
    """CRPS (or any metric) per physical parameter value.  Model × param.
    [Table 3 / S14 / S16 / S17]"""
    models = models if models is not None else ALL_ROWS_GEN
    df = _df()
    sub = (df[(df["pde"] == pde) & df["model"].isin(models)]
           .dropna(subset=[param, metric])
           .groupby(["model", param])[metric].agg(_reduce).unstack(param))
    sub = sub.reindex([m for m in models if m in sub.index])
    sub.index.name = "Model"
    sub.columns.name = param
    return sub

def build_spectral_ablation(pde, param, models=None):
    """Spectral RMSE per band × parameter value.  Model × (band, param). [S15/S18]"""
    models = models if models is not None else ALL_ROWS_GEN
    df = _df()
    sub = (df[(df["pde"] == pde) & df["model"].isin(models)]
           .dropna(subset=[param])
           .melt(id_vars=["model", param], value_vars=list(BAND_COLS),
                 var_name="band_key", value_name="v")
           .assign(band=lambda x: x["band_key"].map(BAND_COLS))
           .groupby(["model", "band", param])["v"].agg(_reduce)
           .unstack(["band", param]).reindex(columns=BANDS, level="band"))
    sub = sub.reindex([m for m in models if m in sub.index])
    sub.index.name = "Model"
    return sub

# ===========================================================================
# 10 · Across-seed aggregation  (mean ± std, elementwise, over seeds)
# ===========================================================================
def _union(indexes):
    seen, out = set(), []
    for ix in indexes:
        for v in ix:
            if v not in seen:
                seen.add(v); out.append(v)
    proto = indexes[0]
    if isinstance(proto, pd.MultiIndex):
        return pd.MultiIndex.from_tuples(out, names=proto.names)
    return pd.Index(out, name=proto.name)

def across_seeds(build_fn, frame, *args, **kwargs):
    """Run build_fn once per seed in `frame`; return (mean_df, std_df, n_df)."""
    seeds = sorted(frame["seed"].dropna().unique().astype(int).tolist())
    per_seed = []
    for s in seeds:
        set_frame(frame[frame["seed"] == s])
        t = build_fn(*args, **kwargs)
        if t is not None and t.size:
            per_seed.append(t)
    if not per_seed:
        raise ValueError("No per-seed tables produced.")
    idx  = _union([t.index for t in per_seed])
    cols = _union([t.columns for t in per_seed])
    arr = np.stack([t.reindex(index=idx, columns=cols).to_numpy(float) for t in per_seed])
    import warnings
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells -> NaN
        mean = np.nanmean(arr, axis=0)
        n    = np.sum(~np.isnan(arr), axis=0)
        std  = np.where(n >= 2, np.nanstd(arr, axis=0, ddof=STD_DDOF), np.nan)
    mk = lambda a: pd.DataFrame(a, index=idx, columns=cols)
    return mk(mean).round(DEC), mk(std).round(DEC), mk(n.astype(float))


def seed_spread(build_fn, seeds_df, *args, **kwargs):
    """Diagnostic: {seed -> per-seed table}, so you can see whether the seeds
    actually differ. If these tables are (nearly) identical, the std across
    seeds is genuinely ~0 — i.e. the metric files barely vary by seed, which is
    a property of the data, not of the aggregation."""
    frame = best_view(seeds_df)
    out = {}
    for s in sorted(frame["seed"].dropna().unique().astype(int).tolist()):
        set_frame(frame[frame["seed"] == s])
        out[s] = build_fn(*args, **kwargs)
    return out


def build_rb_combined(models, pde="rayleigh_benard"):
    """Rayleigh–Bénard only: spectral coherence RMSE by band, plus VRMSE and
    CRPS by temporal regime, in one wide table.  Model × (metric-group, sub)."""
    df = _df()
    d = df[(df["pde"] == pde) & df["model"].isin(models)]
    sp = (d.groupby("model")[list(BAND_COLS)].agg(_reduce)
          .rename(columns=BAND_COLS).reindex(columns=BANDS))
    sp.columns = pd.MultiIndex.from_tuples([("Spectral RMSE", b) for b in sp.columns])

    def by_regime(metric, name):
        h = (d.groupby(["model", "horizon"])[metric].agg(_reduce)
             .unstack("horizon").reindex(columns=HORIZONS[pde]))
        h.columns = pd.MultiIndex.from_tuples([(name, c) for c in h.columns])
        return h

    #out = pd.concat([sp, by_regime("vrmse", "VRMSE"), by_regime("crps", "CRPS")], axis=1)
    out = pd.concat([sp, by_regime("crps", "CRPS")], axis=1)
    out = out.reindex([m for m in models if m in out.index])
    out.index.name = "Model"
    return out


# ===========================================================================
# 11 · Ranking + display (bold best, underline second, per column)
# ===========================================================================
def _rank_styles(rank_on):
    out = pd.DataFrame("", index=rank_on.index, columns=rank_on.columns)
    for col in rank_on.columns:
        order = rank_on[col].dropna().sort_values().index
        if len(order) >= 1: out.loc[order[0], col] = "font-weight:bold"
        if len(order) >= 2: out.loc[order[1], col] = "text-decoration:underline"
    return out

def show(mean_df, std_df=None):
    """Styler for a numeric table (single run) or a mean±std table (seeds).
    Rows come in keyed by model key; they are relabelled for display here."""
    disp = mean_df.rename(index=LABELS)
    if std_df is None:
        return (disp.style.apply(lambda _: _rank_styles(disp), axis=None)
                .format(f"{{:.{DEC}f}}", na_rep="--"))
    txt = pd.DataFrame(index=disp.index, columns=disp.columns, dtype=object)
    for i in mean_df.index:
        for c in mean_df.columns:
            m, s = mean_df.loc[i, c], std_df.loc[i, c]
            txt.loc[LABELS.get(i, i), c] = ("--" if pd.isna(m) else f"{m:.{DEC}f}"
                             + ("" if pd.isna(s) else f" ± {s:.{DEC}f}"))
    return txt.style.apply(lambda _: _rank_styles(disp), axis=None)


# ===========================================================================
# 12 · LaTeX
# ===========================================================================
def _cell(val, rank, std=None):
    if pd.isna(val):
        return "--"
    s = f"{val:.{DEC}f}"
    if std is not None and not pd.isna(std):
        s += r"{\scriptstyle\,\pm" + f"{std:.{DEC}f}" + "}"
    if rank == 1: return r"\textbf{" + s + "}"
    if rank == 2: return r"\underline{" + s + "}"
    return s

def _ranks(df):
    return {col: {m: i + 1 for i, m in enumerate(
                df[col].dropna().sort_values().index)}
            for col in df.columns}

def _table_env(caption, label, colspec, body, placement="ht"):
    return "\n".join([
        rf"\begin{{table}}[{placement}]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
        *body, r"\bottomrule", r"\end{tabular}", r"\end{table}"])

def latex_wide(df, caption, label, std_df=None, groups=None, row_label="Rollout"):
    """2-level column table (pde × sub) — used for horizon and band tables."""
    assert isinstance(df.columns, pd.MultiIndex)
    pdes = list(dict.fromkeys(df.columns.get_level_values(0)))
    subs = {p: [s for pp, s in df.columns if pp == p] for p in pdes}
    ranks = _ranks(df)
    colspec = "l" + "".join(" " + "c" * len(subs[p]) for p in pdes)
    top, cmid, start = [row_label], [], 2
    for p in pdes:
        n = len(subs[p])
        top.append(rf"\multicolumn{{{n}}}{{c}}{{{PDE_LABELS.get(p, p)}}}")
        cmid.append(rf"\cmidrule(lr){{{start}-{start + n - 1}}}")
        start += n
    header = [" & ".join(top) + r" \\", " ".join(cmid),
              " & ".join([""] + [s for p in pdes for s in subs[p]]) + r" \\",
              r"\midrule"]
    groups = groups or [list(df.index)]
    body = list(header)
    for gi, group in enumerate(groups):
        if gi > 0: body.append(r"\midrule")
        for m in group:
            if m not in df.index: continue
            cells = [LABELS.get(m, m)] + [_cell(df.loc[m, c], ranks[c].get(m, 99),
                           None if std_df is None else std_df.loc[m, c])
                           for c in df.columns]
            body.append(" & ".join(cells) + r" \\")
    return _table_env(caption, label, colspec, body)

def latex_flat(df, caption, label, std_df=None, groups=None, corner=""):
    """Single-level column table — used for selection and CRPS ablations."""
    ranks = _ranks(df)
    colspec = "l" + "c" * len(df.columns)
    body = [f"{corner} & " + " & ".join(str(c) for c in df.columns) + r" \\",
            r"\midrule"]
    groups = groups or [list(df.index)]
    for gi, group in enumerate(groups):
        if gi > 0: body.append(r"\midrule")
        for m in group:
            if m not in df.index: continue
            cells = [str(LABELS.get(m, m))] + [_cell(df.loc[m, c], ranks[c].get(m, 99),
                                None if std_df is None else std_df.loc[m, c])
                                for c in df.columns]
            body.append(" & ".join(cells) + r" \\")
    return _table_env(caption, label, colspec, body)

def latex_spectral_ablation(df, caption, label, std_df=None, param_label=""):
    """2-level (band × param) table with a \\multirow model column.  [S15/S18]"""
    assert isinstance(df.columns, pd.MultiIndex)
    bands  = list(dict.fromkeys(df.columns.get_level_values(0)))
    params = list(dict.fromkeys(df.columns.get_level_values(1)))
    ranks = _ranks(df)
    colspec = "ll" + "c" * len(params)
    head = (param_label + " & & " if param_label else "Model & Band & ")
    body = [head + " & ".join(str(p) for p in params) + r" \\", r"\midrule"]
    for mi, m in enumerate(df.index):
        if mi > 0: body.append(r"\midrule")
        corner = rf"\multirow{{{len(bands)}}}{{*}}{{{LABELS.get(m, m)}}}"
        for bi, band in enumerate(bands):
            cells = [corner if bi == 0 else "", band]
            for p in params:
                col = (band, p)
                v = df.loc[m, col] if col in df.columns else np.nan
                s = None if std_df is None or col not in std_df.columns else std_df.loc[m, col]
                cells.append(_cell(v, ranks.get(col, {}).get(m, 99), s))
            body.append(" & ".join(cells) + r" \\")
    return _table_env(caption, label, colspec, body)


# ===========================================================================
# 13 · High-level emit  (renders + saves both flavours of one table)
# ===========================================================================
OUT = Path("outputs")

def _save(text, name):
    (OUT / "latex").mkdir(parents=True, exist_ok=True)
    (OUT / "latex" / f"{name}.tex").write_text(text)

def _save_csv(df, name):
    (OUT / "csv").mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "csv" / f"{name}.csv")


# --- Table 1 / S10: selection is highlighted per WFM scale, not globally -----
def _selection_cells(t1):
    """Bool frame, True at the chosen wavelet per (PDE column, WFM scale)."""
    mask = pd.DataFrame(False, index=t1.index, columns=t1.columns)
    for (pde, _scale), (key, _val) in select_wavelets().items():
        if key in mask.index and pde in mask.columns:
            mask.loc[key, pde] = True
    return mask

def show_selection(t1):
    """Styler for Table 1: bold the wavelet selected within each WFM scale."""
    disp   = t1.rename(index=LABELS, columns=PDE_LABELS)
    styles = (_selection_cells(t1).rename(index=LABELS, columns=PDE_LABELS)
              .map(lambda b: "font-weight:bold" if b else ""))
    return (disp.style.apply(lambda _: styles, axis=None)
            .format(f"{{:.{DEC}f}}", na_rep="--"))

def latex_selection(t1, caption, label, groups=None):
    """Table 1 LaTeX: PDE labels in the header, bold the per-scale selection."""
    mask = _selection_cells(t1)
    cols = list(t1.columns)
    colspec = "l" + "c" * len(cols)
    body = [" & " + " & ".join(PDE_LABELS.get(c, c) for c in cols) + r" \\", r"\midrule"]
    groups = groups or [list(t1.index)]
    for gi, group in enumerate(groups):
        if gi > 0: body.append(r"\midrule")
        for m in group:
            if m not in t1.index: continue
            cells = [LABELS.get(m, m)]
            for c in cols:
                v = t1.loc[m, c]
                s = "--" if pd.isna(v) else f"{v:.{DEC}f}"
                cells.append(r"\textbf{" + s + "}" if mask.loc[m, c] else s)
            body.append(" & ".join(cells) + r" \\")
    return _table_env(caption, label, colspec, body)

def emit_selection(name, paper_df, models=None, groups=None, caption="", label=""):
    """Render + save Table 1 (wavelet selection) and Table S10, and cross-check
    the data-derived selection against the hard-coded BEST_WFM_PER_PDE.

    Table 1 / S10 compare all four wavelets, so they need the single-run stats
    frame (the across-seed runs only cover the already-selected wavelets). If
    that frame is absent, say so instead of failing."""
    from IPython.display import display, Markdown
    if paper_df is None or len(paper_df) == 0:
        display(Markdown(
            "_Table 1 / S10 need the single-run **stats** data (all four wavelets); "
            "the across-seed runs only cover the selected wavelet per PDE. "
            "Set `EMIT_PAPER = True` for these. A seeds version (band-averaged "
            "spectral RMSE for the best models) is produced below if metrics are loaded._"))
        return None, None
    models = models if models is not None else ALL_ROWS_GEN
    set_frame(paper_df)
    t1 = build_selection_table(models)
    display(Markdown(f"**{name} — Table 1: wavelet selection (single run)**"))
    display(show_selection(t1))
    _save(latex_selection(t1, caption, label, groups=groups), f"{name}__paper")
    _save_csv(t1, f"{name}__paper")

    s10 = build_s10()
    display(Markdown("**Table S10 — selected wavelet, derived from this data:**"))
    display(s10)
    miss = selection_mismatch()
    display(Markdown(("⚠️ derived selection differs from `BEST_WFM_PER_PDE`: " + "; ".join(miss))
                     if miss else "✅ derived selection matches `BEST_WFM_PER_PDE`."))
    _save_csv(s10, f"{name}__S10")
    return t1, s10


def emit_selection_seeds(name, seeds_df, caption="", label="tab:selection"):
    """Seeds analogue of Table 1: band-averaged spectral coherence RMSE per PDE,
    best model per PDE, mean ± std across seeds. (Wavelet *selection* / S10 is
    not reproduced here — that needs the single-run stats with all wavelets.)"""
    from IPython.display import display, Markdown
    if seeds_df is None or len(seeds_df) == 0:
        return None
    mean, std, _ = across_seeds(build_selection_table, best_view(seeds_df),
                                models=BEST_ROWS_GEN)
    mean = mean.rename(columns=PDE_LABELS)
    std  = std.rename(columns=PDE_LABELS)
    display(Markdown(f"**{name} — seeds (band-averaged spectral RMSE, mean ± std, best models)**"))
    display(show(mean, std))
    _save(latex_flat(mean, caption + r" (mean\,$\pm$\,std over seeds)", label + "_seeds",
                     std_df=std, groups=GROUPS_BEST_GEN, corner=""), f"{name}__seeds")
    _save_csv(mean, f"{name}__seeds_mean")
    _save_csv(std,  f"{name}__seeds_std")
    return mean, std


def emit_rb_combined(name, seeds_df, caption="", label="tab:rb_combined"):
    """Rayleigh–Bénard combined table (spectral bands + VRMSE + CRPS by regime)
    across seeds, including LatentFM. Metrics-only (LatentFM has no stats run)."""
    from IPython.display import display, Markdown
    if seeds_df is None or len(seeds_df) == 0:
        display(Markdown("_This table needs the across-seed **metrics** data (`EMIT_SEEDS = True`)._"))
        return None
    frame = best_view(seeds_df, include_det=False, extra=[LATENTFM])
    rows = [WFM1_BEST, WFM3_BEST] + GEN_MODELS + [LATENTFM]
    mean, std, n = across_seeds(build_rb_combined, frame, models=rows)
    display(Markdown(f"**{name} — Rayleigh–Bénard, across seeds (mean ± std, best models + LatentFM)**"))
    display(show(mean, std))
    groups = [[WFM1_BEST, WFM3_BEST], GEN_MODELS + [LATENTFM]]
    _save(latex_wide(mean, caption, label, std_df=std, groups=groups, row_label="Model"),
          f"{name}__seeds")
    _save_csv(mean, f"{name}__seeds_mean")
    _save_csv(std,  f"{name}__seeds_std")
    return mean, std


def emit(name, build_fn, latex_fn, caption, label,
         paper_df=None, seeds_df=None,
         paper_models=None, seeds_models=None,
         paper_groups=None, seeds_groups=None, latex_kw=None,
         run_paper=True, run_seeds=True, **build_kw):
    """Build, display and save a table in both flavours.

    paper_df / seeds_df : the loaded stats / metrics frames.
    *_models, *_groups  : row order and \\midrule blocks for each flavour.
    latex_kw            : extra kwargs forwarded to latex_fn.
    Any remaining build_kw are forwarded to build_fn.
    """
    from IPython.display import display, Markdown
    latex_kw = latex_kw or {}
    results = {}

    if run_paper and paper_df is not None:
        set_frame(paper_df if SINGLE_SEED is None
                  else paper_df[paper_df["seed"] == SINGLE_SEED])
        m = build_fn(**({"models": paper_models} if paper_models else {}), **build_kw)
        display(Markdown(f"**{name} — paper (single run, all models)**"))
        display(show(m))
        tex = latex_fn(m, caption, label,
                       **({"groups": paper_groups} if paper_groups else {}), **latex_kw)
        _save(tex, f"{name}__paper"); _save_csv(m, f"{name}__paper")
        results["paper"] = m

    if run_seeds and seeds_df is not None:
        mean, std, _ = across_seeds(
            build_fn, best_view(seeds_df),
            **({"models": seeds_models} if seeds_models else {}), **build_kw)
        display(Markdown(f"**{name} — seeds (mean ± std, best models)**"))
        display(show(mean, std))
        tex = latex_fn(mean, caption + r" (mean\,$\pm$\,std over seeds)",
                       label + "_seeds", std_df=std,
                       **({"groups": seeds_groups} if seeds_groups else {}), **latex_kw)
        _save(tex, f"{name}__seeds"); _save_csv(mean, f"{name}__seeds_mean")
        _save_csv(std, f"{name}__seeds_std")
        results["seeds"] = (mean, std)

    return results


# ===========================================================================
# 14 · Synthetic fixtures  (smoke-test without the real data — optional)
# ===========================================================================
def make_synthetic_dataset(root, seeds=(0, 1, 2)):
    """Write throwaway *_stats.csv (single run, all models) and *_metrics.csv
    (per seed, best models + baselines only) so the notebook can run end to
    end with no real data. Returns the root path."""
    rng = np.random.default_rng(0)
    root = Path(root)
    base = ["m1", "m2", "spread", "crps", "rmse", "nrmse", "vrmse",
            "rmse_c_low", "rmse_c_mid", "rmse_c_high"]
    steps = {"turbulent_radiative_layer_2D": 97,
             "rayleigh_benard": 196, "active_matter": 77}
    params = {"turbulent_radiative_layer_2D": {"tcool": [0.03, 0.1, 0.3]},
              "rayleigh_benard": {"Rayleigh": [1e6, 1e8, 1e10], "Prandtl": [0.1, 1.0, 10.0]},
              "active_matter": {"zeta": [1.0, 3.0, 5.0], "alpha": [-1, -3, -5]}}

    def rows(pde, quality, seed=None):
        combos = [dict(zip(params[pde], v)) for v in
                  __import__("itertools").product(*params[pde].values())]
        # small per-seed offset so the seeds genuinely differ (non-zero std)
        seed_off = 0.0 if seed is None else 0.03 * quality * ((seed % 3) - 1)
        rec = []
        for c in combos:
            for st in range(steps[pde]):
                base_err = quality * (1 + st / steps[pde]) + seed_off
                r = {"step": st, "field": 0}
                for col in base:
                    r[col] = float(abs(base_err + rng.normal(0, 0.05 * quality)))
                r.update(c)
                if seed is not None:
                    r["seed"] = seed
                rec.append(r)
        return pd.DataFrame(rec)

    quality = {m: (0.5 if "WaveletFlow" in m else 0.9 if MODEL_REGISTRY[m]["group"] == "generative"
                   else 1.6) for m in ALL_ROWS}
    for pde in PDES:
        d = root / "experiments" / pde / "metrics"
        d.mkdir(parents=True, exist_ok=True)
        # stats: every model, single run
        for m in ALL_ROWS:
            rows(pde, quality[m]).to_csv(d / f"{m}_nsampling50_nens8_{pde}_stats.csv", index=False)
        # metrics: only the best WFM per PDE + baselines, across seeds
        sel = BEST_WFM_PER_PDE[pde]
        best = [wfm_key("1scale", sel["1scale"]), wfm_key("3scale", sel["3scale"])]
        metric_models = best + GEN_MODELS + DET_MODELS
        if pde == "rayleigh_benard":          # LatentFM: RB, metrics-only
            metric_models = metric_models + [LATENTFM]
        for m in metric_models:
            q = quality.get(m, 0.8)
            for s in seeds:
                rows(pde, q, seed=s).to_csv(
                    d / f"{m}_nsampling50_nens8_seed{s}_{pde}_metrics.csv", index=False)
    return str(root)
