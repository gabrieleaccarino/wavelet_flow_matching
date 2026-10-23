"""Results tables: builders, across-seed aggregation and LaTeX export.

Aggregation (identical for every table):
  1. within one inference seed, average the metric over trajectories, fields
     and lead times  -> one number per (model, seed, cell);
  2. across seeds, report mean and sample std (ddof=1) -> "mean ± std".
Deterministic baselines (FNO/TFNO/WNO) have a single run: mean only.

Time averaging inside step 1 is table-specific and reproduces the paper:
  * TIME_AGG = "horizon" — mean within each rollout horizon, then mean of the
    three horizons (horizon-balanced). Used by Table 1, S12 (spectral
    coherence "averaged across rollout").
  * TIME_AGG = "step"    — plain mean over all lead times. Used by the
    physical-parameter breakdowns (Table 3, S13–S17).
The two only differ on Rayleigh–Bénard, whose last horizon (65:196) is 4x
longer than the first two.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import catalog as C
from .paths import TABLES_CSV, TABLES_TEX, ensure_artifact_dirs

DEC = 3
STD_DDOF = 1
STD_IN_LATEX = True       # False -> means only in LaTeX (std still saved to CSV)


# ===========================================================================
# helpers
# ===========================================================================
def _tmean(df, keys, cols, time_agg):
    if time_agg == "horizon":
        return df.groupby(keys + ["horizon"])[cols].mean().groupby(keys).mean()
    if time_agg == "step":
        return df.groupby(keys)[cols].mean()
    raise ValueError(time_agg)


def _rows(models, present):
    return [m for m in models if m in present]


def best_view(df, keep=()):
    """Keep the selected wavelet per PDE (relabelled WFM 1-/3-scale) + `keep`."""
    out = []
    for pde, sub in df.groupby("pde", sort=False):
        sel = C.BEST_WFM_PER_PDE[pde]
        k1, k3 = C.wfm_key("1scale", sel["1scale"]), C.wfm_key("3scale", sel["3scale"])
        sub = sub[sub["model"].isin([k1, k3, *keep])].copy()
        sub["model"] = sub["model"].replace({k1: C.WFM1_BEST, k3: C.WFM3_BEST})
        out.append(sub)
    return pd.concat(out, ignore_index=True)


# ===========================================================================
# builders — each takes ONE seed's rows and returns Model x columns
# ===========================================================================
def horizon_table(df, metric, models):
    """VRMSE / CRPS per rollout horizon.  [Tables S10, S11]"""
    parts = {}
    for pde in C.PDES:
        d = df[(df.pde == pde) & df.model.isin(models)]
        if d.empty:
            continue
        t = d.groupby(["model", "horizon"])[metric].mean().unstack().reindex(columns=C.HORIZONS[pde])
        parts[pde] = t
    return _hcat(parts, models)


def bands_table(df, models, time_agg="horizon"):
    """Spectral coherence RMSE per frequency band.  [Table S12]"""
    parts = {}
    for pde in C.PDES:
        d = df[(df.pde == pde) & df.model.isin(models)]
        if d.empty:
            continue
        t = _tmean(d, ["model"], list(C.BAND_COLS), time_agg).rename(columns=C.BAND_COLS)[C.BANDS]
        parts[pde] = t
    return _hcat(parts, models)


def selection_table(df, models, time_agg="horizon"):
    """Band-averaged spectral coherence RMSE, one column per PDE.  [Table 1]"""
    cols = {}
    for pde in C.PDES:
        d = df[(df.pde == pde) & df.model.isin(models)]
        if d.empty:
            continue
        cols[pde] = _tmean(d, ["model"], list(C.BAND_COLS), time_agg).mean(axis=1)
    if not cols:
        return None
    out = pd.DataFrame(cols)
    return out.reindex(_rows(models, out.index))


def param_table(df, pde, param, models, metric="crps", time_agg="step"):
    """Metric per physical-parameter value.  [Table 3a/3b, S13, S15, S16]"""
    d = df[(df.pde == pde) & df.model.isin(models)].dropna(subset=[param, metric])
    if d.empty:
        return None
    t = _tmean(d, ["model", param], [metric], time_agg)[metric].unstack(param)
    return t.reindex(_rows(models, t.index))


def spectral_param_table(df, pde, param, models, time_agg="step"):
    """Spectral coherence RMSE per band x parameter value.  [S14, S17]"""
    d = df[(df.pde == pde) & df.model.isin(models)].dropna(subset=[param])
    if d.empty:
        return None
    t = _tmean(d, ["model", param], list(C.BAND_COLS), time_agg).rename(columns=C.BAND_COLS)[C.BANDS]
    t = t.unstack(param)               # columns: (band, param)
    t = t.reindex(columns=pd.MultiIndex.from_product([C.BANDS, sorted(d[param].unique())]))
    return t.reindex(_rows(models, t.index))


def rb_summary_table(df, models, pde="rayleigh_benard"):
    """RB: spectral bands (horizon-balanced) + CRPS per horizon.  [LatentFM table]"""
    d = df[(df.pde == pde) & df.model.isin(models)]
    if d.empty:
        return None
    sp = _tmean(d, ["model"], list(C.BAND_COLS), "horizon").rename(columns=C.BAND_COLS)[C.BANDS]
    sp.columns = pd.MultiIndex.from_tuples([("Spectral coherence RMSE", b) for b in sp.columns])
    cr = d.groupby(["model", "horizon"])["crps"].mean().unstack().reindex(columns=C.HORIZONS[pde])
    cr.columns = pd.MultiIndex.from_tuples([("CRPS", h) for h in cr.columns])
    out = pd.concat([sp, cr], axis=1)
    return out.reindex(_rows(models, out.index))


def _hcat(parts, models):
    if not parts:
        return None
    frames = []
    for pde, t in parts.items():
        t = t.copy()
        t.columns = pd.MultiIndex.from_tuples([(pde, c) for c in t.columns])
        frames.append(t)
    out = pd.concat(frames, axis=1)
    return out.reindex(_rows(models, out.index))


# ===========================================================================
# across seeds
# ===========================================================================
def across_seeds(builder, df, **kw):
    """Run `builder` once per seed slice; return (mean, std, n) frames.
    Seed 0 marks single-run models (deterministic baselines)."""
    tables = []
    for s in sorted(df["seed"].unique()):
        t = builder(df[df["seed"] == s], **kw)
        if t is not None and t.size:
            tables.append(t)
    idx = _union([t.index for t in tables])
    models = kw.get("models")
    if models is not None:
        idx = pd.Index([m for m in models if m in idx])
    cols = _union([t.columns for t in tables])
    arr = np.stack([t.reindex(index=idx, columns=cols).to_numpy(float) for t in tables])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        n = (~np.isnan(arr)).sum(0)
        mean = np.nanmean(arr, 0)
        std = np.where(n >= 2, np.nanstd(arr, 0, ddof=STD_DDOF), np.nan)
    mk = lambda a: pd.DataFrame(a, index=idx, columns=cols)
    return mk(mean), mk(std), mk(n.astype(float))


def _union(ixs):
    seen, out = set(), []
    for ix in ixs:
        for v in ix:
            if v not in seen:
                seen.add(v)
                out.append(v)
    if isinstance(ixs[0], pd.MultiIndex):
        return pd.MultiIndex.from_tuples(out)
    return pd.Index(out)


# ===========================================================================
# wavelet selection  [Table 1 bold / Table S9]
# ===========================================================================
def select_wavelets(sel_mean: pd.DataFrame) -> dict:
    """(pde, scale) -> (model key, value): lowest band-averaged spectral RMSE."""
    out = {}
    for pde in sel_mean.columns:
        for scale, keys in (("1scale", C.WFM_1SCALE), ("3scale", C.WFM_3SCALE)):
            s = sel_mean.loc[[k for k in keys if k in sel_mean.index], pde].dropna()
            if len(s):
                out[(pde, scale)] = (s.idxmin(), float(s.min()))
    return out


def check_selection(sel_mean) -> list[str]:
    bad = []
    for (pde, scale), (key, _) in select_wavelets(sel_mean).items():
        wl = key.split("_")[2]
        if wl != C.BEST_WFM_PER_PDE[pde][scale]:
            bad.append(f"{pde}/{scale}: data -> {wl}, BEST_WFM_PER_PDE -> {C.BEST_WFM_PER_PDE[pde][scale]}")
    return bad


def s9_table(sel_mean, sel_std) -> pd.DataFrame:
    rows = []
    for (pde, scale), (key, val) in select_wavelets(sel_mean).items():
        rows.append(dict(pde=pde, scale=scale, wavelet=key.split("_")[2],
                         mean=val, std=sel_std.loc[key, pde]))
    return pd.DataFrame(rows)


# ===========================================================================
# seed-spread audit
# ===========================================================================
def spread_report(name, mean, std, n):
    """Long-format relative std (%) of every cell with >=2 seeds."""
    m, s = mean.stack(list(range(mean.columns.nlevels)), future_stack=True), \
           std.stack(list(range(std.columns.nlevels)), future_stack=True)
    nn = n.stack(list(range(n.columns.nlevels)), future_stack=True)
    r = pd.DataFrame({"mean": m, "std": s, "n": nn}).dropna(subset=["mean"])
    r = r[r["n"] >= 2]
    r["rel_std_pct"] = 100 * r["std"] / r["mean"].abs()
    r.insert(0, "table", name)
    return r.reset_index()


# ===========================================================================
# LaTeX
# ===========================================================================
# Mean-only mode (STD_IN_LATEX = False): cells whose across-seed std exceeds
# STD_REL x |mean| get STD_MARK, and the caption states the bound (see seeds_note).
STD_REL = 0.05
STD_MARK = r"$^{*}$"


def exceeds(v, s) -> bool:
    """True if std s is above STD_REL of |mean v| (single runs never are)."""
    if v is None or s is None or pd.isna(v) or pd.isna(s) or v == 0:
        return False
    return float(s) > STD_REL * abs(float(v))


def any_exceeds(mean, std) -> bool:
    """True if any cell of the (mean, std) frames/series breaks the relative bound."""
    m = np.asarray(mean.to_numpy() if hasattr(mean, "to_numpy") else mean, dtype=float).ravel()
    sd = np.asarray(std.to_numpy() if hasattr(std, "to_numpy") else std, dtype=float).ravel()
    ok = ~np.isnan(m) & ~np.isnan(sd) & (m != 0)
    return bool(np.any(sd[ok] > STD_REL * np.abs(m[ok])))


def _fmt(v, s, rank):
    if pd.isna(v):
        return "--"
    txt = f"{v:.{DEC}f}"
    if STD_IN_LATEX and s is not None and not pd.isna(s):
        # "<" must be in math mode: in text mode (OT1 encoding) it prints as an inverted "!"
        sd = f"{s:.{DEC}f}" if s >= 0.5 * 10 ** -DEC else "$<$" + f"{10 ** -DEC:.{DEC}f}"
        txt += r"{\scriptsize$\,\pm\,$" + sd + "}"
    if rank == 1:
        txt = r"\textbf{" + txt + "}"
    elif rank == 2:
        txt = r"\underline{" + txt + "}"
    if not STD_IN_LATEX and exceeds(v, s):
        txt += STD_MARK
    return txt


# How best (bold) / second best (underline) are assigned per column:
#   "unrounded": rank the full-precision means -> exactly one best and one second;
#   "shared"   : rank the means rounded to DEC -> entries that PRINT the same
#                share the mark (several bold / underlined cells possible).
TIE_MODE = "unrounded"


def _ranks(mean, rows):
    """1 = best (lowest), 2 = second, per column, over the displayed rows."""
    out = {}
    for c in mean.columns:
        col = mean.loc[[r for r in rows if r in mean.index], c].dropna()
        if TIE_MODE == "shared":
            col = col.round(DEC)
            uniq = sorted(col.unique())
            out[c] = {m: (1 if v == uniq[0] else 2 if len(uniq) > 1 and v == uniq[1] else 99)
                      for m, v in col.items()}
        else:
            order = col.sort_values(kind="mergesort").index      # stable: row order breaks exact ties
            out[c] = {m: (i + 1 if i < 2 else 99) for i, m in enumerate(order)}
    return out


def _env(caption, label, colspec, body, resize=False, placement="ht"):
    tab = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", *body, r"\bottomrule", r"\end{tabular}"]
    if resize:
        tab = [r"\resizebox{\linewidth}{!}{%", *tab, "}"]
    return "\n".join([rf"\begin{{table}}[{placement}]", r"\centering",
                      rf"\caption{{{caption}}}", rf"\label{{{label}}}", *tab, r"\end{table}", ""])


def latex_table(mean, std, caption, label, groups, corner="", top_labels=None,
                sub_labels=None, rank_rows=None, resize=False, rank=True):
    """Generic table: rows = models in `groups` (\\midrule between groups),
    columns = 1- or 2-level index. `top_labels`/`sub_labels` map column keys
    to LaTeX headers."""
    top_labels = top_labels or {}
    sub_labels = sub_labels or (lambda c: str(c))
    rows = [m for g in groups for m in g if m in mean.index]
    ranks = _ranks(mean, rank_rows or rows) if rank else {c: {} for c in mean.columns}
    two = isinstance(mean.columns, pd.MultiIndex)
    body = []
    if two:
        tops = list(dict.fromkeys(mean.columns.get_level_values(0)))
        subs = {t: [c for c in mean.columns if c[0] == t] for t in tops}
        head, cmid, start = [corner], [], 2
        for t in tops:
            k = len(subs[t])
            head.append(rf"\multicolumn{{{k}}}{{c}}{{{top_labels.get(t, t)}}}")
            cmid.append(rf"\cmidrule(lr){{{start}-{start + k - 1}}}")
            start += k
        body += [" & ".join(head) + r" \\", " ".join(cmid),
                 " & ".join([""] + [sub_labels(c[1]) for c in mean.columns]) + r" \\", r"\midrule"]
        colspec = "l" + "".join(" " + "c" * len(subs[t]) for t in tops)
    else:
        body += [" & ".join([corner] + [top_labels.get(c, sub_labels(c)) for c in mean.columns]) + r" \\",
                 r"\midrule"]
        colspec = "l" + "c" * len(mean.columns)
    first = True
    for g in groups:
        g = [m for m in g if m in mean.index]
        if not g:
            continue
        if not first:
            body.append(r"\midrule")
        first = False
        for m in g:
            cells = [C.LABELS.get(m, m)] + [
                _fmt(mean.loc[m, c], std.loc[m, c] if std is not None else None, ranks[c].get(m, 99))
                for c in mean.columns]
            body.append(" & ".join(cells) + r" \\")
    return _env(caption, label, colspec, body, resize=resize)


def latex_spectral_param(mean, std, caption, label, groups, param_label, param):
    """Rows = (model, band) with \\multirow model names.  [S14, S17]"""
    params = list(dict.fromkeys(mean.columns.get_level_values(1)))
    fmt_p = C.PARAM_LABELS.get(param, lambda v: f"{v:g}")
    rows = [m for g in groups for m in g if m in mean.index]
    ranks = _ranks(mean, rows)
    body = [rf"Model & {param_label} & " + " & ".join(fmt_p(p) for p in params) + r" \\", r"\midrule"]
    first = True
    for g in groups:
        for m in [m for m in g if m in mean.index]:
            if not first:
                body.append(r"\midrule")
            first = False
            for bi, b in enumerate(C.BANDS):
                lead = rf"\multirow{{3}}{{*}}{{{C.LABELS[m]}}}" if bi == 0 else ""
                cells = [lead, b] + [_fmt(mean.loc[m, (b, p)], std.loc[m, (b, p)], ranks[(b, p)].get(m, 99))
                                     for p in params]
                body.append(" & ".join(cells) + r" \\")
    return _env(caption, label, "ll" + "c" * len(params), body, resize=True)


def latex_selection(mean, std, caption, label, groups):
    """Table 1: bold = selected wavelet within each WFM scale (not global best)."""
    sel = {(p, k) for (p, _), (k, _) in select_wavelets(mean).items()}
    body = [" & " + " & ".join(C.PDE_LABELS[p] for p in mean.columns) + r" \\", r"\midrule"]
    first = True
    for g in groups:
        g = [m for m in g if m in mean.index]
        if not g:
            continue
        if not first:
            body.append(r"\midrule")
        first = False
        for m in g:
            cells = [C.LABELS[m]]
            for p in mean.columns:
                c = _fmt(mean.loc[m, p], std.loc[m, p], 99)
                cells.append(r"\textbf{" + c + "}" if (p, m) in sel else c)
            body.append(" & ".join(cells) + r" \\")
    return _env(caption, label, "l" + "c" * len(mean.columns), body)


# ===========================================================================
# export
# ===========================================================================
def save(name, mean, std, n, tex=None):
    """Write tex + a tidy CSV (mean, std, n per cell) to artifacts/tables/."""
    ensure_artifact_dirs()
    if tex is not None:
        (TABLES_TEX / f"{name}.tex").write_text(tex)
    nl = list(range(mean.columns.nlevels))
    long = pd.DataFrame({
        "mean": mean.stack(nl, future_stack=True),
        "std": std.stack(nl, future_stack=True),
        "n_seeds": n.stack(nl, future_stack=True),
    }).dropna(subset=["mean"])
    long.index = long.index.set_names(["model"] + [f"col{i}" for i in nl])
    long = long.reset_index()
    long.insert(1, "label", long["model"].map(C.LABELS))
    long.to_csv(TABLES_CSV / f"{name}.csv", index=False, float_format="%.6g")
    return long


def show(mean, std=None):
    """Plain-text mean ± std frame for notebook display."""
    disp = mean.rename(index=C.LABELS).copy().astype(object)
    for i, m in enumerate(mean.index):
        for j, c in enumerate(mean.columns):
            v = mean.iat[i, j]
            s = None if std is None else std.iat[i, j]
            disp.iat[i, j] = "--" if pd.isna(v) else (
                f"{v:.{DEC}f}" + ("" if s is None or pd.isna(s) else f" ± {s:.{DEC + 1}f}"))
    return disp
