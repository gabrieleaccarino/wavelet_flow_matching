"""
Table and figure helpers for the physics metrics.

Turns the long-format DataFrames from `physics.py` into the two artefacts
you actually need for the rebuttal: a lead-time-banded LaTeX table in the
style of Table 1, and per-metric rollout figures.

Aggregation order matters and is fixed here: metrics are already per-member
in `physics.py`, so the only reduction left is over (traj, step-within-band).
That is a plain mean; it is NOT a re-averaging of fields.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

# Default lead-time bands, matching the a:b convention in the paper.
# Override per dataset -- TRL has 101 stored steps, RB 200, AM 81.
DEFAULT_BANDS: dict[str, tuple[int, int]] = {
    "01:20":  (0, 19),
    "21:60":  (20, 59),
    "61:100": (60, 99),
}

# Metrics where lower is better (essentially all of them; listed explicitly
# so `to_latex` knows which direction to bold).
LOWER_IS_BETTER = {
    "ke_bias", "ke_mae", "urms_bias", "urms_mae",
    "enstrophy_bias", "enstrophy_mae",
    "palinstrophy_bias", "palinstrophy_mae",
    "gradsq_bias", "gradsq_mae", "diss_bias", "diss_mae",
    "mass_bias", "mass_mae", "absmax_bias", "absmax_mae",
    "p01_bias", "p99_bias",
    "w1_pdf", "w1_pdf_pooled", "w1_vort", "w1_vort_pooled", "w1_logT",
    "nusselt_bias", "nusselt_mae", "eps_b_bias", "eps_u_bias",
    "reynolds_bias", "massflux_bias", "cooling_bias", "cold_frac_bias",
    "nematic_bias", "activepow_bias", "conc_var_bias",
    "neg_frac_pred",
}

# Human-readable names for table headers.
PRETTY = {
    "ke_bias":            r"$E_k$",
    "enstrophy_bias":     r"$Z$",
    "palinstrophy_bias":  r"$P$",
    "gradsq_bias":        r"$\langle|\nabla f|^2\rangle$",
    "diss_bias":          r"$\varepsilon_f$",
    "mass_bias":          r"$\int f$",
    "w1_pdf":             r"$W_1(p(f))$",
    "w1_vort":            r"$W_1(p(\omega))$",
    "w1_logT":            r"$W_1(p_M(\log T))$",
    "divnorm_ratio":      r"$\|\nabla\!\cdot\!u\|$ ratio",
    "continuity_ratio":   r"$\mathcal{R}_\rho$ ratio",
    "nusselt_bias":       r"$\mathrm{Nu}$",
    "eps_b_bias":         r"$\varepsilon_b$",
    "eps_u_bias":         r"$\varepsilon_u$",
    "massflux_bias":      r"$\langle\rho u_y\rangle$",
    "cold_frac_bias":     r"$f_{\rm cold}$",
    "nematic_bias":       r"$S$",
    "activepow_bias":     r"$\langle D\!:\!E\rangle$",
    "neg_frac_pred":      r"neg.\ frac.",
}


# ══════════════════════════════════════════════════════════════════════════════
# Aggregation
# ══════════════════════════════════════════════════════════════════════════════

def add_band(df: pd.DataFrame,
             bands: Mapping[str, tuple[int, int]] = DEFAULT_BANDS) -> pd.DataFrame:
    """Attach a categorical `band` column from the `step` column."""
    df = df.copy()
    band = pd.Series(pd.NA, index=df.index, dtype="object")
    for name, (lo, hi) in bands.items():
        band[(df["step"] >= lo) & (df["step"] <= hi)] = name
    df["band"] = pd.Categorical(band, categories=list(bands), ordered=True)
    return df


def summarize(
    df:      pd.DataFrame,
    metrics: Sequence[str],
    bands:   Mapping[str, tuple[int, int]] = DEFAULT_BANDS,
    by:      Sequence[str] = ("model",),
    ci:      bool = True,
) -> pd.DataFrame:
    """
    Mean (and bootstrap-free standard error over trajectories) of each metric,
    grouped by `by` and lead-time band.

    Works for both `build_physics_field_df` (pass by=("model","field")) and
    `build_physics_global_df` (pass by=("model",)).

    Returns a wide frame indexed by `by` with a (metric, band) column
    MultiIndex, ready for `to_latex` or direct display.
    """
    d = add_band(df, bands).dropna(subset=["band"])
    keys = list(by) + ["band"]

    agg = {m: ["mean", "sem"] for m in metrics if m in d.columns}
    if not agg:
        raise KeyError(f"none of {list(metrics)} present; "
                       f"available: {sorted(c for c in d.columns)}")

    g = d.groupby(keys, observed=True).agg(agg)
    g.columns = pd.MultiIndex.from_tuples(
        [(m, s) for m, s in g.columns], names=["metric", "stat"])
    wide = g.unstack("band")
    if not ci:
        wide = wide.xs("mean", level="stat", axis=1)
    return wide.sort_index(axis=1)


def rollout_curve(
    df:      pd.DataFrame,
    metric:  str,
    by:      Sequence[str] = ("model",),
) -> pd.DataFrame:
    """
    Metric vs lead time, averaged over trajectories, with the across-traj
    standard error. Feed straight into a line plot with error bands.
    """
    keys = list(by) + ["step"]
    g = df.groupby(keys, observed=True)[metric].agg(["mean", "sem", "count"])
    return g.reset_index()


# ══════════════════════════════════════════════════════════════════════════════
# LaTeX
# ══════════════════════════════════════════════════════════════════════════════

def to_latex(
    summary:   pd.DataFrame,
    metrics:   Sequence[str],
    bands:     Sequence[str] | None = None,
    bold_best: bool = True,
    precision: int = 3,
    caption:   str = "",
    label:     str = "",
) -> str:
    """
    Render a `summarize(..., ci=False)` output as a booktabs table with the
    best entry per (metric, band) column in bold -- same visual grammar as
    Table 1 in the paper.

    Pass the frame from `summarize(..., ci=False)`.
    """
    if isinstance(summary.columns, pd.MultiIndex):
        cols = [(m, b) for m in metrics for b in
                (bands or summary.columns.get_level_values(-1).unique())
                if (m, b) in summary.columns]
    else:
        cols = [m for m in metrics if m in summary.columns]

    sub = summary[cols]
    fmt = f"{{:.{precision}f}}"
    body = sub.map(lambda v: fmt.format(v) if pd.notna(v) else "--")

    if bold_best:
        for c in sub.columns:
            col = sub[c]
            name = c[0] if isinstance(c, tuple) else c
            if col.notna().sum() == 0:
                continue
            best = col.idxmin() if name in LOWER_IS_BETTER else col.idxmax()
            body.loc[best, c] = r"\textbf{" + body.loc[best, c] + "}"

    if isinstance(sub.columns, pd.MultiIndex):
        body.columns = pd.MultiIndex.from_tuples(
            [(PRETTY.get(m, m.replace('_', r'\_')), b) for m, b in sub.columns])
    else:
        body.columns = [PRETTY.get(m, m.replace('_', r'\_')) for m in sub.columns]

    tex = body.to_latex(escape=False, multicolumn_format="c",
                        column_format="l" + "r" * len(body.columns))
    tex = (tex.replace(r"\toprule", r"\toprule")
              .replace("\\begin{tabular}", "\\begin{tabular}"))
    if caption or label:
        tex = ("\\begin{table}[t]\n\\centering\n" + tex
               + (f"\\caption{{{caption}}}\n" if caption else "")
               + (f"\\label{{{label}}}\n" if label else "")
               + "\\end{table}\n")
    return tex


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

def plot_rollout(
    df:       pd.DataFrame,
    metrics:  Sequence[str],
    by:       str = "model",
    logy:     bool = True,
    ncols:    int = 3,
    figsize_per: tuple[float, float] = (3.4, 2.6),
    ax_titles: Mapping[str, str] | None = None,
):
    """
    Grid of "metric vs lead time" panels, one per metric, one line per model,
    shaded by the across-trajectory standard error.

    This is the figure to put next to the spectral-coherence plot: it shows
    a physical quantity degrading with rollout length, which is far more
    legible to a physics reviewer than a band-averaged RMSE.

    Returns (fig, axes). Requires matplotlib.
    """
    import matplotlib.pyplot as plt

    metrics = [m for m in metrics if m in df.columns]
    nrows = int(np.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, squeeze=False,
        figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows))

    for ax, metric in zip(axes.flat, metrics):
        curve = rollout_curve(df, metric, by=(by,))
        for name, sub in curve.groupby(by, observed=True):
            ax.plot(sub["step"], sub["mean"], label=str(name), lw=1.6)
            ax.fill_between(sub["step"],
                            sub["mean"] - sub["sem"],
                            sub["mean"] + sub["sem"], alpha=0.18, lw=0)
        ax.set_title((ax_titles or {}).get(metric, metric), fontsize=9)
        ax.set_xlabel("lead time")
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.25, lw=0.5)

    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig, axes


def plot_conservation(
    df:      pd.DataFrame,
    ratio:   str = "divnorm_ratio",
    by:      str = "model",
):
    """
    Single-panel conservation-residual figure with the ground-truth level at
    1.0 marked. Values near 1 mean the emulator violates the constraint no
    more than the reference solver does; values >> 1 are the headline.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    curve = rollout_curve(df, ratio, by=(by,))
    for name, sub in curve.groupby(by, observed=True):
        ax.plot(sub["step"], sub["mean"], label=str(name), lw=1.6)
        ax.fill_between(sub["step"], sub["mean"] - sub["sem"],
                        sub["mean"] + sub["sem"], alpha=0.18, lw=0)
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="reference solver")
    ax.set_yscale("log")
    ax.set_xlabel("lead time")
    ax.set_ylabel(PRETTY.get(ratio, ratio))
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    return fig, ax
