"""One function per table of the camera-ready paper.

Each function builds its table from the loaded metric frames, writes
``artifacts/tables/tex/<name>.tex`` + ``artifacts/tables/csv/<name>.csv``,
records the cell-wise seed spread, and returns ``(mean, std, n)`` frames.
They are called identically by ``notebooks/01_tables.ipynb`` and
``scripts/reproduce_tables.py`` (via :func:`build_all`), so both produce
byte-identical artifacts.

    from waveletflux.analysis import paper_tables as P
    df, gdf = P.load_all()
    mean, std, n = P.t1_wavelet_selection(df)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import catalog as C
from . import tables as T
from .ic_sensitivity import ic_tables, load_ic
from .paths import CHECKS, ensure_artifact_dirs
from .physics import ke_table
from .profiling import load_profiles, selected

def set_show_std(show: bool) -> None:
    """True: cells read "mean +- std" over seeds; False: mean over seeds only.
    Captions follow automatically. CSVs always keep mean, std and n_seeds."""
    T.STD_IN_LATEX = bool(show)


def seeds_note() -> str:
    """Caption sentence describing the cells; follows tables.STD_IN_LATEX."""
    if T.STD_IN_LATEX:
        return r" Mean\,$\pm$\,std over three inference seeds (42, 84, 168)."
    return r" Mean over three inference seeds (42, 84, 168)."
DET_NOTE = r" Deterministic baselines (WNO, FNO, TFNO) are single runs."
LF_NOTE = r" LatentFM$_{\div 64}$ is evaluated on Rayleigh--B\'enard only."

_SPREADS: dict[str, pd.DataFrame] = {}     # name -> per-cell seed spread (re-running a table overwrites)
VERBOSE = True


def _emit(name, res, tex):
    mean, std, n = res
    T.save(name, mean, std, n, tex)
    _SPREADS[name] = T.spread_report(name, mean, std, n)
    if VERBOSE:
        print(f"  wrote {name}")
    return res


def load_all(verbose=True):
    """Load every metric file; write artifacts/checks/coverage.csv."""
    ensure_artifact_dirs()
    df, gdf = C.load(verbose=verbose)
    C.coverage(pd.concat([df, gdf])).to_csv(CHECKS / "coverage.csv", index=False)
    return df, gdf


# ---------------------------------------------------------------------------
# Main text
# ---------------------------------------------------------------------------
def t1_wavelet_selection(df):
    """Table 1 (+ S10 csv): band- and horizon-averaged spectral coherence RMSE."""
    sel = T.across_seeds(T.selection_table, df, models=C.ROWS_GEN, time_agg="horizon")
    bad = T.check_selection(sel[0])
    if bad:
        raise ValueError("wavelet selection changed: " + "; ".join(bad))
    _emit("T1_wavelet_selection", sel, T.latex_selection(
        sel[0], sel[1],
        r"Spectral Coherence RMSE averaged across \textit{Low}, \textit{Mid} and \textit{High} "
        r"frequency bands and rollout horizons per PDE system. For each WFM variant, \textbf{bold} "
        r"denotes the optimal wavelet representation used in all subsequent experiments." + LF_NOTE + seeds_note(),
        "tab:avg_sc_rmse_gen_wavelets", C.GROUPS_GEN))
    s10 = T.s10_table(sel[0], sel[1])
    s10.to_csv(T.TABLES_CSV / "S10_optimal_wavelets.csv", index=False)
    (T.TABLES_TEX / "S10_optimal_wavelets.tex").write_text(latex_s10(s10))
    if VERBOSE:
        print("  wrote S10_optimal_wavelets")
    return sel


S10_HEADERS = {"turbulent_radiative_layer_2D": "Turbulent Radiative Layer 2D",
               "rayleigh_benard": r"Rayleigh--B\'{e}nard",
               "active_matter": "Active Matter"}


def _wavelet_tex(w):
    return r"$\mathrm{haar}$" if w == "haar" else rf"$\mathrm{{db}}{w[2:]}$"


def latex_s10(s10):
    """Table S10 in the manuscript's layout: rows = WFM 1-/3-scale, columns =
    PDEs, cell = selected wavelet (mean over seeds, +- std if STD_IN_LATEX)."""
    body = [" & " + " & ".join(S10_HEADERS[p] for p in C.PDES) + r" \\", r"\midrule"]
    for scale, name in (("1scale", "WFM 1-scale"), ("3scale", "WFM 3-scale")):
        cells = [name]
        for pde in C.PDES:
            r = s10[(s10.pde == pde) & (s10.scale == scale)].iloc[0]
            cells.append(f"{_wavelet_tex(r.wavelet)} ({T._fmt(r['mean'], r['std'], 99)})")
        body.append(" & ".join(cells) + r" \\")
    cap = (r"Optimal mother wavelet per WFM variant and PDE system, selected by lowest spectral "
           r"coherence RMSE averaged across \textit{Low}, \textit{Mid}, and \textit{High} frequency "
           r"bands and rollout horizons (Table~\ref{tab:avg_sc_rmse_gen_wavelets}). In parentheses: "
           + (r"the selected wavelet's score, mean\,$\pm$\,std over three inference seeds (42, 84, 168); "
              if T.STD_IN_LATEX else
              r"the selected wavelet's score, mean over three inference seeds (42, 84, 168); ")
           + r"the selection is made on the seed mean.")
    return T._env(cap, "si-tab:optimal_wavelets", "lccc", body)


def t2_s19_s20_profiling():
    """Table 2 (tex) and S19 / S20 (csv): inference speed and memory."""
    prof = load_profiles()
    prof.to_csv(T.TABLES_CSV / "S20_profiling_all_variants.csv", index=False, float_format="%.6g")
    sel19 = selected(prof)
    sel19.to_csv(T.TABLES_CSV / "S19_profiling_selected.csv", index=False, float_format="%.6g")
    t2 = []
    for pde in C.PDES:
        r = sel19[sel19.pde == pde].set_index("model")
        k1 = C.wfm_key("1scale", C.BEST_WFM_PER_PDE[pde]["1scale"])
        k3 = C.wfm_key("3scale", C.BEST_WFM_PER_PDE[pde]["3scale"])
        t2.append(rf"{C.PDE_SHORT[pde]} & {r.loc[k1, 'speedup']:.2f}$\times$ & {r.loc[k3, 'speedup']:.2f}$\times$ \\")
    (T.TABLES_TEX / "T2_inference_speedup.tex").write_text("\n".join([
        r"\begin{tabular}{lcc}", r"\toprule", r"\textbf{PDE} & \textbf{WFM 1-scale} & \textbf{WFM 3-scale} \\",
        r"\midrule", *t2, r"\bottomrule", r"\end{tabular}", ""]))
    if VERBOSE:
        print("  wrote T2/S19/S20 profiling")
    return prof


# ---------------------------------------------------------------------------
# Appendix: horizons and bands
# ---------------------------------------------------------------------------
def s11_vrmse(df):
    r = T.across_seeds(T.horizon_table, df, metric="vrmse", models=C.ROWS_ALL)
    return _emit("S11_vrmse", r, T.latex_table(
        *r[:2], "VRMSE across rollout horizons for all models and PDE systems." + DET_NOTE + LF_NOTE + seeds_note(),
        "si-tab:appendix_vrmse", C.GROUPS_ALL, corner="Rollout", top_labels=C.PDE_LABELS, resize=True))


def s12_crps(df):
    r = T.across_seeds(T.horizon_table, df, metric="crps", models=C.ROWS_GEN)
    return _emit("S12_crps", r, T.latex_table(
        *r[:2], "CRPS across rollout horizons for all models and PDE systems." + LF_NOTE + seeds_note(),
        "tab:appendix_crps", C.GROUPS_GEN, corner="Rollout", top_labels=C.PDE_LABELS, resize=True))


def s13_spectral_bands(df):
    r = T.across_seeds(T.bands_table, df, models=C.ROWS_GEN, time_agg="horizon")
    return _emit("S13_spectral_bands", r, T.latex_table(
        *r[:2], r"Spectral coherence RMSE across \textit{Low}, \textit{Mid}, and \textit{High} "
        r"frequency bands for all models and PDE systems, averaged over rollout horizons." + LF_NOTE + seeds_note(),
        "tab:appendix_spectral", C.GROUPS_GEN, corner="Frequency Bands",
        top_labels=C.PDE_LABELS, resize=True))


# ---------------------------------------------------------------------------
# Physical-parameter breakdowns (Table 3 = first columns of S14 / S16)
# ---------------------------------------------------------------------------
_PARAM = {
    "S14_crps_tcool": ("turbulent_radiative_layer_2D", "tcool", "tab:ablation_trl_crps_tcool",
                       r"CRPS per $t_{\mathrm{cool}}$ on TRL, averaged over rollout steps.",
                       r"$t_{\mathrm{cool}}$", "tab:ablation_trl_crps_tcool_main", "T3_crps_tcool_main"),
    "S16_crps_Ra": ("rayleigh_benard", "Rayleigh", "tab:ablation_rb_crps_ra",
                    r"CRPS per Rayleigh number $Ra$ on Rayleigh--B\'enard, averaged over Prandtl numbers.",
                    r"$Ra$", "tab:ablation_rb_crps_ra_main", "T3_crps_Ra_main"),
    "S17_crps_Pr": ("rayleigh_benard", "Prandtl", "tab:ablation_rb_crps_pr",
                    r"CRPS per Prandtl number $Pr$ on Rayleigh--B\'enard, averaged over Rayleigh numbers.",
                    r"$Pr$", None, None),
}


def _crps_param(df, name):
    pde, param, lab, cap, corner, lab3, name3 = _PARAM[name]
    r = T.across_seeds(T.param_table, df, pde=pde, param=param, models=C.ROWS_GEN, time_agg="step")
    fmt = C.PARAM_LABELS[param]
    _emit(name, r, T.latex_table(*r[:2], cap + seeds_note(), lab, C.GROUPS_GEN, corner=corner,
                                 sub_labels=fmt, resize=param == "tcool"))
    if name3:                                   # main-text Table 3 subtable: first three values
        m, s, n = r
        cols = list(m.columns)[:3]
        sub = (m[cols], s[cols], n[cols])
        _emit(name3, sub, T.latex_table(*sub[:2], cap + seeds_note(), lab3, C.GROUPS_GEN,
                                        corner=corner, sub_labels=fmt))
    return r


def s14_crps_tcool(df):
    """S14, and Table 3a (its first three t_cool values)."""
    return _crps_param(df, "S14_crps_tcool")


def s16_crps_ra(df):
    """S16, and Table 3b."""
    return _crps_param(df, "S16_crps_Ra")


def s17_crps_pr(df):
    return _crps_param(df, "S17_crps_Pr")


_SPEC = {
    "S15_spectral_tcool": ("turbulent_radiative_layer_2D", "tcool", "tab:ablation_trl_spectral_tcool",
                           r"Spectral coherence RMSE per $t_{\mathrm{cool}}$ and frequency band on TRL.", r"$t_{cool}$"),
    "S18a_spectral_Ra": ("rayleigh_benard", "Rayleigh", "tab:ablation_rb_spectral_ra",
                         r"Spectral coherence RMSE per frequency band on Rayleigh--B\'enard vs $Ra$ (averaged over $Pr$).", r"$Ra$"),
    "S18b_spectral_Pr": ("rayleigh_benard", "Prandtl", "tab:ablation_rb_spectral_pr",
                         r"Spectral coherence RMSE per frequency band on Rayleigh--B\'enard vs $Pr$ (averaged over $Ra$).", r"$Pr$"),
}


def _spectral_param(df, name):
    pde, param, lab, cap, plab = _SPEC[name]
    r = T.across_seeds(T.spectral_param_table, df, pde=pde, param=param, models=C.ROWS_GEN, time_agg="step")
    return _emit(name, r, T.latex_spectral_param(*r[:2], cap + seeds_note(), lab, C.GROUPS_GEN, plab, param))


def s15_spectral_tcool(df):
    return _spectral_param(df, "S15_spectral_tcool")


def s18a_spectral_ra(df):
    return _spectral_param(df, "S18a_spectral_Ra")


def s18b_spectral_pr(df):
    return _spectral_param(df, "S18b_spectral_Pr")


# ---------------------------------------------------------------------------
# Rebuttal additions
# ---------------------------------------------------------------------------
def r1_rb_latentfm(df):
    """RB summary as shown in the rebuttal (selected wavelets + generative baselines).
    Optional: LatentFM is now also a row of every table with an RB column."""
    rows = C.ROWS_BEST_GEN
    r = T.across_seeds(T.rb_summary_table, T.best_view(df, keep=C.GEN_MODELS), models=rows)
    return _emit("R1_rb_latentfm", r, T.latex_table(
        *r[:2], r"Rayleigh--B\'enard: spectral coherence RMSE per band (averaged over rollout horizons) "
        r"and CRPS per rollout horizon, including the latent-space baseline LatentFM$_{\div 64}$ "
        r"(autoencoder of~\citet{rozet2025})." + seeds_note(),
        "tab:rb_latentfm", [C.BEST, C.GEN_MODELS], corner="Model", resize=True))


def r2_ke_mae(gdf):
    """Kinetic-energy relative MAE per horizon (uses the _global_ rows)."""
    r = ke_table(T.best_view(gdf, keep=C.GEN_MODELS), C.ROWS_BEST_GEN)
    return _emit("R2_ke_mae", r, T.latex_table(
        *r[:2], r"Kinetic-energy relative MAE across rollout horizons (error normalised by the "
        r"trajectory time-mean of the true kinetic energy)." + seeds_note(),
        "tab:ke_mae", C.GROUPS_BEST_GEN, corner="Rollout", top_labels=C.PDE_LABELS, resize=True))


def r3_ic_perturbation():
    """TRL, WFM 3-scale haar under IC noise: CRPS per horizon and spectral per band."""
    ic = ic_tables(load_ic())
    out = {}
    for key, lab, cap in [("crps", "tab:ic_pert_crps", "CRPS across rollout horizons"),
                          ("spectral", "tab:ic_pert_spectral", "Spectral coherence RMSE across bands")]:
        m, s = ic[key]
        n = pd.DataFrame(np.broadcast_to(ic["n_seeds"].to_numpy()[:, None], m.shape).astype(float),
                         index=m.index, columns=m.columns)
        m.index = [f"$\\eta={e:g}$" for e in m.index]
        s.index = m.index
        n.index = m.index
        T.save(f"R3_ic_{key}", m, s, n)
        tex = T.latex_table(m, s, f"TRL, WFM 3-scale (haar): {cap} under Gaussian perturbations of the "
                            r"initial conditions with standard deviation $\eta\,\sigma_c$."
                            + seeds_note(), lab, [list(m.index)],
                            corner="IC noise", rank=False)
        (T.TABLES_TEX / f"R3_ic_{key}.tex").write_text(tex)
        if VERBOSE:
            print(f"  wrote R3_ic_{key}")
        out[key] = (m, s, n)
    return out


# ---------------------------------------------------------------------------
# Audit + everything
# ---------------------------------------------------------------------------
TABLE_ORDER = ["T1_wavelet_selection", "S11_vrmse", "S12_crps", "S13_spectral_bands",
               "S14_crps_tcool", "T3_crps_tcool_main", "S16_crps_Ra", "T3_crps_Ra_main", "S17_crps_Pr",
               "S15_spectral_tcool", "S18a_spectral_Ra", "S18b_spectral_Pr", "R1_rb_latentfm", "R2_ke_mae"]


def write_seed_spread():
    """artifacts/checks/seed_spread{,_summary}.csv from every table built so far."""
    names = [n for n in TABLE_ORDER if n in _SPREADS] + [n for n in _SPREADS if n not in TABLE_ORDER]
    sp = pd.concat([_SPREADS[n] for n in names], ignore_index=True)
    sp.to_csv(CHECKS / "seed_spread.csv", index=False, float_format="%.6g")
    summ = sp.groupby("table")["rel_std_pct"].agg(
        cells="size", median="median", max="max",
        pct_over_0p5=lambda x: 100 * (x > 0.5).mean(), pct_over_1=lambda x: 100 * (x > 1).mean())
    summ.round(2).to_csv(CHECKS / "seed_spread_summary.csv")
    return summ.round(2)


def build_all(verbose=True, show_std=True):
    """Build every table in paper order; returns the seed-spread summary.
    show_std=False writes the seed mean only (see set_show_std)."""
    set_show_std(show_std)
    df, gdf = load_all(verbose)
    t1_wavelet_selection(df)
    s11_vrmse(df)
    s12_crps(df)
    s13_spectral_bands(df)
    s14_crps_tcool(df)
    s16_crps_ra(df)
    s17_crps_pr(df)
    s15_spectral_tcool(df)
    s18a_spectral_ra(df)
    s18b_spectral_pr(df)
    r1_rb_latentfm(df)
    r2_ke_mae(gdf)
    r3_ic_perturbation()
    t2_s19_s20_profiling()
    return write_seed_spread()
