"""One function per table of the camera-ready paper.

Each function builds its table from the loaded metric frames, writes
``artifacts/tables/tex/<name>.tex`` + ``artifacts/tables/csv/<name>.csv``,
records the cell-wise seed spread, and returns ``(mean, std, n)`` frames.
They are called identically by ``notebooks/01_tables.ipynb`` and
``scripts/reproduce_tables.py`` (via :func:`build_all`), so both produce
byte-identical artifacts.

    from wavelet_flow_matching.analysis import paper_tables as P
    df, gdf = P.load_all()
    mean, std, n = P.t1_wavelet_selection(df)
"""
from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd

from . import catalog as C
from . import tables as T
from .ic_sensitivity import ic_tables, load_ic
from .paths import CHECKS, ensure_artifact_dirs
from .physics import ke_table
from .profiling import load_profiles, selected

def set_show_std(show: bool) -> None:
    """Default for every table: True -> cells read "mean +- std" over seeds,
    False -> mean over seeds only. Each table function also takes its own
    ``show_std`` argument, which overrides this default for that table only.
    Captions follow automatically. CSVs always keep mean, std and n_seeds."""
    global DEFAULT_SHOW_STD
    DEFAULT_SHOW_STD = bool(show)


DEFAULT_SHOW_STD = True


@contextlib.contextmanager
def _std(show, rel=None):
    """For one table: set the +- std switch (None = DEFAULT_SHOW_STD) and, in
    mean-only mode, the relative std bound (fraction of the mean, None = keep 0.05)."""
    saved = T.STD_IN_LATEX, T.STD_REL
    T.STD_IN_LATEX = DEFAULT_SHOW_STD if show is None else bool(show)
    if rel is not None:
        T.STD_REL = float(rel)
    try:
        yield
    finally:
        T.STD_IN_LATEX, T.STD_REL = saved


def seeds_note(mean=None, std=None) -> str:
    """Caption sentence describing the cells. With std shown: "Mean +- std ...".
    Mean only: states the relative std bound, and mentions the * mark only if
    some cell of (mean, std) actually exceeds it."""
    if T.STD_IN_LATEX:
        return r" Mean\,$\pm$\,std over three inference seeds (42, 84, 168)."
    pct = f"{100 * T.STD_REL:g}"
    base = (r" Mean over three inference seeds (42, 84, 168); the standard deviation across seeds "
            rf"is below {pct}\% of the mean for all entries")
    if mean is not None and std is not None and T.any_exceeds(mean, std):
        return base + rf" except those marked {T.STD_MARK}."
    return base + "."


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
def t1_wavelet_selection(df, show_std=None, show_std_s11=None, std_rel=None, std_rel_s11=None):
    """Table 1 and Table S11: band- and horizon-averaged spectral coherence RMSE.
    show_std: Table 1 cells; show_std_s11: S9 cells (defaults to show_std).
    std_rel(_s9): bound stated in the caption in mean-only mode."""
    sel = T.across_seeds(T.selection_table, df, models=C.ROWS_GEN, time_agg="horizon")
    bad = T.check_selection(sel[0])
    if bad:
        raise ValueError("wavelet selection changed: " + "; ".join(bad))
    with _std(show_std, std_rel):
        _emit("T1_wavelet_selection", sel, T.latex_selection(
            sel[0], sel[1],
            r"Spectral Coherence RMSE averaged across \textit{Low}, \textit{Mid} and \textit{High} "
            r"frequency bands and rollout horizons per PDE system. For each WFM variant, \textbf{bold} "
            r"denotes the optimal wavelet representation used in all subsequent experiments." + LF_NOTE + seeds_note(sel[0], sel[1]),
            "tab:avg_sc_rmse_gen_wavelets", C.GROUPS_GEN))
    s9 = T.s9_table(sel[0], sel[1])
    s9.to_csv(T.TABLES_CSV / "S11_optimal_wavelets.csv", index=False)
    with _std(show_std if show_std_s11 is None else show_std_s11,
              std_rel if std_rel_s11 is None else std_rel_s11):
        (T.TABLES_TEX / "S11_optimal_wavelets.tex").write_text(latex_s11(s9))
    if VERBOSE:
        print("  wrote S11_optimal_wavelets")
    return sel


S11_HEADERS = {"turbulent_radiative_layer_2D": "Turbulent Radiative Layer 2D",
               "rayleigh_benard": r"Rayleigh--B\'{e}nard",
               "active_matter": "Active Matter"}


def _wavelet_tex(w):
    return r"$\mathrm{haar}$" if w == "haar" else rf"$\mathrm{{db}}{w[2:]}$"


def latex_s11(s9):
    """Table S11 in the manuscript's layout: rows = WFM 1-/3-scale, columns =
    PDEs, cell = selected wavelet (mean over seeds, +- std if STD_IN_LATEX)."""
    body = [" & " + " & ".join(S11_HEADERS[p] for p in C.PDES) + r" \\", r"\midrule"]
    for scale, name in (("1scale", "WFM 1-scale"), ("3scale", "WFM 3-scale")):
        cells = [name]
        for pde in C.PDES:
            r = s9[(s9.pde == pde) & (s9.scale == scale)].iloc[0]
            cells.append(f"{_wavelet_tex(r.wavelet)} ({T._fmt(r['mean'], r['std'], 99)})")
        body.append(" & ".join(cells) + r" \\")
    cap = (r"Optimal mother wavelet per WFM variant and PDE system, selected by lowest spectral "
           r"coherence RMSE averaged across \textit{Low}, \textit{Mid}, and \textit{High} frequency "
           r"bands and rollout horizons (Table~\ref{tab:avg_sc_rmse_gen_wavelets}). In parentheses: "
           + (r"the selected wavelet's score, mean\,$\pm$\,std over three inference seeds (42, 84, 168); "
              r"the selection is made on the seed mean."
              if T.STD_IN_LATEX else
              r"the selected wavelet's score; the selection is made on the seed mean." + seeds_note(s9["mean"], s9["std"])))
    return T._env(cap, "si-tab:optimal_wavelets", "lccc", body)


def t2_s23_s24_profiling():
    """Table 2 (tex) and S18 / S19 (csv): inference speed and memory."""
    prof = load_profiles()
    prof.to_csv(T.TABLES_CSV / "S24_profiling_all_variants.csv", index=False, float_format="%.6g")
    sel18 = selected(prof)
    sel18.to_csv(T.TABLES_CSV / "S23_profiling_selected.csv", index=False, float_format="%.6g")
    t2 = []
    for pde in C.PDES:
        r = sel18[sel18.pde == pde].set_index("model")
        k1 = C.wfm_key("1scale", C.BEST_WFM_PER_PDE[pde]["1scale"])
        k3 = C.wfm_key("3scale", C.BEST_WFM_PER_PDE[pde]["3scale"])
        t2.append(rf"{C.PDE_SHORT[pde]} & {r.loc[k1, 'speedup']:.2f}$\times$ & {r.loc[k3, 'speedup']:.2f}$\times$ \\")
    (T.TABLES_TEX / "T2_inference_speedup.tex").write_text("\n".join([
        r"\begin{tabular}{lcc}", r"\toprule", r"\textbf{PDE} & \textbf{WFM 1-scale} & \textbf{WFM 3-scale} \\",
        r"\midrule", *t2, r"\bottomrule", r"\end{tabular}", ""]))
    (T.TABLES_TEX / "S23_profiling_selected.tex").write_text(latex_profiling(prof, ablation=False))
    (T.TABLES_TEX / "S24_profiling_all_variants.tex").write_text(latex_profiling(prof, ablation=True))
    if VERBOSE:
        print("  wrote T2/S18/S19 profiling")
    return prof


_WAV_TEX = {"haar": r"$\mathrm{haar}$", "db2": r"$\mathrm{db}2$", "db4": r"$\mathrm{db}4$", "db6": r"$\mathrm{db}6$"}


def _prof_cells(r, base=False):
    cells = [f"{r.wall_clock_s:.1f}", f"{r.sec_per_step:.4f}", f"{r.gpu_mem_GiB:.2f}", f"{r.throughput_Mfr_s:.2f}"]
    if base:
        return cells + [r"\textemdash", r"\textemdash"]
    return cells + [rf"{r.speedup:.2f}$\times$", rf"{r.mem_ratio:.2f}$\times$"]


def latex_profiling(prof, ablation=False):
    """Appendix profiling tables: selected wavelets (``tab:inference_perf``) or all
    wavelets (``tab:inference_perf_ablation``), in the paper's layout."""
    head = [r"\textbf{PDE}", r"\textbf{Model}"] + ([r"\textbf{Wavelet}"] if ablation else []) + [
        r"\textbf{Wall Clock (s)}", r"\textbf{sec / step}", r"\textbf{GPU Mem (GiB)}",
        r"\textbf{Throughput (Mfr/s)}", r"\textbf{Speedup $\times$}", r"\textbf{Mem ratio $\times$}"]
    if ablation:
        cap = r"Complete inference profiling across all WFM variants (scale and choice of the mother wavelet) relative to FM$_{pixel}$."
        lab, cols = "tab:inference_perf_ablation", r"ll c cccc cc"
    else:
        cap = (r"Inference performance across PDEs and models. The reported WFM variants are selected based on the "
               r"procedure in Section~\ref{sec:wfm_eval}, and summarized in Table~\ref{si-tab:optimal_wavelets}.")
        lab, cols = "tab:inference_perf", r"ll rrrr rr"
    lines = [r"\begin{table}[ht]", rf"\caption{{{cap}}}", rf"\label{{{lab}}}", r"\resizebox{\textwidth}{!}{%",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule", " & ".join(head) + r" \\"]
    pix = r"$\text{FM}_{\text{pixel}}$"
    for pde in C.PDES:
        d = prof[prof.pde == pde].set_index("model")
        sel = C.BEST_WFM_PER_PDE[pde]
        lines.append(r"\midrule")
        if not ablation:
            lines.append(rf"\multirow{{3}}{{*}}{{{C.PDE_SHORT[pde]}}}")
            lines.append("  & " + " & ".join([pix] + _prof_cells(d.loc["FM_pixel"], base=True)) + r" \\")
            for sc in ("1scale", "3scale"):
                k = C.wfm_key(sc, sel[sc])
                name = rf"WFM {sc[0]}-scale {{\scriptsize({_WAV_TEX[sel[sc]]})}}"
                lines.append("  & " + " & ".join([name] + _prof_cells(d.loc[k])) + r" \\")
            continue
        lines.append(rf"\multirow{{9}}{{*}}{{{C.PDE_SHORT[pde]}}}")
        lines.append("  & " + " & ".join([pix, "---"] + _prof_cells(d.loc["FM_pixel"], base=True)) + r" \\")
        for sc in ("1scale", "3scale"):
            lines.append(r"  \cmidrule(l){2-9}")
            for i, w in enumerate(C.WAVELETS):
                k = C.wfm_key(sc, w)
                first = rf"\multirow{{4}}{{*}}{{WFM {sc[0]}-scale}}" if i == 0 else ""
                wtxt = rf"\textbf{{{w}}}" if w == sel[sc] else w
                lines.append("  & " + " & ".join([first, wtxt] + _prof_cells(d.loc[k])) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}", ""]
    return "\n".join(lines)


def s2_cross_scale():
    """Cross-scale sensitivity matrix (Appendix B.1) from the JSON written by
    ``05_cross_scale_coupling.ipynb`` (``artifacts/tables/raw``)."""
    import json
    raw = T.TABLES_CSV.parent / "raw" / "cross_scale_sensitivity_trl_wfm3_haar.json"
    r = json.loads(raw.read_text())
    S = np.load(raw.with_suffix(".npz"))["mean"]          # full precision (the JSON is rounded)
    names = ["fine", "mid", "coarse"]
    rows = [rf"    $\hat{{u}}_{i + 1}$ ({names[i]}) & " + " & ".join(f"{v:.3f}" for v in S[i]) + r" \\" for i in range(3)]
    tex = "\n".join([
        r"\begin{table}[ht]", r"  \centering",
        r"  \caption{Cross-scale sensitivity $S_{jj'}$ (Equation~\ref{eq:sensitivity}) of WFM 3-scale (haar) on TRL, "
        rf"with $\delta = 10^{{{int(np.log10(r['delta']))}}}$, averaged over test windows and $\tau$. Rows: affected head $\hat{{u}}_j$; "
        r"columns: perturbed scale $w^{\tau}_{j'}$ ($j=1$ finest). Independent per-scale flows would give zero "
        rf"off-diagonal entries; the measured off-diagonal mass is $\rho = {r['off_diagonal_mass_pct']:.1f}\%$.}}",
        r"  \label{tab:sensitivity}", r"  \begin{tabular}{lccc}", r"    \toprule",
        r"    & $w^{\tau}_1$ (fine) & $w^{\tau}_2$ (mid) & $w^{\tau}_3$ (coarse) \\", r"    \midrule",
        *rows, r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""])
    (T.TABLES_TEX / "S2_cross_scale_sensitivity.tex").write_text(tex)
    pd.DataFrame(S, index=[f"u_{i + 1}" for i in range(3)], columns=[f"w_{i + 1}" for i in range(3)]).to_csv(
        T.TABLES_CSV / "S2_cross_scale_sensitivity.csv", float_format="%.4f")
    if VERBOSE:
        print(f"  wrote cross-scale sensitivity (rho = {r['off_diagonal_mass_pct']} %)")
    return r


# ---------------------------------------------------------------------------
# Appendix: horizons and bands
# ---------------------------------------------------------------------------
def s12_vrmse(df, show_std=None, std_rel=None):
    with _std(show_std, std_rel):
        r = T.across_seeds(T.horizon_table, df, metric="vrmse", models=C.ROWS_ALL)
        return _emit("S12_vrmse", r, T.latex_table(
            *r[:2], "VRMSE across rollout horizons for all models and PDE systems." + DET_NOTE + LF_NOTE + seeds_note(r[0], r[1]),
            "si-tab:appendix_vrmse", C.GROUPS_ALL, corner="Rollout", top_labels=C.PDE_LABELS, resize=True))


def s13_crps(df, show_std=None, std_rel=None):
    with _std(show_std, std_rel):
        r = T.across_seeds(T.horizon_table, df, metric="crps", models=C.ROWS_GEN)
        return _emit("S13_crps", r, T.latex_table(
            *r[:2], "CRPS across rollout horizons for all models and PDE systems." + LF_NOTE + seeds_note(r[0], r[1]),
            "tab:appendix_crps", C.GROUPS_GEN, corner="Rollout", top_labels=C.PDE_LABELS, resize=True))


def s14_spectral_bands(df, show_std=None, std_rel=None):
    with _std(show_std, std_rel):
        r = T.across_seeds(T.bands_table, df, models=C.ROWS_GEN, time_agg="horizon")
        return _emit("S14_spectral_bands", r, T.latex_table(
            *r[:2], r"Spectral coherence RMSE across \textit{Low}, \textit{Mid}, and \textit{High} "
            r"frequency bands for all models and PDE systems, averaged over rollout horizons." + LF_NOTE + seeds_note(r[0], r[1]),
            "tab:appendix_spectral", C.GROUPS_GEN, corner="Frequency Bands",
            top_labels=C.PDE_LABELS, resize=True))


# ---------------------------------------------------------------------------
# Physical-parameter breakdowns (Table 3 = first columns of S13 / S15)
# ---------------------------------------------------------------------------
_PARAM = {
    "S15_crps_tcool": ("turbulent_radiative_layer_2D", "tcool", "tab:ablation_trl_crps_tcool",
                       r"CRPS per $t_{\mathrm{cool}}$ on TRL, averaged over rollout steps.",
                       r"$t_{\mathrm{cool}}$", "tab:ablation_trl_crps_tcool_main", "T3a_crps_tcool"),
    "S17_crps_Ra": ("rayleigh_benard", "Rayleigh", "tab:ablation_rb_crps_ra",
                    r"CRPS per Rayleigh number $Ra$ on Rayleigh--B\'enard, averaged over Prandtl numbers.",
                    r"$Ra$", "tab:ablation_rb_crps_ra_main", "T3b_crps_Ra"),
    "S18_crps_Pr": ("rayleigh_benard", "Prandtl", "tab:ablation_rb_crps_pr",
                    r"CRPS per Prandtl number $Pr$ on Rayleigh--B\'enard, averaged over Rayleigh numbers.",
                    r"$Pr$", None, None),
}


def _crps_param(df, name, show_std=None, show_std_main=None, std_rel=None, std_rel_main=None):
    pde, param, lab, cap, corner, lab3, name3 = _PARAM[name]
    r = T.across_seeds(T.param_table, df, pde=pde, param=param, models=C.ROWS_GEN, time_agg="step")
    fmt = C.PARAM_LABELS[param]
    with _std(show_std, std_rel):
        _emit(name, r, T.latex_table(*r[:2], cap + seeds_note(r[0], r[1]), lab, C.GROUPS_GEN, corner=corner,
                                     sub_labels=fmt, resize=param == "tcool"))
    if name3:                                   # main-text Table 3 subtable: first three values
        m, s, n = r
        cols = list(m.columns)[:3]
        sub = (m[cols], s[cols], n[cols])
        with _std(show_std if show_std_main is None else show_std_main,
                  std_rel if std_rel_main is None else std_rel_main):
            _emit(name3, sub, T.latex_table(*sub[:2], cap + seeds_note(sub[0], sub[1]), lab3, C.GROUPS_GEN,
                                            corner=corner, sub_labels=fmt))
    return r


def s15_crps_tcool(df, show_std=None, show_std_main=None, std_rel=None, std_rel_main=None):
    """S13 (show_std), and Table 3a = its first three t_cool values (show_std_main,
    defaults to show_std)."""
    return _crps_param(df, "S15_crps_tcool", show_std, show_std_main, std_rel, std_rel_main)


def s17_crps_ra(df, show_std=None, show_std_main=None, std_rel=None, std_rel_main=None):
    """S15 (show_std), and Table 3b (show_std_main, defaults to show_std)."""
    return _crps_param(df, "S17_crps_Ra", show_std, show_std_main, std_rel, std_rel_main)


def s18_crps_pr(df, show_std=None, std_rel=None):
    return _crps_param(df, "S18_crps_Pr", show_std, None, std_rel)


_SPEC = {
    "S16_spectral_tcool": ("turbulent_radiative_layer_2D", "tcool", "tab:ablation_trl_spectral_tcool",
                           r"Spectral coherence RMSE per $t_{\mathrm{cool}}$ and frequency band on TRL.", r"$t_{cool}$"),
    "S19a_spectral_Ra": ("rayleigh_benard", "Rayleigh", "tab:ablation_rb_spectral_ra",
                         r"Spectral coherence RMSE per frequency band on Rayleigh--B\'enard vs $Ra$ (averaged over $Pr$).", r"$Ra$"),
    "S19b_spectral_Pr": ("rayleigh_benard", "Prandtl", "tab:ablation_rb_spectral_pr",
                         r"Spectral coherence RMSE per frequency band on Rayleigh--B\'enard vs $Pr$ (averaged over $Ra$).", r"$Pr$"),
}


def _spectral_param(df, name):
    pde, param, lab, cap, plab = _SPEC[name]
    r = T.across_seeds(T.spectral_param_table, df, pde=pde, param=param, models=C.ROWS_GEN, time_agg="step")
    return _emit(name, r, T.latex_spectral_param(*r[:2], cap + seeds_note(r[0], r[1]), lab, C.GROUPS_GEN, plab, param))


def s16_spectral_tcool(df, show_std=None, std_rel=None):
    with _std(show_std, std_rel):
        return _spectral_param(df, "S16_spectral_tcool")


def s19a_spectral_ra(df, show_std=None, std_rel=None):
    with _std(show_std, std_rel):
        return _spectral_param(df, "S19a_spectral_Ra")


def s19b_spectral_pr(df, show_std=None, std_rel=None):
    with _std(show_std, std_rel):
        return _spectral_param(df, "S19b_spectral_Pr")


# ---------------------------------------------------------------------------
# Rebuttal additions
# ---------------------------------------------------------------------------
def x_rb_latentfm(df, show_std=None, std_rel=None):
    """RB summary as shown in the rebuttal (selected wavelets + generative baselines).
    Optional: LatentFM is now also a row of every table with an RB column."""
    with _std(show_std, std_rel):
        rows = C.ROWS_BEST_GEN
        r = T.across_seeds(T.rb_summary_table, T.best_view(df, keep=C.GEN_MODELS), models=rows)
        return _emit("X_rb_latentfm", r, T.latex_table(
            *r[:2], r"Rayleigh--B\'enard: spectral coherence RMSE per band (averaged over rollout horizons) "
            r"and CRPS per rollout horizon, including the latent-space baseline LatentFM$_{\div 64}$ "
            r"(autoencoder of~\citet{rozet2025})." + seeds_note(r[0], r[1]),
            "tab:rb_latentfm", [C.BEST, C.GEN_MODELS], corner="Model", resize=True))


def s20_ke_mae(gdf, show_std=None, std_rel=None):
    """Kinetic-energy relative MAE per horizon (uses the _global_ rows)."""
    with _std(show_std, std_rel):
        r = ke_table(T.best_view(gdf, keep=C.GEN_MODELS), C.ROWS_BEST_GEN)
        return _emit("S20_ke_mae", r, T.latex_table(
            *r[:2], r"Kinetic-energy relative MAE across rollout horizons (error normalised by the "
            r"trajectory time-mean of the true kinetic energy)." + seeds_note(r[0], r[1]),
            "tab:ke_mae", C.GROUPS_BEST_GEN, corner="Rollout", top_labels=C.PDE_LABELS, resize=True))


_IC_NAME = {"crps": "S21_ic_crps", "spectral": "S22_ic_spectral"}


def s21_s22_ic_perturbation(show_std=None, std_rel_crps=None, std_rel_spectral=None):
    """TRL, WFM 3-scale haar under IC noise: CRPS per horizon and spectral per band."""
    with _std(show_std):
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
            T.save(f"{_IC_NAME[key]}", m, s, n)
            thr = std_rel_crps if key == "crps" else std_rel_spectral
            saved_rel = T.STD_REL
            if thr is not None:
                T.STD_REL = float(thr)
            tex = T.latex_table(m, s, f"TRL, WFM 3-scale (haar): {cap} under Gaussian perturbations of the "
                                r"initial conditions with standard deviation $\eta\,\sigma_c$."
                                + seeds_note(m, s), lab, [list(m.index)],
                                corner="IC noise", rank=False)
            T.STD_REL = saved_rel
            (T.TABLES_TEX / f"{_IC_NAME[key]}.tex").write_text(tex)
            if VERBOSE:
                print(f"  wrote {_IC_NAME[key]}")
            out[key] = (m, s, n)
        return out


# ---------------------------------------------------------------------------
# Audit + everything
# ---------------------------------------------------------------------------
TABLE_ORDER = ["T1_wavelet_selection", "S12_vrmse", "S13_crps", "S14_spectral_bands",
               "S15_crps_tcool", "T3a_crps_tcool", "S17_crps_Ra", "T3b_crps_Ra", "S18_crps_Pr",
               "S16_spectral_tcool", "S19a_spectral_Ra", "S19b_spectral_Pr", "X_rb_latentfm", "S20_ke_mae"]


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


def build_all(verbose=True, show_std=True, std_rel=None):
    """Build every table in paper order; returns the seed-spread summary.

    show_std: True/False for every table, or a dict {table name: bool} using the
    output names (e.g. {"T3a_crps_tcool": False, "S16_spectral_tcool": False});
    tables not in the dict use True.
    std_rel: relative std bound for mean-only captions (fraction, default 0.05)."""
    f = (lambda n: show_std.get(n, True)) if isinstance(show_std, dict) else (lambda n: bool(show_std))
    # std_rel: None (5% everywhere), one fraction for all tables, or {table: fraction}
    t = (lambda n: std_rel.get(n)) if isinstance(std_rel, dict) else (lambda n: std_rel)
    df, gdf = load_all(verbose)
    t1_wavelet_selection(df, f("T1_wavelet_selection"), f("S11_optimal_wavelets"),
                         t("T1_wavelet_selection"), t("S11_optimal_wavelets"))
    s12_vrmse(df, f("S12_vrmse"), t("S12_vrmse"))
    s13_crps(df, f("S13_crps"), t("S13_crps"))
    s14_spectral_bands(df, f("S14_spectral_bands"), t("S14_spectral_bands"))
    s15_crps_tcool(df, f("S15_crps_tcool"), f("T3a_crps_tcool"), t("S15_crps_tcool"), t("T3a_crps_tcool"))
    s17_crps_ra(df, f("S17_crps_Ra"), f("T3b_crps_Ra"), t("S17_crps_Ra"), t("T3b_crps_Ra"))
    s18_crps_pr(df, f("S18_crps_Pr"), t("S18_crps_Pr"))
    s16_spectral_tcool(df, f("S16_spectral_tcool"), t("S16_spectral_tcool"))
    s19a_spectral_ra(df, f("S19a_spectral_Ra"), t("S19a_spectral_Ra"))
    s19b_spectral_pr(df, f("S19b_spectral_Pr"), t("S19b_spectral_Pr"))
    x_rb_latentfm(df, f("X_rb_latentfm"), t("X_rb_latentfm"))
    s20_ke_mae(gdf, f("S20_ke_mae"), t("S20_ke_mae"))
    s21_s22_ic_perturbation(f("S21_S22_ic"), t("S21_ic_crps"), t("S22_ic_spectral"))
    t2_s23_s24_profiling()
    s2_cross_scale()
    return write_seed_spread()
