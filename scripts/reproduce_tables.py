#!/usr/bin/env python
"""Rebuild every results table of the camera-ready paper from the metric CSVs.

    python scripts/reproduce_tables.py            # all tables
    WAVELETFLUX_ROOT=/path/to/repo python ...     # if experiments/ lives elsewhere

Outputs
  artifacts/tables/tex/<name>.tex   LaTeX, \\label = the paper's label
  artifacts/tables/csv/<name>.csv   tidy mean / std / n_seeds per cell
  artifacts/checks/seed_spread.csv  relative std of every cell (all tables)
  artifacts/checks/coverage.csv     which seeds/files fed each (pde, model)
Needs only numpy + pandas.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from waveletflux.analysis import catalog as C
from waveletflux.analysis import tables as T
from waveletflux.analysis.paths import CHECKS, ensure_artifact_dirs
from waveletflux.analysis.physics import ke_table
from waveletflux.analysis.ic_sensitivity import ic_tables, load_ic
from waveletflux.analysis.profiling import load_profiles, selected

SEEDS_NOTE = r" Mean\,$\pm$\,std over three inference seeds (42, 84, 168)."
DET_NOTE = r" Deterministic baselines (WNO, FNO, TFNO) are single runs."


def main():
    ensure_artifact_dirs()
    df, gdf = C.load()
    C.coverage(pd.concat([df, gdf])).to_csv(CHECKS / "coverage.csv", index=False)
    spreads = []

    def emit(name, res, tex):
        mean, std, n = res
        T.save(name, mean, std, n, tex)
        spreads.append(T.spread_report(name, mean, std, n))
        print(f"  wrote {name}")

    # -- Table 1 / S10: wavelet selection (horizon-balanced) -----------------
    sel = T.across_seeds(T.selection_table, df, models=C.ROWS_GEN, time_agg="horizon")
    bad = T.check_selection(sel[0])
    if bad:
        raise SystemExit("wavelet selection changed: " + "; ".join(bad))
    emit("T1_wavelet_selection", sel, T.latex_selection(
        sel[0], sel[1],
        r"Spectral Coherence RMSE averaged across \textit{Low}, \textit{Mid} and \textit{High} "
        r"frequency bands and rollout horizons per PDE system. For each WFM variant, \textbf{bold} "
        r"denotes the optimal wavelet representation used in all subsequent experiments." + SEEDS_NOTE,
        "tab:avg_sc_rmse_gen_wavelets", C.GROUPS_GEN))
    s10 = T.s10_table(sel[0], sel[1])
    s10.to_csv(T.TABLES_CSV / "S10_optimal_wavelets.csv", index=False)

    # -- S11 / S12: VRMSE, CRPS per horizon ----------------------------------
    for name, metric, rows, groups, lab, cap in [
        ("S11_vrmse", "vrmse", C.ROWS_ALL, C.GROUPS_ALL, "si-tab:appendix_vrmse",
         "VRMSE across rollout horizons for all models and PDE systems." + DET_NOTE),
        ("S12_crps", "crps", C.ROWS_GEN, C.GROUPS_GEN, "tab:appendix_crps",
         "CRPS across rollout horizons for all models and PDE systems.")]:
        r = T.across_seeds(T.horizon_table, df, metric=metric, models=rows)
        emit(name, r, T.latex_table(*r[:2], cap + SEEDS_NOTE, lab, groups, corner="Rollout",
                                    top_labels=C.PDE_LABELS, resize=True))

    # -- S9 / S13: spectral per band (horizon-balanced) ----------------------
    r = T.across_seeds(T.bands_table, df, models=C.ROWS_GEN, time_agg="horizon")
    emit("S13_spectral_bands", r, T.latex_table(
        *r[:2], r"Spectral coherence RMSE across \textit{Low}, \textit{Mid}, and \textit{High} "
        r"frequency bands for all models and PDE systems, averaged over rollout horizons." + SEEDS_NOTE,
        "tab:appendix_spectral", C.GROUPS_GEN, corner="Frequency Bands",
        top_labels=C.PDE_LABELS, resize=True))

    # -- Table 3 / S14 / S16 / S17: CRPS per physical parameter (step mean) --
    for name, pde, param, lab, cap, corner in [
        ("S14_crps_tcool", "turbulent_radiative_layer_2D", "tcool", "tab:ablation_trl_crps_tcool",
         r"CRPS per $t_{\mathrm{cool}}$ on TRL, averaged over rollout steps.", r"$t_{\mathrm{cool}}$"),
        ("S16_crps_Ra", "rayleigh_benard", "Rayleigh", "tab:ablation_rb_crps_ra",
         r"CRPS per Rayleigh number $Ra$ on Rayleigh--B\'enard, averaged over Prandtl numbers.", r"$Ra$"),
        ("S17_crps_Pr", "rayleigh_benard", "Prandtl", "tab:ablation_rb_crps_pr",
         r"CRPS per Prandtl number $Pr$ on Rayleigh--B\'enard, averaged over Rayleigh numbers.", r"$Pr$")]:
        r = T.across_seeds(T.param_table, df, pde=pde, param=param, models=C.ROWS_GEN, time_agg="step")
        fmt = C.PARAM_LABELS[param]
        emit(name, r, T.latex_table(*r[:2], cap + SEEDS_NOTE, lab, C.GROUPS_GEN, corner=corner,
                                    sub_labels=fmt, resize=param == "tcool"))
        if name in ("S14_crps_tcool", "S16_crps_Ra"):     # main-text Table 3 subtables
            m, s, n = r
            cols = list(m.columns)[:3]
            sub = (m[cols], s[cols], n[cols])
            lab3 = "tab:ablation_trl_crps_tcool_main" if param == "tcool" else "tab:ablation_rb_crps_ra_main"
            emit(f"T3_{name.split('_', 1)[1]}_main", sub, T.latex_table(
                *sub[:2], cap + SEEDS_NOTE, lab3, C.GROUPS_GEN, corner=corner, sub_labels=fmt))

    # -- S15 / S18: spectral per band x parameter (step mean) ----------------
    for name, pde, param, lab, cap, plab in [
        ("S15_spectral_tcool", "turbulent_radiative_layer_2D", "tcool", "tab:ablation_trl_spectral_tcool",
         r"Spectral coherence RMSE per $t_{\mathrm{cool}}$ and frequency band on TRL.", r"$t_{cool}$"),
        ("S18a_spectral_Ra", "rayleigh_benard", "Rayleigh", "tab:ablation_rb_spectral_ra",
         r"Spectral coherence RMSE per frequency band on Rayleigh--B\'enard vs $Ra$ (averaged over $Pr$).", r"$Ra$"),
        ("S18b_spectral_Pr", "rayleigh_benard", "Prandtl", "tab:ablation_rb_spectral_pr",
         r"Spectral coherence RMSE per frequency band on Rayleigh--B\'enard vs $Pr$ (averaged over $Ra$).", r"$Pr$")]:
        r = T.across_seeds(T.spectral_param_table, df, pde=pde, param=param, models=C.ROWS_GEN, time_agg="step")
        emit(name, r, T.latex_spectral_param(*r[:2], cap + SEEDS_NOTE, lab, C.GROUPS_GEN, plab, param))

    # -- Rebuttal: RB summary with LatentFM (selected wavelets) --------------
    rb_rows = C.ROWS_BEST_GEN + [C.LATENTFM]
    r = T.across_seeds(T.rb_summary_table, T.best_view(df, keep=C.GEN_MODELS + [C.LATENTFM]), models=rb_rows)
    emit("R1_rb_latentfm", r, T.latex_table(
        *r[:2], r"Rayleigh--B\'enard: spectral coherence RMSE per band (averaged over rollout horizons) "
        r"and CRPS per rollout horizon, including the latent-space baseline LatentFM$_{\div 64}$ "
        r"(autoencoder of~\citet{rozet2025})." + SEEDS_NOTE,
        "tab:rb_latentfm", [C.BEST, C.GEN_MODELS + [C.LATENTFM]], corner="Model", resize=True))

    # -- Rebuttal: kinetic-energy MAE per horizon (selected wavelets) --------
    gb = T.best_view(gdf, keep=C.GEN_MODELS)
    r = ke_table(gb, C.ROWS_BEST_GEN)
    emit("R2_ke_mae", r, T.latex_table(
        *r[:2], r"Kinetic-energy relative MAE across rollout horizons (error normalised by the "
        r"trajectory time-mean of the true kinetic energy)." + SEEDS_NOTE,
        "tab:ke_mae", C.GROUPS_BEST_GEN, corner="Rollout", top_labels=C.PDE_LABELS, resize=True))

    # -- Rebuttal: IC perturbation (TRL, WFM 3-scale haar) -------------------
    ic = ic_tables(load_ic())
    for key, lab, cap in [("crps", "tab:ic_pert_crps", "CRPS across rollout horizons"),
                          ("spectral", "tab:ic_pert_spectral", "Spectral coherence RMSE across bands")]:
        m, s = ic[key]
        n = pd.DataFrame(np.broadcast_to(ic["n_seeds"].to_numpy()[:, None], m.shape).astype(float),
                         index=m.index, columns=m.columns)
        m.index = [f"$\\eta={e:g}$" for e in m.index]; s.index = m.index; n.index = m.index
        T.save(f"R3_ic_{key}", m, s, n)
        tex = T.latex_table(m, s, f"TRL, WFM 3-scale (haar): {cap} under Gaussian perturbations of the "
                            r"initial conditions with standard deviation $\eta\,\sigma_c$." + 
                            r" Mean\,$\pm$\,std over three seeds.", lab, [list(m.index)],
                            corner="IC noise", rank=False)
        (T.TABLES_TEX / f"R3_ic_{key}.tex").write_text(tex)
        print(f"  wrote R3_ic_{key}")

    # -- Tables 2 / S19 / S20: inference profiling --------------------------
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
    print("  wrote T2/S19/S20 profiling")

    sp = pd.concat(spreads, ignore_index=True)
    sp.to_csv(CHECKS / "seed_spread.csv", index=False, float_format="%.6g")
    summ = sp.groupby("table")["rel_std_pct"].agg(
        cells="size", median="median", max="max",
        pct_over_0p5=lambda x: 100 * (x > 0.5).mean(), pct_over_1=lambda x: 100 * (x > 1).mean())
    summ.round(2).to_csv(CHECKS / "seed_spread_summary.csv")
    print("\nrelative std across seeds (%):\n", summ.round(2).to_string())


if __name__ == "__main__":
    main()
