"""
aggregate_ic_results.py
-----------------------
Collect the per-job metrics CSVs written by inference_wavelet_fm_ic_pert.py and
produce the IC-perturbation rebuttal table: VRMSE (per horizon), CRPS, and
spectral-coherence RMSE, averaged over fields/steps/trajectories within each
(ic_noise, seed), then reported as mean ± std across seeds.

Usage
    python aggregate_ic_results.py --results_dir <dir> [--glob '*_metrics.csv']
                                   [--out_prefix ic_sensitivity]

Outputs (written to results_dir)
    <out_prefix>_summary.csv   mean ± std over seeds, per ic_noise
    <out_prefix>_delta.csv     change vs ic_noise=0, per metric
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

GLOBAL_FIELD_LABEL = "_global_"

# TRL rollout horizons, matching Table S10. `step` in the CSV is the 0-indexed
# lead time (step 0 = first predicted frame = lead time 1), so lead times a:b
# correspond to step in [a-1, b-1].
HORIZONS = [("01:32", 0, 31), ("33:64", 32, 63), ("65:97", 64, 96)]
SPEC_BANDS = [("low", "Low"), ("mid", "Mid"), ("high", "High")]


def _mean_over_seeds(df, col, lo=None, hi=None):
    """mean over (field, step[in range], traj) per (ic_noise, seed),
    then mean/std across seeds -> DataFrame indexed by ic_noise."""
    sub = df
    if lo is not None:
        sub = sub[(sub["step"] >= lo) & (sub["step"] <= hi)]
    per_seed = sub.groupby(["ic_noise", "seed"])[col].mean().reset_index()
    agg = per_seed.groupby("ic_noise")[col].agg(["mean", "std"])
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--glob", default="*_metrics.csv")
    ap.add_argument("--out_prefix", default="ic_sensitivity")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.results_dir, args.glob)))
    if not files:
        raise SystemExit(f"no CSVs matched {args.glob} in {args.results_dir}")
    print(f"reading {len(files)} CSV(s):")
    for f in files:
        print("  ", os.path.basename(f))

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # per-field ML rows only (drop the global-diagnostics rows)
    if "field" in df.columns:
        df = df[df["field"] != GLOBAL_FIELD_LABEL]

    required = {"ic_noise", "seed", "step"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing required columns: {missing}")

    noise_levels = sorted(df["ic_noise"].unique())
    seeds = sorted(df["seed"].unique())
    print(f"\nic_noise levels: {noise_levels}")
    print(f"seeds          : {seeds}")
    print(f"trajectories   : {sorted(df['traj'].unique())}")

    # which metric columns are present
    have_spec = [b for b, _ in SPEC_BANDS if f"rmse_c_{b}" in df.columns]

    def cell(agg, eta):
        if eta not in agg.index:
            return "n/a"
        m = agg.loc[eta, "mean"]
        s = agg.loc[eta, "std"]
        s = 0.0 if pd.isna(s) else s
        return f"{m:.3f} ± {s:.3f}"

    rows = []
    for eta in noise_levels:
        row = {"ic_noise": eta, "n_seeds": int((df["ic_noise"] == eta)
                                               .pipe(lambda x: df.loc[x, "seed"].nunique()))}
        if "vrmse" in df.columns:
            for name, lo, hi in HORIZONS:
                row[f"VRMSE {name}"] = cell(_mean_over_seeds(df, "vrmse", lo, hi), eta)
            row["VRMSE all"] = cell(_mean_over_seeds(df, "vrmse"), eta)
        if "crps" in df.columns:
            row["CRPS all"] = cell(_mean_over_seeds(df, "crps"), eta)
        for b, lab in SPEC_BANDS:
            if b in have_spec:
                row[f"SpecCoh {lab}"] = cell(_mean_over_seeds(df, f"rmse_c_{b}"), eta)
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("ic_noise")
    out_summary = os.path.join(args.results_dir, f"{args.out_prefix}_summary.csv")
    summary.to_csv(out_summary)
    print("\n=== SUMMARY (mean ± std over seeds) ===")
    print(summary.to_string())
    print(f"\nsaved: {out_summary}")

    # delta vs baseline (mean only)
    base = noise_levels[0]
    delta_cols = {}
    if "vrmse" in df.columns:
        delta_cols["d_VRMSE_all"] = "vrmse"
    if "crps" in df.columns:
        delta_cols["d_CRPS_all"] = "crps"
    for b, lab in SPEC_BANDS:
        if b in have_spec:
            delta_cols[f"d_SpecCoh_{lab}"] = f"rmse_c_{b}"

    drows = []
    means = {name: _mean_over_seeds(df, col)["mean"] for name, col in delta_cols.items()}
    for eta in noise_levels:
        drow = {"ic_noise": eta}
        for name, series in means.items():
            drow[name] = series.get(eta, np.nan) - series.get(base, np.nan)
        drows.append(drow)
    delta = pd.DataFrame(drows).set_index("ic_noise").round(4)
    out_delta = os.path.join(args.results_dir, f"{args.out_prefix}_delta.csv")
    delta.to_csv(out_delta)
    print("\n=== DELTA vs ic_noise=0 (mean over seeds) ===")
    print(delta.to_string())
    print(f"\nsaved: {out_delta}")


if __name__ == "__main__":
    main()
