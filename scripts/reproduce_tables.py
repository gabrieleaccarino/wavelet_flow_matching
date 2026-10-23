#!/usr/bin/env python
"""Rebuild every results table of the camera-ready paper (headless).

Same functions as notebooks/01_tables.ipynb (waveletflux.analysis.paper_tables),
so both write byte-identical files:
  artifacts/tables/tex/<name>.tex   LaTeX, \\label = the paper's label
  artifacts/tables/csv/<name>.csv   tidy mean / std / n_seeds per cell
  artifacts/checks/                 coverage, seed spread
Needs only numpy, pandas, pyyaml.  WAVELETFLUX_ROOT=/path overrides the repo root.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from waveletflux.analysis import paper_tables as P

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mean-only", nargs="*", metavar="TABLE",
                    help="seed mean only (no +- std). Without names: every table. With names: just "
                         "those, e.g. --mean-only T3_crps_tcool_main T3_crps_Ra_main S15_spectral_tcool")
    ap.add_argument("--std-threshold", type=float, default=None,
                    help="one bound for every mean-only caption (default: per-table "
                         "RECOMMENDED_STD_THRESHOLD in waveletflux/analysis/paper_tables.py)")
    args = ap.parse_args()
    if args.mean_only is None:
        show = True
    elif len(args.mean_only) == 0:
        show = False
    else:
        show = {name: False for name in args.mean_only}
    summ = P.build_all(show_std=show, std_threshold=args.std_threshold)
    print("\nrelative std across seeds (%):\n", summ.to_string())
