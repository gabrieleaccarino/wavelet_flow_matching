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
    summ = P.build_all()
    print("\nrelative std across seeds (%):\n", summ.to_string())
