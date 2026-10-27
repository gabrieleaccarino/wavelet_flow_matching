#!/usr/bin/env python
"""Rebuild the submitted paper's tables from the single-run files and compare
cell by cell; see wavelet_flow_matching/analysis/checks.py. Exit code 1 on any mismatch."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wavelet_flow_matching.analysis.checks import paper_reproduction, summary


def main():
    out = paper_reproduction()
    print(summary(out).to_string())
    bad = out[~out["match"]]
    if len(bad):
        print(f"\n{len(bad)} mismatching cells:\n" + bad.to_string(index=False))
        sys.exit(1)
    print(f"\nall {len(out)} cells reproduced exactly.")


if __name__ == "__main__":
    main()
