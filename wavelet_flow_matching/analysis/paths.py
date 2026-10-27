"""Repository-relative paths, so analysis code runs unchanged on a laptop or on
the cluster (no hard-coded /Users/... or /glade/... roots).

Override with the environment variable ``WAVELET_FLOW_MATCHING_ROOT`` if the experiment
outputs live outside the repository checkout.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get(
    "WAVELET_FLOW_MATCHING_ROOT", Path(__file__).resolve().parents[2])).resolve()
EXPERIMENTS = PROJECT_ROOT / "experiments"
ARTIFACTS = PROJECT_ROOT / "artifacts"

TABLES_TEX = ARTIFACTS / "tables" / "tex"
TABLES_CSV = ARTIFACTS / "tables" / "csv"
FIGURES = ARTIFACTS / "figures"
CHECKS = ARTIFACTS / "checks"


def metrics_dir(pde: str) -> Path:
    return EXPERIMENTS / pde / "metrics"


def ensure_artifact_dirs() -> None:
    for d in (TABLES_TEX, TABLES_CSV, FIGURES, CHECKS):
        d.mkdir(parents=True, exist_ok=True)
