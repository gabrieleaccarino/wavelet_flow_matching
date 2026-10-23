"""Fast regression tests for waveletflux.analysis (python -m pytest tests/test_analysis.py)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from waveletflux.analysis import catalog as C, tables as T  # noqa: E402
from waveletflux.analysis.checks import paper_reproduction  # noqa: E402


def test_paper_reproduction():
    out = paper_reproduction(save=False)
    assert out["match"].all(), out[~out["match"]]


def test_seed_coverage_and_selection():
    df, _ = C.load(verbose=False)
    cov = C.coverage(df)
    seeded = cov[cov.model.map(lambda m: C.MODEL_REGISTRY[m]["source"] == "metrics")]
    assert all(s == list(C.SEEDS) for s in seeded.seeds), seeded
    sel = T.across_seeds(T.selection_table, df, models=C.ROWS_GEN, time_agg="horizon")
    assert T.check_selection(sel[0]) == []
