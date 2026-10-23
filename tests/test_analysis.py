"""Fast regression tests for waveletflux.analysis (run: python -m pytest tests/test_analysis.py)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_paper_reproduction():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/check_paper_reproduction.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_selection_and_coverage():
    sys.path.insert(0, str(ROOT))
    from waveletflux.analysis import catalog as C, tables as T
    df, _ = C.load(verbose=False)
    cov = C.coverage(df)
    seeded = cov[cov.model.map(lambda m: C.MODEL_REGISTRY[m]["source"] == "metrics")]
    assert all(s == list(C.SEEDS) for s in seeded.seeds), seeded
    sel = T.across_seeds(T.selection_table, df, models=C.ROWS_GEN, time_agg="horizon")
    assert T.check_selection(sel[0]) == []
