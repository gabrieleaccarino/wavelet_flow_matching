"""Physics-consistency diagnostics from the `_global_` metric rows.

Rebuttal (reviewer jBaL, W2): kinetic-energy error per rollout horizon.

The stored ``{base}_mae`` columns are relative to the *instantaneous* truth,
|Q_pred - Q_true| / |Q_true(t)|, which blows up during Rayleigh–Bénard spin-up
(E_k^true ~ 1e-10). We therefore renormalise by the per-trajectory time-mean
of |Q_true|:

    {base}_mae_tm = {base}_mae * |Q_true(t)| / mean_t |Q_true|

i.e. the absolute error divided by a strictly positive, truth-only constant
that is identical for every model and seed. This is the ``ke_mae_tm``
quantity reported in the rebuttal table ("Kinetic energy MAE").
"""
from __future__ import annotations

import pandas as pd

from . import catalog as C
from .tables import across_seeds

RENORM_BASES = ["ke", "enstrophy", "massflux", "nusselt", "nematic", "gradsq"]


def add_tm_metrics(g: pd.DataFrame) -> pd.DataFrame:
    """Add ``{base}_mae_tm`` / ``{base}_bias_tm`` to `_global_` rows."""
    g = g.copy()
    key = ["pde", "traj"]
    for base in RENORM_BASES:
        t = f"{base}_true"
        if t not in g.columns or g[t].notna().sum() == 0:
            continue
        truth = g.dropna(subset=[t]).groupby(key + ["step"], as_index=False)[t].mean()
        den = (truth.assign(_a=truth[t].abs()).groupby(key, as_index=False)["_a"].mean()
                    .rename(columns={"_a": "_den"}))
        g = g.merge(den, on=key, how="left")
        for stat in ("mae", "bias"):
            c = f"{base}_{stat}"
            if c in g.columns:
                g[f"{c}_tm"] = g[c] * g[t].abs() / g["_den"]
        g = g.drop(columns="_den")
    return g


def physics_horizon_table(g, metric, models):
    """One seed's rows -> Model x (pde, horizon) mean of `metric`."""
    parts = []
    for pde in C.PDES:
        d = g[(g.pde == pde) & g.model.isin(models)].dropna(subset=[metric])
        if d.empty:
            continue
        t = d.groupby(["model", "horizon"])[metric].mean().unstack().reindex(columns=C.HORIZONS[pde])
        t.columns = pd.MultiIndex.from_tuples([(pde, h) for h in t.columns])
        parts.append(t)
    if not parts:
        return None
    out = pd.concat(parts, axis=1)
    return out.reindex([m for m in models if m in out.index])


def ke_table(global_df, models, metric="ke_mae_tm"):
    """Kinetic-energy relative MAE per horizon, mean/std/n across seeds."""
    g = add_tm_metrics(global_df)
    return across_seeds(lambda df, models: physics_horizon_table(df, metric, models), g, models=models)
