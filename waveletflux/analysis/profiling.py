"""Inference profiling (Tables 2, S18, S19) from ``experiments/<pde>/performance/<MODEL>_<pde>_perf.yaml``.

Each YAML is written by the inference driver: wall_clock_s (max over ranks),
total_steps_global, throughput_*, gpu_peak_mem_GiB (per rank). All models use
NFE = 50 Euler steps, 8 samples, 4 x A100.

  speedup   = wall_clock(FM_pixel) / wall_clock(WFM)          (> 1: WFM faster)
  mem_ratio = gpu_mem(FM_pixel)    / gpu_mem(WFM)             (> 1: WFM lighter)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from . import catalog as C
from .paths import EXPERIMENTS

BASE = "FM_pixel"


def _mem(block):
    if isinstance(block, dict):
        v = [float(x) for x in block.values() if x is not None]
        return float(np.mean(v)) if v else np.nan
    return float(block) if block is not None else np.nan


def load_profiles(pdes=C.PDES) -> pd.DataFrame:
    recs = []
    for pde in pdes:
        for p in sorted((EXPERIMENTS / pde / "performance").glob(f"*_{pde}_perf.yaml")):
            model = p.name[: -len(f"_{pde}_perf.yaml")]
            raw = yaml.safe_load(p.read_text())
            wc, ts = raw.get("wall_clock_s"), raw.get("total_steps_global")
            recs.append(dict(pde=pde, model=model, wall_clock_s=wc,
                             sec_per_step=wc / ts if wc and ts else np.nan,
                             gpu_mem_GiB=_mem(raw.get("gpu_peak_mem_GiB")),
                             throughput_Mfr_s=(raw.get("throughput_frames_s") or np.nan) / 1e6))
    df = pd.DataFrame(recs)
    base = df[df.model == BASE].set_index("pde")
    df["speedup"] = df.pde.map(base.wall_clock_s) / df.wall_clock_s
    df["mem_ratio"] = df.pde.map(base.gpu_mem_GiB) / df.gpu_mem_GiB
    return df


def selected(df: pd.DataFrame) -> pd.DataFrame:
    """FM_pixel + the selected wavelet per PDE and scale (Table 2 / S18)."""
    keep = []
    for pde, sel in C.BEST_WFM_PER_PDE.items():
        keep += [(pde, BASE), (pde, C.wfm_key("1scale", sel["1scale"])), (pde, C.wfm_key("3scale", sel["3scale"]))]
    return df[[(p, m) in keep for p, m in zip(df.pde, df.model)]]
