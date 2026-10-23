"""waveletflux — Wavelet Flow Matching for multi-scale physics emulation.

Top-level names are resolved lazily so that light-weight sub-packages
(e.g. ``waveletflux.analysis``, which only needs numpy/pandas) can be imported
without pulling in torch / lightning / the_well.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_LAZY = {
    "WaveletConfig":           ("waveletflux.utils.configs", "WaveletConfig"),
    "VelocityNetConfig":       ("waveletflux.utils.configs", "VelocityNetConfig"),
    "FlowMatchingConfig":      ("waveletflux.utils.configs", "FlowMatchingConfig"),
    "OptimizerConfig":         ("waveletflux.utils.configs", "OptimizerConfig"),
    "WaveletFlowUNet":         ("waveletflux.models.wavelet_unet", "WaveletFlowUNet"),
    "get_well_dataset":        ("waveletflux.utils.well_data", "get_well_dataset"),
    "get_well_inputs":         ("waveletflux.utils.well_data", "get_well_inputs"),
    "WellDataModuleLightning": ("waveletflux.utils.well_data", "WellDataModuleLightning"),
    "compose_augmentation":    ("waveletflux.utils.augmentations", "compose_augmentation"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod, attr = _LAZY[name]
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(f"module 'waveletflux' has no attribute {name!r}")


__all__ = list(_LAZY)
