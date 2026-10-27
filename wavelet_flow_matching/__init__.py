"""wavelet_flow_matching — Wavelet Flow Matching for multi-scale physics emulation.

Top-level names are resolved lazily so that light-weight sub-packages
(e.g. ``wavelet_flow_matching.analysis``, which only needs numpy/pandas) can be imported
without pulling in torch / lightning / the_well.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_LAZY = {
    "WaveletConfig":           ("wavelet_flow_matching.utils.configs", "WaveletConfig"),
    "VelocityNetConfig":       ("wavelet_flow_matching.utils.configs", "VelocityNetConfig"),
    "FlowMatchingConfig":      ("wavelet_flow_matching.utils.configs", "FlowMatchingConfig"),
    "OptimizerConfig":         ("wavelet_flow_matching.utils.configs", "OptimizerConfig"),
    "WaveletFlowUNet":         ("wavelet_flow_matching.models.wavelet_unet", "WaveletFlowUNet"),
    "get_well_dataset":        ("wavelet_flow_matching.utils.well_data", "get_well_dataset"),
    "get_well_inputs":         ("wavelet_flow_matching.utils.well_data", "get_well_inputs"),
    "WellDataModuleLightning": ("wavelet_flow_matching.utils.well_data", "WellDataModuleLightning"),
    "compose_augmentation":    ("wavelet_flow_matching.utils.augmentations", "compose_augmentation"),
}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod, attr = _LAZY[name]
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(f"module 'wavelet_flow_matching' has no attribute {name!r}")


__all__ = list(_LAZY)
