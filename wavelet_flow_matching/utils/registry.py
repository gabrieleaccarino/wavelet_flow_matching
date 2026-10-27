r"""
registry.py
-----------
Single entry point used by ``inference.py`` to load either the WaveletFlow
model (also covers the data-space / Fourier variants — same module class) or the
LatentFlow baseline from one driver. Both classes expose an identical public API
(``from_checkpoint(path, map_location=...)`` and
``rollout(seed, scalar_conds, rollout_steps, ...)``), so the only thing that
varies is which class to instantiate.

inference.py already calls this as:

    from wavelet_flow_matching.utils.registry import build_module_from_checkpoint
    model = build_module_from_checkpoint(
        ckpt_path,
        map_location  = device,
        model_class   = cfg.get("model_class", "wavelet"),
        ae_state_path = cfg.get("ae_state_path", None),
    )

Set ``model_class: "latent"`` in the inference YAML for the LFM checkpoints,
``"wavelet"`` (or omit) for the wavelet/Fourier ones, or ``"auto"`` to detect
from the checkpoint (a latent checkpoint carries an ``ae_cfg`` in its
hyper-parameters; a wavelet one does not).
"""

from __future__ import annotations
from typing import Optional
import torch


_WAVELET_ALIASES = {"wavelet", "wfm", "waveletflow", "fourier", "fourierflow", "data_space", "dataspace", "fm"}
_LATENT_ALIASES  = {"latent", "lfm", "latentflow"}

def _detect_class(ckpt_path: str, map_location) -> str:
    """Peek at the checkpoint's hyper-parameters to decide the model family."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {}) or {}
    return "latent" if "ae_cfg" in hp else "wavelet"


def build_module_from_checkpoint(
    ckpt_path: str,
    map_location="cpu",
    model_class: str = "wavelet",
    ae_state_path: Optional[str] = None,
):
    key = (model_class or "auto").strip().lower()

    if key == "auto":
        key = _detect_class(ckpt_path, map_location)

    if key in _WAVELET_ALIASES:
        from wavelet_flow_matching.modules.wavelet_fm import WaveletFlowModule
        return WaveletFlowModule.from_checkpoint(ckpt_path, map_location=map_location)

    if key in _LATENT_ALIASES:
        from wavelet_flow_matching.modules.latent_fm import LatentFlowModule
        return LatentFlowModule.from_checkpoint(
            ckpt_path, map_location=map_location, ae_state_path=ae_state_path,
        )

    raise ValueError(
        f"unknown model_class={model_class!r}; use 'wavelet', 'latent', or 'auto'."
    )