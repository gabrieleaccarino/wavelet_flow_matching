r"""
latent_norm.py
--------------
Pixel-space normalization for the latent-flow-matching baseline.

The pre-trained LOLA autoencoders were trained on states standardized by
``field_preprocess`` (a fixed per-field mean/std taken from the run's
``config.yaml``, optionally preceded by a non-linear transform), NOT by The
Well's ``ZScoreNormalization``. To feed the encoder the exact distribution it
saw during training we reproduce ``lola.data.field_preprocess`` /
``field_postprocess`` verbatim, and provide a loader that reads the ``ae`` block
and dataset statistics straight from a LOLA autoencoder run directory.

Only pixel-space normalization lives here. LOLA applies **no** further latent
standardization — the encoder's ``softclip2`` saturation already bounds the
latent to ~[-5, 5], so the flow matching operates directly on encoder outputs.
"""

from __future__ import annotations

import os
import torch
import yaml

from torch import Tensor
from typing import Dict, Optional, Tuple


# Faithful to lola.data.TRANSFORMS
TRANSFORMS = {
    "log":     (torch.log,   torch.exp),
    "log_eps": (lambda x: torch.log(x + 1e-6), lambda x: torch.exp(x) - 1e-6),
    "log1p":   (torch.log1p, torch.expm1),
    "arcsinh": (torch.arcsinh, torch.sinh),
}


def field_preprocess(
    x: Tensor,
    mean: Optional[Tensor] = None,
    std: Optional[Tensor] = None,
    transform: Dict[int, str] = {},   # noqa: B006
    dim: int = -1,
) -> Tensor:
    r"""Standardize physical fields the way the LOLA autoencoder expects.

    Arguments:
        x: state tensor with the channel axis at ``dim`` (default last).
        mean, std: per-field statistics, shape ``(C,)``.
        transform: optional per-field non-linear transform, ``{channel: name}``.
    """
    x = x.movedim(dim, -1)
    for i, key in transform.items():
        t, _ = TRANSFORMS[key]
        x[..., int(i)] = t(x[..., int(i)])
    if mean is not None:
        x = x - mean.to(x)
    if std is not None:
        x = x / std.to(x)
    return x.movedim(-1, dim)


def field_postprocess(
    x: Tensor,
    mean: Optional[Tensor] = None,
    std: Optional[Tensor] = None,
    transform: Dict[int, str] = {},   # noqa: B006
    dim: int = -1,
) -> Tensor:
    r"""Invert :func:`field_preprocess` (latent decode → physical fields)."""
    x = x.movedim(dim, -1)
    if std is not None:
        x = x * std.to(x)
    if mean is not None:
        x = x + mean.to(x)
    for i, key in transform.items():
        _, t_inv = TRANSFORMS[key]
        x[..., int(i)] = t_inv(x[..., int(i)])
    return x.movedim(-1, dim)


def _resolve_transform(transform_cfg) -> Dict[int, str]:
    """LOLA stores ``transform`` as a possibly-empty {channel: name} mapping."""
    if not transform_cfg:
        return {}
    return {int(k): str(v) for k, v in dict(transform_cfg).items()}


def load_lola_autoencoder(
    run_dir: Optional[str] = None,
    config_yaml: Optional[str] = None,
    state_path: Optional[str] = None,
    map_location: str = "cpu",
):
    r"""Build a DC-AE and load its pre-trained weights from a LOLA run.

    A LOLA autoencoder run directory contains ``config.yaml`` (with the ``ae``
    block and ``dataset.stats``) and ``state.pth`` (``autoencoder.state_dict()``).

    Arguments:
        run_dir: path to the run directory (``config.yaml`` + ``state.pth``).
        config_yaml, state_path: explicit overrides for either file.

    Returns:
        ``(autoencoder, ae_cfg, pix_mean, pix_std, transform)`` where
        ``autoencoder`` is a frozen, eval-mode :class:`AutoEncoder`,
        ``ae_cfg`` is the raw ``ae`` config dict, ``pix_mean``/``pix_std`` are
        1-D tensors of per-field statistics and ``transform`` is the
        ``{channel: name}`` field-transform mapping.
    """
    from wavelet_flow_matching.models.dcae import get_autoencoder

    if config_yaml is None:
        assert run_dir is not None, "provide run_dir or config_yaml"
        config_yaml = os.path.join(run_dir, "config.yaml")
    if state_path is None:
        assert run_dir is not None, "provide run_dir or state_path"
        state_path = os.path.join(run_dir, "state.pth")

    with open(config_yaml, "r") as fh:
        cfg = yaml.safe_load(fh)

    ae_cfg = dict(cfg["ae"])
    stats = cfg["dataset"]["stats"]
    transform = _resolve_transform(cfg["dataset"].get("transform", {}))

    pix_mean = torch.as_tensor(stats["mean"], dtype=torch.float32)
    pix_std = torch.as_tensor(stats["std"], dtype=torch.float32)

    autoencoder = get_autoencoder(**ae_cfg)

    state = torch.load(state_path, map_location=map_location, weights_only=True)
    missing, unexpected = autoencoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"autoencoder weight mismatch loading {state_path}\n"
            f"  missing:    {list(missing)[:4]}{' ...' if len(missing) > 4 else ''}\n"
            f"  unexpected: {list(unexpected)[:4]}{' ...' if len(unexpected) > 4 else ''}"
        )

    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad_(False)

    return autoencoder, ae_cfg, pix_mean, pix_std, transform
