"""
latent_configs.py — Configuration dataclass for the VAE latent flow matching module.

Drop this file in waveletflux/utils/ alongside the existing configs.py.
Import it in train_latent.py as:

    from waveletflux.utils.latent_configs import LatentVAEConfig
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# LatentVAEConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LatentVAEConfig:
    """
    Configuration for the frozen AutoencoderKL used to encode/decode
    physical fields into/from the VAE latent space.

    TOML section
    ─────────────
    [vae]
    model_id        = "stabilityai/sd-vae-ft-mse"
    freeze          = true
    latent_scale    = 1.0   # global scale applied BEFORE VAE input (on top of
                            # the per-sample rescaling).  Leave at 1.0 unless
                            # you want to globally shrink/expand the latent dist.
    use_mode        = true  # encode with posterior.mode(); false → sample instead

    Parameters
    ──────────
    model_id : str
        HuggingFace Hub ID for the pretrained AutoencoderKL.
        Defaults to the Stable Diffusion 2.1 VAE fine-tuned for MSE.
    freeze : bool
        If True (default), the VAE weights are frozen — only the velocity
        UNet is trained.
    latent_scale : float
        Optional multiplicative factor applied to the latent *after*
        encoding (before flow matching) and divided back before decoding.
        Useful to rescale the latent distribution to unit variance if needed.
    use_mode : bool
        Whether to use the posterior mode (deterministic) or sample from it
        during encoding.  Defaults to True (mode) for reproducibility.
    """

    model_id:     str   = "stabilityai/sd-vae-ft-mse"
    freeze:       bool  = True
    latent_scale: float = 1.0
    use_mode:     bool  = True

    # ── Constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg) -> "LatentVAEConfig":
        """
        Build from a munchified TOML section.

        Accepts both::

            [vae]
            model_id = "..."

        and the nested variant::

            [vae]
            [vae.params]
            model_id = "..."
        """
        # Support both flat [vae] and nested [vae.params]
        p = getattr(cfg, 'params', cfg)
        return cls(
            model_id     = getattr(p, 'model_id',     "stabilityai/sd-vae-ft-mse"),
            freeze       = getattr(p, 'freeze',       True),
            latent_scale = getattr(p, 'latent_scale', 1.0),
            use_mode     = getattr(p, 'use_mode',     True),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "LatentVAEConfig":
        """Reconstruct from the plain dict stored in a Lightning checkpoint."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        """Serialise for checkpoint storage."""
        return asdict(self)
