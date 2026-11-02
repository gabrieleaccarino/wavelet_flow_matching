"""
configs.py

Typed configuration dataclasses for WaveletFlow.

Each dataclass maps to one logical section of the TOML config.
The `from_config` classmethod on each handles the TOML → Python
translation in one place, so nothing else needs to unpack dicts
or call getattr with defaults.

To plug in a new architecture: define a new dataclass (e.g. TransformerNetConfig)
with its own `from_config`, pass it to WaveletFlowModule. Existing configs are
unaffected.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Wavelet
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WaveletConfig:
    use_wavelet : bool
    wavelet     : str   = 'haar'
    mode        : str   = 'periodic'
    n_scales    : int   = 3
    n_coeff_mat : int   = 4      # subbands per scale (4 for standard 2-D DWT)

    @classmethod
    def from_config(cls, cfg) -> WaveletConfig:
        return cls(
            use_wavelet = cfg.use_wavelet,
            n_coeff_mat = int(cfg.n_coeff_mat),
            wavelet     = cfg.params.wavelet,
            mode        = cfg.params.mode,
            n_scales    = int(cfg.params.n_scales),
        )
    
    @classmethod
    def from_dict(cls, d: dict) -> "WaveletConfig":
        return cls(**d)
    
    def to_dict(self) -> dict:
        return asdict(self)   # works if all fields are primitives / lists

    def decomposer_kwargs(self) -> dict:
        """Kwargs forwarded directly to WaveletDecomposer2D."""
        return dict(wavelet=self.wavelet, mode=self.mode, n_scales=self.n_scales)


# ─────────────────────────────────────────────────────────────────────────────
# Velocity network  (architecture + conditioning)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VelocityNetConfig:
    # Architecture knobs
    base_ch  : int = 32
    n_res    : int = 1
    n_levels : int = 3
    cond_dim : int = 128
    # Conditioning / context
    num_cond            : int = 0
    context_in_channels : int = 4
    num_context_frames  : int = 0

    @classmethod
    def from_config(cls, cfg) -> VelocityNetConfig:
        p = cfg.params
        return cls(
            base_ch             = int(getattr(p, 'base_ch',             32)),
            n_res               = int(getattr(p, 'n_res',               1)),
            n_levels            = int(getattr(p, 'n_levels',            3)),  # must >= wavelet.params.n_scales
            cond_dim            = int(getattr(p, 'cond_dim',            128)),
            num_cond            = int(getattr(p, 'num_cond',            0)),
            context_in_channels = int(getattr(p, 'context_in_channels', 4)),
            num_context_frames  = int(getattr(p, 'num_context_frames',  0)),
        )
    
    @classmethod
    def from_dict(cls, d: dict) -> "VelocityNetConfig":
        return cls(**d)
    
    def to_dict(self) -> dict:
        return asdict(self)   # works if all fields are primitives / lists


# ─────────────────────────────────────────────────────────────────────────────
# Flow matching  (interpolant + ensemble)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FlowMatchingConfig:
    n_ensemble         : int   = 1    # M: ensemble members (independent realizations) per initial condition
    num_sampling_steps : int   = 20
    sigma              : float = 0.0

    @classmethod
    def from_config(cls, fm_cfg, sampling_cfg) -> FlowMatchingConfig:
        return cls(
            n_ensemble         = int(fm_cfg.n_ensemble),
            sigma              = float(getattr(fm_cfg.params, 'sigma', 0.0)),
            num_sampling_steps = int(sampling_cfg.num_sampling_steps),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "FlowMatchingConfig":
        return cls(**d)
    
    def to_dict(self) -> dict:
        return asdict(self)   # works if all fields are primitives / lists


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer + scheduler  (grouped because the scheduler depends on warmup)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OptimizerConfig:
    cls           : str         = 'AdamW'
    lr            : float       = 3e-4
    weight_decay  : float       = 1e-2
    betas         : Tuple       = (0.9, 0.999)
    warmup_epochs : int         = 50
    scheduler_cls : str         = 'CosineAnnealingLR'
    eta_min       : float       = 1e-7

    @classmethod
    def from_config(cls, opt_cfg, sched_cfg, train_cfg) -> OptimizerConfig:
        op = dict(opt_cfg.params)
        sp = dict(sched_cfg.params)
        return cls(
            cls           = opt_cfg.cls,
            lr            = float(op.get('lr',           3e-4)),
            weight_decay  = float(op.get('weight_decay', 1e-2)),
            betas         = tuple(op.get('betas',        (0.9, 0.999))),
            warmup_epochs = int(getattr(train_cfg, 'warmup_epochs', 50)),
            scheduler_cls = sched_cfg.cls,
            eta_min       = float(sp.get('eta_min',      1e-7)),
        )
    
    @classmethod
    def from_dict(cls, d: dict) -> "OptimizerConfig":
        return cls(**d)
    
    def to_dict(self) -> dict:
        return asdict(self)   # works if all fields are primitives / lists
    
    def optimizer_kwargs(self) -> dict:
        """Kwargs forwarded to the PyTorch optimizer constructor."""
        return dict(lr=self.lr, weight_decay=self.weight_decay, betas=self.betas)


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