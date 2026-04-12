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
    n_ensemble         : int   = 1
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
# Neural operators / diffusion baselines  (FNO / TFNO / WNO / WDNO)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NeuralOperatorConfig:
    model_type: str = "wno"
    hidden_channels: int = 64
    n_layers: int = 4
    modes: int = 16
    tucker_rank: float = 0.42
    level: int = 4
    n_scales: int = 3
    n_levels: int = 3
    n_res: int = 2
    cond_dim: int = 128
    diffusion_steps: int = 100
    sampling_steps: int = 20
    beta_schedule: str = "cosine"
    ddim_eta: float = 0.0
    ema_decay: float = 0.995
    ema_update_every: int = 1
    ema_start_step: int = 0
    wavelet: str = "db6"
    mode: str = "symmetric"
    padding: int = 0
    fc_channels: int = 128
    grid_range: Tuple[float, float] = (1.0, 1.0)

    @classmethod
    def from_config(cls, cfg) -> "NeuralOperatorConfig":
        p = cfg.params
        return cls(
            model_type=getattr(cfg, "model_type", "wno"),
            hidden_channels=int(getattr(p, "hidden_channels", getattr(p, "hidden", 64))),
            n_layers=int(getattr(p, "n_layers", getattr(p, "n_blocks", 4))),
            modes=int(getattr(p, "modes", 16)),
            tucker_rank=float(getattr(p, "tucker_rank", 0.42)),
            level=int(getattr(p, "level", 4)),
            n_scales=int(getattr(p, "n_scales", 3)),
            n_levels=int(getattr(p, "n_levels", getattr(p, "n_layers", 3))),
            n_res=int(getattr(p, "n_res", 2)),
            cond_dim=int(getattr(p, "cond_dim", 128)),
            diffusion_steps=int(getattr(p, "diffusion_steps", 100)),
            sampling_steps=int(getattr(p, "sampling_steps", 20)),
            beta_schedule=getattr(p, "beta_schedule", "cosine"),
            ddim_eta=float(getattr(p, "ddim_eta", 0.0)),
            ema_decay=float(getattr(p, "ema_decay", 0.995)),
            ema_update_every=int(getattr(p, "ema_update_every", 1)),
            ema_start_step=int(getattr(p, "ema_start_step", 0)),
            wavelet=getattr(p, "wavelet", "db6"),
            mode=getattr(p, "mode", "symmetric"),
            padding=int(getattr(p, "padding", 0)),
            fc_channels=int(getattr(p, "fc_channels", 128)),
            grid_range=tuple(getattr(p, "grid_range", (1.0, 1.0))),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "NeuralOperatorConfig":
        if "grid_range" in d:
            d = dict(d)
            d["grid_range"] = tuple(d["grid_range"])
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)


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
