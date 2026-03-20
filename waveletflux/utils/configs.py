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
