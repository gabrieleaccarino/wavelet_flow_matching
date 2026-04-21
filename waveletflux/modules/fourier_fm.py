"""
fourier_fm.py
-------------
Multi-scale Fourier flow matching module.

This module is a drop-in replacement for WaveletFlowModule.  The wavelet
decomposition is replaced by a 2-D real FFT; flow matching is performed
directly on the (truncated) Fourier coefficients.  The velocity network is
the same WaveletFlowUNet — its interface is agnostic to whether the
multi-scale features come from wavelets or Fourier modes.

Fourier pyramid
---------------
For n_scales > 1, a nested frequency pyramid is built from the truncated
rfft2 output:

    scale 0  (finest)  :  rows [0 .. n_modes_h),        n_coeff = 2
    scale 1            :  rows [0 .. n_modes_h // 2),    n_coeff = 2
    scale i            :  rows [0 .. max(n_modes_h >> i, 1))
    scale n_scales - 1 :  rows [0 .. max(n_modes_h >> (n-1), 1))

Scale 0 is used for reconstruction (irfft2 with spectral zero-padding).
Coarser scales act as auxiliary training targets that encourage the network
to learn a multi-resolution Fourier representation; they are NOT used during
inference reconstruction, matching how wavelet coarse scales are naturally
handled.

Coefficient layout
------------------
Each scale always has n_coeff = 2 channels per field / time-step:

    channel 0  →  real part  of the truncated Fourier coefficient
    channel 1  →  imaginary part

This is compatible with WaveletFlowUNet without any code change since the
network treats n_coeff as an opaque integer.

subband_weights
---------------
Because n_coeff = 2 for every scale, subband_weights should have exactly
two entries [w_real, w_imag].  Providing None weights real and imaginary
parts equally (standard behaviour).  A value of [1.0, 1.0] is equivalent
to None.

Physical interpretation
-----------------------
OT-CFM in Fourier space is well-defined: the interpolant

    x(τ) = τ · F(x₁) + (1 - τ) · ε,   ε ~ N(0, I)

mixes Gaussian noise with the target Fourier coefficients.  Each scale is
trained independently with a normalised per-channel MSE loss (_subband_vmse),
which prevents the large DC component from dominating the gradient.

n_modes_h selection guide
--------------------------
    grid     |  full spectrum  |  recommended starting point
    ---------+-----------------+-----------------------------
    64 × 64  |  64 × 33        |  n_modes_h = 16–32
    128×128  | 128 × 65        |  n_modes_h = 32–48
    256×256  | 256 × 129       |  n_modes_h = 48–64
    512×256  | 512 × 129       |  n_modes_h = 64–128

Compatibility
-------------
*  FourierFlowModule has the same __init__ signature as WaveletFlowModule
   (FourierConfig replaces WaveletConfig; all other arguments are identical).
*  from_checkpoint works with checkpoints produced by this module.
*  inference.py needs no changes.
*  train.py requires only minimal edits — see the docstring at the bottom of
   this file for the required modifications and a complete sample TOML config.
"""

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch.nn.functional as F
import pytorch_lightning as pl
from einops import rearrange
from torch.optim import *
import torch.nn as nn
import torch
import math

from waveletflux.models.wavelet_unet import WaveletFlowUNet
from waveletflux.utils.configs import (
    VelocityNetConfig,
    FlowMatchingConfig,
    OptimizerConfig,
)
# Re-use the same variance-normalised subband MSE from the wavelet module.
# No circular dependency: wavelet_fm does not import fourier_fm.
from waveletflux.modules.wavelet_fm import _subband_vmse


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FourierConfig:
    """
    Configuration for the 2-D real-FFT decomposer.

    Parameters
    ----------
    n_modes_h  : Number of Fourier row modes to keep (≤ H).
                 The rfft2 output has H rows (full row count); we keep
                 the first n_modes_h, which correspond to the lowest
                 spatial frequencies along the height dimension.
    n_modes_w  : Number of Fourier column modes to keep (≤ W // 2 + 1).
                 The rfft2 output has W // 2 + 1 complex columns.
    n_scales   : Depth of the frequency pyramid.
                 1  →  single truncated spectrum (recommended default).
                 >1 →  nested pyramid: scale i has max(n_modes_h >> i, 1) rows.
                 Must satisfy n_scales ≤ velocity_net.n_levels.
    norm       : FFT normalisation mode forwarded to torch.fft.rfft2 /
                 torch.fft.irfft2.  'ortho' (default) preserves Parseval's
                 theorem and keeps coefficient magnitudes comparable to the
                 spatial-domain signal.
    """
    n_modes_h : int = 16
    n_modes_w : int = 16
    n_scales  : int = 1
    norm      : str = 'ortho'

    # ── constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg) -> "FourierConfig":
        """Build from a munchified TOML section (flat or with [fourier.params])."""
        p = getattr(cfg, 'params', cfg)
        return cls(
            n_modes_h = int(getattr(p, 'n_modes_h', 16)),
            n_modes_w = int(getattr(p, 'n_modes_w', 16)),
            n_scales  = int(getattr(p, 'n_scales',  1)),
            norm      = str(getattr(p, 'norm',      'ortho')),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "FourierConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)

    def decomposer_kwargs(self, data_shape: Tuple[int, int]) -> dict:
        """Keyword arguments forwarded directly to FourierDecomposer2D."""
        return dict(
            n_modes_h  = self.n_modes_h,
            n_modes_w  = self.n_modes_w,
            n_scales   = self.n_scales,
            norm       = self.norm,
            data_shape = data_shape,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Fourier decomposer
# ─────────────────────────────────────────────────────────────────────────────

class FourierDecomposer2D(nn.Module):
    """
    2-D real-FFT decomposer with an optional nested frequency pyramid.

    The interface mirrors WaveletDecomposer2D so that FourierFlowModule can
    reuse WaveletFlowUNet without any modification.

    forward_transform(x)
    --------------------
    Input  : (B, C, T, H, W)  — any device, real dtype
    Output : { scale_i : (B, C, T, 2, H_i, W_modes) }
             where H_i    = min(n_modes_h >> i, H)   [clamped to ≥ 1]
                   W_modes = min(n_modes_w, W // 2 + 1)
                   dim 3  = 2  (index 0 → real, index 1 → imaginary)

    inverse_transform(coeffs)
    -------------------------
    Input  : { scale_i : (B, C, T, 2, H_i, W_modes) }
    Output : (B, C, T, H, W)  via spectral zero-padding + irfft2

    Reconstruction uses scale 0 only (finest, most Fourier modes).
    High-frequency rows beyond n_modes_h and columns beyond n_modes_w are
    treated as zero (standard band-limited reconstruction as in FNO).
    """

    def __init__(
        self,
        n_modes_h  : int,
        n_modes_w  : int,
        n_scales   : int = 1,
        norm       : str = 'ortho',
        data_shape : Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        assert n_modes_h >= 1, "n_modes_h must be ≥ 1"
        assert n_modes_w >= 1, "n_modes_w must be ≥ 1"
        assert n_scales  >= 1, "n_scales must be ≥ 1"

        self.n_modes_h = n_modes_h
        self.n_modes_w = n_modes_w
        self.n_scales  = n_scales
        self.norm      = norm

        # Store original spatial dimensions for irfft2 reconstruction.
        # Registered as buffers so they travel with the model on device moves
        # and appear in state_dict (checkpointing).
        if data_shape is not None:
            H, W = data_shape
            self.register_buffer('spatial_H', torch.tensor(H, dtype=torch.long))
            self.register_buffer('spatial_W', torch.tensor(W, dtype=torch.long))
        else:
            # Fallback: infer target shape from the finest-scale coefficients at
            # inverse_transform time (gives an approximate reconstruction size).
            self.spatial_H = None
            self.spatial_W = None

    # ── mode-count helpers ────────────────────────────────────────────────────

    def _n_row_modes(self, scale: int, H: int) -> int:
        """Row-mode count at the given pyramid scale, clamped to [1, H]."""
        return max(min(self.n_modes_h >> scale, H), 1)

    def _n_col_modes(self, W: int) -> int:
        """Column-mode count, clamped to the rfft2 output width W // 2 + 1."""
        return min(self.n_modes_w, W // 2 + 1)

    # ── forward (analysis) ────────────────────────────────────────────────────

    def forward_transform(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        x : (B, C, T, H, W)
        Returns { i : (B, C, T, 2, H_i, W_modes) }  for i in 0..n_scales-1
        """
        B, C, T, H, W = x.shape
        n_w = self._n_col_modes(W)

        # rfft2 over the last two (spatial) dims.
        # Output shape: (B, C, T, H, W//2+1) complex64/complex128.
        F_full = torch.fft.rfft2(x, norm=self.norm)

        coeffs: Dict[int, torch.Tensor] = {}
        for i in range(self.n_scales):
            n_h = self._n_row_modes(i, H)
            # Slice the top-left corner of the frequency matrix:
            #   rows 0..n_h  — lowest n_h spatial frequencies along H
            #   cols 0..n_w  — lowest n_w spatial frequencies along W (rfft2 half)
            Fi = F_full[..., :n_h, :n_w]                          # complex (B,C,T,n_h,n_w)
            # Stack real and imaginary as the "coefficient" dimension (dim 3).
            # Shape: (B, C, T, 2, n_h, n_w)  — matches the WaveletDecomposer2D layout
            #   where dim 3 carries the subband index.
            coeffs[i] = torch.stack([Fi.real, Fi.imag], dim=3)

        return coeffs

    # ── inverse (synthesis) ───────────────────────────────────────────────────

    def inverse_transform(self, coeffs: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        coeffs : { i : (B, C, T, 2, H_i, W_modes) }
        Returns  (B, C, T, H, W)

        Only scale 0 is used; coarser scales are ignored here (they serve
        only as auxiliary training targets).  Spectral zero-padding restores
        the full frequency grid before irfft2.
        """
        v    = coeffs[0]                            # (B, C, T, 2, n_h, n_w)
        real = v[:, :, :, 0, :, :]                 # (B, C, T, n_h, n_w)
        imag = v[:, :, :, 1, :, :]
        F0   = torch.complex(real, imag)            # complex (B, C, T, n_h, n_w)

        B, C, T, n_h, n_w = F0.shape

        # Determine the output spatial dimensions.
        H     = int(self.spatial_H) if self.spatial_H is not None else n_h
        W     = int(self.spatial_W) if self.spatial_W is not None else 2 * (n_w - 1)
        W_rfft = W // 2 + 1

        # Spectral zero-padding: embed truncated modes in the top-left corner
        # of a full-size complex frequency tensor.  Rows n_h..H and columns
        # n_w..W_rfft are left as zero (band-limited reconstruction).
        F_padded = torch.zeros(
            B, C, T, H, W_rfft, dtype=F0.dtype, device=F0.device
        )
        F_padded[..., :n_h, :n_w] = F0

        # irfft2 with explicit output shape to handle odd-W grids correctly.
        return torch.fft.irfft2(F_padded, s=(H, W), norm=self.norm)


# ─────────────────────────────────────────────────────────────────────────────
# Lightning module
# ─────────────────────────────────────────────────────────────────────────────

class FourierFlowModule(pl.LightningModule):
    """
    Multi-scale Fourier flow matching module.

    Analogous to WaveletFlowModule: the wavelet decomposition is replaced
    by a 2-D real FFT and the flow matching is performed in Fourier
    coefficient space.  The velocity network is the same WaveletFlowUNet —
    it operates on the multi-scale coefficient dict regardless of whether
    that dict was produced by a wavelet or a Fourier decomposition.

    Parameters
    ----------
    in_channels       : number of physical fields C
    field_names       : field names, used for logging and plotting
    data_shape        : (H, W) of the spatial domain
    n_steps_output    : number of predicted time steps T
    scale_weights     : per-pyramid-scale loss weights; length == n_scales
    subband_weights   : real/imaginary loss weights; must have exactly 2
                        entries [w_real, w_imag], or None for equal weighting
    fourier_cfg       : FourierConfig
    velocity_net_cfg  : VelocityNetConfig — n_levels must satisfy
                        n_levels >= fourier_cfg.n_scales
    flow_matching_cfg : FlowMatchingConfig
    optimizer_cfg     : OptimizerConfig
    normalizer        : normalizer instance (e.g. ZScoreNormalization from
                        the-well)

    Decoupled-depth notes
    ---------------------
    n_scales (from fourier_cfg)       — pyramid depth; controls how many
                                        frequency scales the UNet ingests.
    n_levels (from velocity_net_cfg)  — UNet depth (down/upsamplings).
                                        Must satisfy n_levels >= n_scales so
                                        that every pyramid level has a
                                        corresponding UNet encoder level.
    Use the same n_levels for both Fourier and data-space baselines to keep
    model capacity consistent across experiments.

    T-dimension note
    ----------------
    x0 (the current frame, T=1) is tiled to (B, C, T, H, W) via
    _to_coeff_dict_x0 before being transformed, so that curr[i] and x_tau[i]
    always share the same channel count C·T·2 at every pyramid scale.
    """

    def __init__(
        self,
        in_channels      : int,
        field_names      : List[str],
        data_shape       : Tuple[int, int],
        n_steps_output   : int,
        scale_weights    : List[float],
        subband_weights  : Optional[List[float]],
        fourier_cfg      : FourierConfig,
        velocity_net_cfg : VelocityNetConfig,
        flow_matching_cfg: FlowMatchingConfig,
        optimizer_cfg    : OptimizerConfig,
        normalizer       : Any,
    ):
        super().__init__()

        self.in_channels       = in_channels
        self.field_names       = field_names
        self.data_shape        = data_shape
        self.n_steps_output    = n_steps_output
        self.normalizer        = normalizer
        self.optimizer_cfg     = optimizer_cfg
        self.flow_matching_cfg = flow_matching_cfg
        self.n_scales          = fourier_cfg.n_scales

        # ── Fourier decomposer ────────────────────────────────────────────────
        self.fourier_decomposer = FourierDecomposer2D(
            **fourier_cfg.decomposer_kwargs(data_shape)
        )

        # Shape probe must run after n_steps_output and n_scales are set.
        self.scale_shapes = self._probe_fourier_shapes()

        assert len(scale_weights) == self.n_scales, (
            f"scale_weights has {len(scale_weights)} entries "
            f"but fourier_cfg.n_scales = {self.n_scales}"
        )
        self.register_buffer(
            'scale_weights',
            torch.tensor(scale_weights, dtype=torch.float32),
        )

        # For Fourier, n_coeff = 2 (real + imaginary) at every scale.
        # subband_weights must therefore have exactly 2 entries if provided.
        if subband_weights is not None:
            if len(subband_weights) != 2:
                raise ValueError(
                    f"subband_weights for FourierFlowModule must have exactly 2 "
                    f"entries [w_real, w_imag] (got {len(subband_weights)}). "
                    f"Use None for equal weighting."
                )
            self.register_buffer(
                'subband_weights',
                torch.tensor(subband_weights, dtype=torch.float32),
            )
        else:
            self.subband_weights = None

        # ── Velocity network ──────────────────────────────────────────────────
        # WaveletFlowUNet is used unchanged — it only cares about the shapes in
        # scale_shapes and the integer n_coeff, which equals 2 for Fourier.
        vn = velocity_net_cfg
        self.net = WaveletFlowUNet(
            scale_shapes     = self.scale_shapes,
            in_channels      = self.in_channels,
            n_steps_output   = self.n_steps_output,
            n_scales         = self.n_scales,
            n_levels         = vn.n_levels,
            context_channels = vn.context_in_channels,
            n_context_frames = vn.num_context_frames,
            cond_dim         = vn.cond_dim,
            num_scalar_conds = vn.num_cond,
            base_ch          = vn.base_ch,
            n_res            = vn.n_res,
        )

        self.experiment_manager = None

        self.save_hyperparameters({
            "in_channels"      : in_channels,
            "field_names"      : field_names,
            "data_shape"       : data_shape,
            "n_steps_output"   : n_steps_output,
            "scale_weights"    : scale_weights,
            "subband_weights"  : subband_weights,
            "fourier_cfg"      : fourier_cfg.to_dict(),
            "velocity_net_cfg" : velocity_net_cfg.to_dict(),
            "flow_matching_cfg": flow_matching_cfg.to_dict(),
            "optimizer_cfg"    : optimizer_cfg.to_dict(),
            "normalizer_stats" : {
                "mean"      : {k: v.tolist() for k, v in normalizer.means.items()},
                "std"       : {k: v.tolist() for k, v in normalizer.stds.items()},
                "mean_delta": {k: v.tolist() for k, v in normalizer.means_delta.items()},
                "std_delta" : {k: v.tolist() for k, v in normalizer.stds_delta.items()},
            },
            "normalizer_core_field_names"         : normalizer.core_field_names,
            "normalizer_core_constant_field_names": normalizer.core_constant_field_names,
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Shape probe
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _probe_fourier_shapes(self) -> Dict[int, dict]:
        """
        Run a dummy tensor through the Fourier decomposer to determine the
        spatial shape and coefficient count at each pyramid scale.

        Identical pattern to WaveletFlowModule._probe_wavelet_shapes — the
        probe reads shape[3] as n_coeff, shape[-2] as H, shape[-1] as W.
        For Fourier: n_coeff = 2 at every scale (real, imaginary).
        """
        H, W  = self.data_shape
        dummy = torch.zeros(1, self.in_channels, self.n_steps_output, H, W)
        coeffs = self.fourier_decomposer.forward_transform(dummy)
        return {
            i: {
                'H'      : c.shape[-2],   # frequency-domain row count at scale i
                'W'      : c.shape[-1],   # frequency-domain col count (shared)
                'n_coeff': c.shape[3],    # always 2: real, imaginary
            }
            for i, c in coeffs.items()
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Fourier / coefficient helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _to_coeff_dict(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        (B, C, T, H, W) → { scale : (B, C·T·2, H_scale, W_modes) }
        """
        raw = self.fourier_decomposer.forward_transform(x)
        return {
            i: rearrange(v, 'B C T Coeff H W -> B (C T Coeff) H W')
            for i, v in raw.items()
        }

    def _to_coeff_dict_x0(self, x0: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Convert the current frame x0 (B, C, 1, H, W) to a Fourier coefficient
        dict tiled to T = n_steps_output along the temporal axis.

        Tiling ensures curr[i] and x_tau[i] always share the same channel
        count C·T·2, which is required for the UNet stem concatenation.
        """
        x0_tiled = x0.expand(-1, -1, self.n_steps_output, -1, -1)
        return self._to_coeff_dict(x0_tiled)

    def _from_coeff_dict(self, coeff: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        { scale : (B, C·T·2, H_scale, W_modes) } → (B, T, H, W, C)  de-normalised.
        Reconstruction uses scale 0 only via irfft2.
        """
        raw = {
            i: rearrange(
                v,
                'B (C T Coeff) H W -> B C T Coeff H W',
                C    = self.in_channels,
                T    = self.n_steps_output,
                Coeff= self.scale_shapes[i]['n_coeff'],  # 2
            )
            for i, v in coeff.items()
        }
        spatial = self.fourier_decomposer.inverse_transform(raw)  # (B, C, T, H, W)
        return self.normalizer.denormalize_flattened(
            rearrange(spatial, 'B C T H W -> B T H W C'), mode='variable'
        )

    def _to_spatial_normalized(self, coeff: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        { scale : (B, C·T·2, H_scale, W_modes) } → (B, T, H, W, C)  still NORMALISED.
        Used during rollout to update the conditioning window without discarding
        the z-score normalisation.
        """
        raw = {
            i: rearrange(
                v,
                'B (C T Coeff) H W -> B C T Coeff H W',
                C    = self.in_channels,
                T    = self.n_steps_output,
                Coeff= self.scale_shapes[i]['n_coeff'],
            )
            for i, v in coeff.items()
        }
        spatial = self.fourier_decomposer.inverse_transform(raw)
        return rearrange(spatial, 'B C T H W -> B T H W C')

    # ─────────────────────────────────────────────────────────────────────────
    # Batch preparation  (identical to WaveletFlowModule)
    # ─────────────────────────────────────────────────────────────────────────

    def _prepare_batch(self, batch):
        """
        Normalise inputs, split into current frame x0, targets x1,
        past context xp, and scalar conditions.

        Returns (x0, x1, cond, B)
        x0 : (B, C, 1, H, W) — most recent input frame
        x1 : (B, C, T, H, W) — target output frames
        """
        device = self.device
        inp = self.normalizer.normalize_flattened(
            batch["input_fields"].to(device), mode='variable')
        out = self.normalizer.normalize_flattened(
            batch["output_fields"].to(device), mode='variable')

        x0 = rearrange(inp[:, -1:], 'B 1 H W C -> B C 1 H W')
        x1 = rearrange(out,         'B T H W C -> B C T H W')
        xp = rearrange(inp[:, :-1], 'B T H W C -> B C T H W')

        scalars = batch['constant_scalars'].to(device) \
                  if 'constant_scalars' in batch.keys() else None

        cond = {'const': scalars, 'past_states': xp}
        return x0, x1, cond, x0.shape[0]

    # ─────────────────────────────────────────────────────────────────────────
    # Loss
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_loss(self, batch, log_prefix: str) -> torch.Tensor:
        """
        OT-CFM loss in Fourier space, weighted sum over pyramid scales.

        Uses _subband_vmse to normalise the real and imaginary parts of each
        Fourier coefficient independently, preventing the large DC component
        from dominating the gradient and ensuring high-frequency modes receive
        meaningful learning signal.

        OT-CFM interpolant
        ------------------
            x(τ) = τ · F(x₁) + (1 - τ) · ε,   ε ~ N(0, I)   τ ∈ (0, 1]

        Conditional velocity target
        ---------------------------
            u(x(τ) | x₁) = (x₁ - x(τ)) / (1 - τ)
                          = F(x₁) - (x(τ) - τ·F(x₁)) / (1 - τ)
        """
        x0, x1, cond, B = self._prepare_batch(batch)

        c0 = self._to_coeff_dict_x0(x0)
        c1 = self._to_coeff_dict(x1)

        # Sample flow time uniformly; clamp away from the endpoints to avoid
        # numerical instability in the 1/(1-τ) denominator.
        t     = torch.rand(B, device=self.device).clamp(1e-5, 1 - 1e-5)
        t_exp = t.view(-1, 1, 1, 1)

        # OT-CFM interpolant: mix target Fourier coefficients with isotropic
        # Gaussian noise in coefficient space.
        x_tau = {
            i: t_exp * c1[i] + (1.0 - t_exp) * torch.randn_like(c1[i])
            for i in range(self.n_scales)
        }
        # Closed-form conditional velocity target.
        targets = {
            i: c1[i] - (x_tau[i] - t_exp * c1[i]) / (1.0 - t_exp).clamp(min=1e-5)
            for i in range(self.n_scales)
        }

        vel = self.net(
            curr        = c0,
            x_tau       = x_tau,
            t           = t,
            scalar_conds= cond['const'],
            context     = cond['past_states'],
        )

        w_sum = self.scale_weights.sum().clamp(min=1e-8)
        total = x0.new_zeros(())

        for i in range(self.n_scales):
            loss_i = _subband_vmse(
                pred           = vel[i],
                target         = targets[i],
                in_channels    = self.in_channels,
                n_steps_output = self.n_steps_output,
                n_coeff        = self.scale_shapes[i]['n_coeff'],  # 2
                subband_weights= self.subband_weights,
            )
            total = total + (self.scale_weights[i] / w_sum) * loss_i
            self.log(
                f"{log_prefix}_loss_s{i}", loss_i,
                on_epoch=True,
                on_step=(log_prefix == 'train'),
                prog_bar=False, sync_dist=True, batch_size=B,
            )

        self.log(
            f"{log_prefix}_loss", total,
            on_epoch=True,
            on_step=(log_prefix == 'train'),
            prog_bar=True, sync_dist=True, batch_size=B,
        )
        return total

    # ─────────────────────────────────────────────────────────────────────────
    # Lightning steps
    # ─────────────────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, 'val')

    # ─────────────────────────────────────────────────────────────────────────
    # Sampling  (Euler integration in Fourier coefficient space)
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, batch, batch_idx, n_ensemble: Optional[int] = None):
        """
        Euler integration of the learned vector field in Fourier space.
        Returns (yhat, y) where shapes are (B, E, T, H, W, C) and (B, T, H, W, C)
        respectively, both in physical (de-normalised) units.
        """
        fm    = self.flow_matching_cfg
        n_ens = n_ensemble or fm.n_ensemble
        dt    = 1.0 / fm.num_sampling_steps

        x0, x1, cond, B = self._prepare_batch(batch)

        c0 = self._to_coeff_dict_x0(x0)
        c1 = self._to_coeff_dict(x1)

        members: List[Dict[int, torch.Tensor]] = []

        for _ in range(n_ens):
            # Initialise from standard Gaussian in Fourier coefficient space.
            x_tau = {i: torch.randn_like(c1[i]) for i in range(self.n_scales)}

            for step in range(fm.num_sampling_steps):
                t_batch = torch.full(
                    (B,), step / fm.num_sampling_steps,
                    device=self.device, dtype=x0.dtype,
                )
                vel = self.net(
                    curr        = c0,
                    x_tau       = x_tau,
                    t           = t_batch,
                    scalar_conds= cond['const'],
                    context     = cond['past_states'],
                )
                for i in range(self.n_scales):
                    x_tau[i] = x_tau[i] + dt * vel[i]

            members.append(self._from_coeff_dict(x_tau))

        yhat = torch.stack(members, dim=1)          # (B, E, T, H, W, C)
        y = self.normalizer.denormalize_flattened(
            rearrange(x1, 'B C T H W -> B T H W C'), mode='variable'
        )
        return yhat, y

    # ─────────────────────────────────────────────────────────────────────────
    # Autoregressive rollout
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def rollout(
        self,
        seed            : torch.Tensor,
        scalar_conds    : torch.Tensor,
        rollout_steps   : int,
        n_steps_input   : Optional[int] = None,
        n_ensemble      : Optional[int] = None,
        normalized_input: bool  = False,
        verbose         : bool  = False,
        ode_method      : str   = "heun",
        clamp_sigma     : float = 5.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive rollout with configurable ODE solver in Fourier space.

        Supports Euler, Heun (default), and midpoint methods — identical to
        WaveletFlowModule.rollout in structure and interface.

        Returns
        -------
        preds_norm  : (B, E, rollout_steps, H, W, C)  normalised
        preds_phys  : (B, E, rollout_steps, H, W, C)  physical (de-normalised)
        """
        fm     = self.flow_matching_cfg
        n_ens  = n_ensemble if n_ensemble is not None else fm.n_ensemble
        T_out  = self.n_steps_output
        device = self.device
        dtype  = next(self.parameters()).dtype

        if seed.ndim != 5:
            raise ValueError(
                f"seed must be (B, Ti, H, W, C), got {tuple(seed.shape)}"
            )
        B, Ti_seed, H, W, C = seed.shape
        Ti = n_steps_input if n_steps_input is not None else Ti_seed
        if Ti_seed < Ti:
            raise ValueError(
                f"seed has {Ti_seed} frames but n_steps_input={Ti}."
            )

        seed = seed.to(device=device, dtype=dtype)
        if not normalized_input:
            seed_norm = self.normalizer.normalize_flattened(seed, mode='variable')
        else:
            seed_norm = seed
        seed_norm = seed_norm[:, -Ti:]

        # Expand seed to all ensemble members: (B*E, Ti, H, W, C)
        windows = (
            seed_norm.unsqueeze(1)
            .expand(-1, n_ens, -1, -1, -1, -1)
            .reshape(B * n_ens, Ti, H, W, C)
            .clone()
        )

        if scalar_conds is not None:
            scalar_conds = scalar_conds.to(device=device, dtype=dtype)
            scalar_exp = (
                scalar_conds.unsqueeze(1)
                .expand(-1, n_ens, -1)
                .reshape(B * n_ens, -1)
            )
        else:
            scalar_exp = None

        BxE = B * n_ens
        preds_norm = torch.zeros(
            BxE, rollout_steps, H, W, C, device='cpu', dtype=dtype
        )

        step_out   = 0
        n_calls    = math.ceil(rollout_steps / T_out)
        call_count = 0

        while step_out < rollout_steps:
            call_count += 1
            if verbose:
                print(
                    f"  rollout call {call_count:>4d}/{n_calls} "
                    f"(frames {step_out+1}–{min(step_out+T_out, rollout_steps)}"
                    f"/{rollout_steps})",
                    end='\r',
                )

            curr_norm = rearrange(windows[:, -1:], 'B 1 H W Cv -> B Cv 1 H W')
            past_norm = rearrange(windows[:, :-1], 'B T H W Cv -> B Cv T H W')
            c0 = self._to_coeff_dict_x0(curr_norm)

            # Initialise x_tau from isotropic Gaussian in Fourier space.
            x_tau = {
                i: torch.randn(
                    BxE,
                    self.in_channels * T_out * self.scale_shapes[i]['n_coeff'],
                    self.scale_shapes[i]['H'],
                    self.scale_shapes[i]['W'],
                    device=device, dtype=dtype,
                )
                for i in range(self.n_scales)
            }

            # ── ODE integration ────────────────────────────────────────────
            for s in range(fm.num_sampling_steps):
                t_s    = s / fm.num_sampling_steps
                t_next = (s + 1) / fm.num_sampling_steps
                dt_s   = t_next - t_s
                t_batch = torch.full((BxE,), t_s, device=device, dtype=dtype)

                if ode_method == "euler":
                    vel = self.net(
                        curr=c0, x_tau=x_tau, t=t_batch,
                        scalar_conds=scalar_exp, context=past_norm,
                    )
                    for i in range(self.n_scales):
                        x_tau[i] = x_tau[i] + dt_s * vel[i]

                elif ode_method == "heun":
                    vel1 = self.net(
                        curr=c0, x_tau=x_tau, t=t_batch,
                        scalar_conds=scalar_exp, context=past_norm,
                    )
                    x_tau_pred = {
                        i: x_tau[i] + dt_s * vel1[i]
                        for i in range(self.n_scales)
                    }
                    t_next_batch = torch.full(
                        (BxE,), t_next, device=device, dtype=dtype
                    )
                    vel2 = self.net(
                        curr=c0, x_tau=x_tau_pred, t=t_next_batch,
                        scalar_conds=scalar_exp, context=past_norm,
                    )
                    for i in range(self.n_scales):
                        x_tau[i] = x_tau[i] + dt_s * 0.5 * (vel1[i] + vel2[i])

                elif ode_method == "midpoint":
                    vel1 = self.net(
                        curr=c0, x_tau=x_tau, t=t_batch,
                        scalar_conds=scalar_exp, context=past_norm,
                    )
                    x_tau_mid = {
                        i: x_tau[i] + 0.5 * dt_s * vel1[i]
                        for i in range(self.n_scales)
                    }
                    t_mid = torch.full(
                        (BxE,), t_s + 0.5 * dt_s, device=device, dtype=dtype
                    )
                    vel_mid = self.net(
                        curr=c0, x_tau=x_tau_mid, t=t_mid,
                        scalar_conds=scalar_exp, context=past_norm,
                    )
                    for i in range(self.n_scales):
                        x_tau[i] = x_tau[i] + dt_s * vel_mid[i]

                else:
                    raise ValueError(
                        f"Unknown ode_method={ode_method!r}. "
                        "Choose 'euler', 'heun', or 'midpoint'."
                    )

            # Reconstruct via irfft2 from scale-0 Fourier coefficients.
            pred_norm = self._to_spatial_normalized(x_tau)  # (BxE, T, H, W, C)

            frames_to_store = min(T_out, rollout_steps - step_out)
            preds_norm[:, step_out : step_out + frames_to_store] = (
                pred_norm[:, :frames_to_store].cpu()
            )
            step_out += frames_to_store

            # Clamp normalised predictions before feeding back as context to
            # prevent runaway autoregressive rollout.
            if clamp_sigma > 0:
                pred_norm = pred_norm.clamp(-clamp_sigma, clamp_sigma)

            windows = torch.cat([windows, pred_norm], dim=1)[:, -Ti:]

        if verbose:
            print()

        preds_norm_out = preds_norm.view(B, n_ens, rollout_steps, H, W, C)
        preds_phys = self.normalizer.denormalize_flattened(
            preds_norm_out.reshape(B * n_ens * rollout_steps, H, W, C).to(device),
            mode='variable',
        ).reshape(B, n_ens, rollout_steps, H, W, C).cpu()

        return preds_norm_out, preds_phys

    # ─────────────────────────────────────────────────────────────────────────
    # Optimiser + scheduler  (identical to WaveletFlowModule)
    # ─────────────────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        oc  = self.optimizer_cfg
        opt = eval(oc.cls)(self.parameters(), **oc.optimizer_kwargs())

        if not oc.scheduler_cls:
            return opt

        cosine_epochs = max(self.trainer.max_epochs - oc.warmup_epochs, 1)
        sched = SequentialLR(
            opt,
            schedulers=[
                LinearLR(opt, start_factor=1e-3, end_factor=1.0,
                         total_iters=oc.warmup_epochs),
                CosineAnnealingLR(opt, T_max=cosine_epochs, eta_min=oc.eta_min),
            ],
            milestones=[oc.warmup_epochs],
        )
        return {
            "optimizer"   : opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch", "frequency": 1},
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Epoch-end plotting
    # ─────────────────────────────────────────────────────────────────────────

    def on_train_epoch_end(self, ensemble_reduce: str = 'random'):
        if self.experiment_manager is None:
            return
        should_save = (
            self.current_epoch % self.experiment_manager.log_every_n_epochs == 0
            or self.current_epoch == self.trainer.max_epochs - 1
        )
        if not should_save:
            return

        batch, batch_idx = self._get_random_validation_sample()
        yhat, y = self.sample(batch, batch_idx)

        E = yhat.shape[1]
        if ensemble_reduce == 'mean':
            yhat_reduced = yhat.mean(dim=1)
        else:
            e = torch.randint(0, E, (1,)).item()
            yhat_reduced = yhat[:, e]

        self.experiment_manager.dump_sample_plots(
            yhat        = rearrange(yhat_reduced, 'B T H W C -> B C T H W'),
            y           = rearrange(y,            'B T H W C -> B C T H W'),
            field_names = self.field_names,
            fig_name    = f"epoch_{self.current_epoch}_val_batch_{batch_idx}.png",
            cmap        = 'RdBu_r',
            dpi         = 300,
        )

    def _get_random_validation_sample(self):
        loader = self.trainer.val_dataloaders
        idx    = torch.randint(0, len(loader), (1,)).item()
        for i, batch in enumerate(loader):
            if i == idx:
                break
        B = batch["output_fields"].shape[0]
        s = torch.randint(0, B, (1,)).item()
        return {k: v[s:s + 1] for k, v in batch.items()}, idx

    # ─────────────────────────────────────────────────────────────────────────
    # Portable checkpoint loader
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        path        : str,
        map_location: str  = "cpu",
        stats       : dict = None,
    ) -> "FourierFlowModule":
        """
        Load a FourierFlowModule from a Lightning checkpoint.

        Parameters
        ----------
        path         : path to the .ckpt file
        map_location : torch map_location string or device
        stats        : optional normalizer stats dict to override the stored one
                       (useful when evaluating on a different normalizer version)
        """
        from the_well.data.normalization import ZScoreNormalization

        ckpt = torch.load(path, map_location=map_location)
        hp   = ckpt["hyper_parameters"]

        if stats is not None:
            normalizer_stats = stats
        elif hp.get("normalizer_stats") is not None:
            normalizer_stats = hp["normalizer_stats"]
        else:
            raise KeyError(
                f"Cannot find normalizer stats in checkpoint. "
                f"Keys present: {list(hp.keys())}"
            )

        normalizer = ZScoreNormalization(
            stats                     = normalizer_stats,
            core_field_names          = hp["normalizer_core_field_names"],
            core_constant_field_names = hp["normalizer_core_constant_field_names"],
        )

        model = cls(
            in_channels       = hp["in_channels"],
            field_names       = hp["field_names"],
            data_shape        = tuple(hp["data_shape"]),
            n_steps_output    = hp["n_steps_output"],
            scale_weights     = hp["scale_weights"],
            subband_weights   = hp.get("subband_weights", None),
            fourier_cfg       = FourierConfig.from_dict(hp["fourier_cfg"]),
            velocity_net_cfg  = VelocityNetConfig.from_dict(hp["velocity_net_cfg"]),
            flow_matching_cfg = FlowMatchingConfig.from_dict(hp["flow_matching_cfg"]),
            optimizer_cfg     = OptimizerConfig.from_dict(hp["optimizer_cfg"]),
            normalizer        = normalizer,
        )
        model.load_state_dict(ckpt["state_dict"])
        return model


# ─────────────────────────────────────────────────────────────────────────────
# train.py integration notes
# ─────────────────────────────────────────────────────────────────────────────
#
# The following minimal edits to train.py switch the training loop from
# WaveletFlowModule to FourierFlowModule.  Everything else (data loading,
# trainer, callbacks, experiment manager) is unchanged.
#
# 1. Replace imports
# ──────────────────
#   - from waveletflux.modules.wavelet_fm import WaveletFlowModule
#   + from waveletflux.modules.fourier_fm import FourierFlowModule, FourierConfig
#
#   Remove:
#   - from waveletflux.utils.configs import WaveletConfig, ...
#   Add:
#   + from waveletflux.utils.configs import VelocityNetConfig, FlowMatchingConfig, OptimizerConfig
#
# 2. Replace typed config construction
# ─────────────────────────────────────
#   - wavelet_cfg      = WaveletConfig.from_config(config.wavelet)
#   + fourier_cfg      = FourierConfig.from_config(config.fourier)
#
# 3. Update scale_weights assertion
# ──────────────────────────────────
#   - assert len(config.loss.scale_weights) == wavelet_cfg.n_scales, ...
#   + assert len(config.loss.scale_weights) == fourier_cfg.n_scales, ...
#
# 4. Replace model construction
# ──────────────────────────────
#   - model = WaveletFlowModule(
#   -     wavelet_cfg       = wavelet_cfg,
#   + model = FourierFlowModule(
#   +     fourier_cfg       = fourier_cfg,
#       # all other arguments are identical
#     )
#
# 5. Remove the patched-checkpoint resume block for subband_weights
#    (FourierFlowModule always stores it; no patching needed for new runs).
#    Keep it if you need to resume from an old WaveletFlowModule checkpoint.
#
# ─────────────────────────────────────────────────────────────────────────────
# Sample TOML configuration  (rayleigh_benard, single-scale Fourier)
# ─────────────────────────────────────────────────────────────────────────────
#
# [main]
#     exp_name = 'FourierFlow_rayleigh_benard_64modes_v1.0'
#
# [resume]
# ckpt = ''
#
# [path]
# project_path      = '/glade/work/gaccarino/waveletflux'
# well_base_path    = '/glade/work/gaccarino/data/datasets'
# well_dataset_name = 'rayleigh_benard'
# stats_path        = '/glade/work/gaccarino/data/datasets/rayleigh_benard/stats.yaml'
# experiments_path  = '/glade/work/gaccarino/waveletflux/experiments/rayleigh_benard'
#
# [random]
# seed = 8940
#
# [train]
# epochs            = 200
# warmup_epochs     = 20
# batch_size        = 32
# n_steps_input     = 4
# n_steps_output    = 1
# max_rollout_steps = 1
# train_shuffle     = true
# val_shuffle       = false
# test_shuffle      = false
# rollout_shuffle   = false
# target_dtype      = 'torch.float32'
# augmentations     = []
# transformations   = ['log_scalars']
#
# ── Fourier decomposer (replaces [wavelet]) ───────────────────────────────────
#
# [fourier]
# [fourier.params]
# n_modes_h = 64      # Fourier row modes to keep  (≤ H)
# n_modes_w = 64      # Fourier col modes to keep  (≤ W // 2 + 1)
# n_scales  = 1       # 1 = single truncated spectrum (recommended)
#                     # 3 = pyramid: 64, 32, 16 row modes; requires n_levels >= 3
# norm      = 'ortho'
#
# [velocity_net]
# cls = 'WaveletFlowUNet'          # same network, no change needed
# [velocity_net.params]
# num_cond            = 2
# context_in_channels = 4
# num_context_frames  = 3
# n_levels = 3                     # must be >= fourier.params.n_scales
# base_ch  = 64
# n_res    = 3
# cond_dim = 256
#
# [flow_matching]
# cls        = 'OT_CFM'
# n_ensemble = 1
# [flow_matching.params]
# sigma = 0.0
#
# [loss]
# # One entry per fourier.params.n_scales
# scale_weights   = [1.0]          # n_scales=1: single weight
# # subband_weights = [1.0, 1.0]   # optional: [w_real, w_imag]; omit for equal weighting
#
# [sampling]
# solver             = 'heun'
# num_sampling_steps = 20
#
# [optimizer]
# cls = 'AdamW'
# [optimizer.params]
# lr           = 1e-4
# weight_decay = 1e-2
# betas        = [0.9, 0.999]
#
# [scheduler]
# cls = 'CosineAnnealingLR'
# [scheduler.params]
# eta_min = 1e-7
#
# [lightning]
# [lightning.datamodules]
# data_workers = 16
# [lightning.trainer]
# accelerator         = 'gpu'
# strategy            = 'ddp'
# precision           = 'bf16-mixed'
# matmul_precision    = 'high'
# gradient_clip_val   = 1.0
# log_every_n_steps   = 1
# limit_train_batches = 1.0
# limit_val_batches   = 1.0
#
# [loggers]
# loggers            = ['csv']
# log_every_n_epochs = 5
#
# [callbacks]
# callbacks = ['model_checkpoint_cb', 'emergency_cb', 'performance_cb']
#
# ─── Multi-scale pyramid variant (n_scales=3) ────────────────────────────────
#
# [fourier.params]
# n_modes_h = 64   # scale 0: 64 rows, scale 1: 32, scale 2: 16
# n_modes_w = 64
# n_scales  = 3
# norm      = 'ortho'
#
# [velocity_net.params]
# n_levels = 3     # must be >= n_scales = 3
# ...              # rest unchanged
#
# [loss]
# scale_weights   = [1.0, 0.5, 0.25]   # down-weight coarser auxiliary scales
# subband_weights = [1.0, 1.0]          # real and imaginary parts equally weighted
