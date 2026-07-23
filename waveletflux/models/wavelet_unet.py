"""
wavelet_unet.py  —  decoupled n_scales / n_levels.

Design
------
n_scales  (wavelet depth)   — controls injection points and the number of
                              velocity outputs (one per wavelet scale).
n_levels  (UNet depth)      — controls the number of downsample / upsample
                              steps.  Must satisfy n_levels >= n_scales.

When n_levels == n_scales the behaviour is equivalent to the previous
single-parameter design (n_scales governed both).  Setting n_levels > n_scales
adds extra encoder/decoder levels that deepen the bottleneck without additional
wavelet injections, increasing the effective receptive field.  This lets you
use the same UNet depth for both wavelet-space and data-space training:

    n_scales=3, n_levels=3   — full wavelet UNet, bottleneck one level deeper
                               than the coarsest wavelet scale
    n_scales=1, n_levels=3   — data-space UNet with the same depth; no wavelet
                               injection beyond the input stem

T-dimension fix
---------------
wav_ch[i] = C * T * n_coeff[i], matching _to_coeff_dict which packs
(B, C, T, Coeff, H, W) -> (B, C*T*Coeff, H, W).
n_steps_output is now an explicit constructor argument so the stem channel
sizes are correct even when predicting multiple future frames.
Callers must tile x0 to (B, C, T, H, W) before calling _to_coeff_dict so
that curr[i] and x_tau[i] always have the same channel count.

Encoder (for lvl = 0 .. n_levels-1)
--------------------------------------
  x = enc_block[lvl](x)           n_res ResBlocks at chs[lvl]
  skip[lvl] = x
  x = enc_down[lvl](x)            stride-2 conv, chs[lvl] -> chs[lvl+1]
  if lvl+1 < n_scales:
      x = align_to(x, stem[lvl+1])
      x = inject[lvl](cat(x, stem[lvl+1]))

Bottleneck (level n_levels, channels chs[n_levels])
----------------------------------------------------
  x = max(n_res, 2) x ResBlock(x)

Decoder (for j = 0 .. n_levels-1)
------------------------------------
  x = dec_up[j](x)
  current_level = n_levels - j - 1
  x = align_to(x, skip[current_level])
  x = dec_block[j](cat(x, skip[current_level]))
  if current_level < n_scales:
      vel[current_level] = out_head[current_level](x)

All velocity outputs are produced in the decoder at the spatial resolution
that matches their wavelet scale.  There is no coarse_head at the bottleneck.
"""

from typing import Dict, List, Optional
import torch.nn.functional as F
import torch.nn as nn
import torch
import math


# ─────────────────────────────────────────────────────────────────────────────
# Spatial helper
# ─────────────────────────────────────────────────────────────────────────────

def _align_to(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Crop then replicate-pad x so its last two dims are exactly (H, W).
    Handles the ±1 pixel mismatches that arise from non-Haar wavelets and
    stride-2 convolutions when the spatial size is odd.
    """
    x = x[..., :H, :W]
    dh = H - x.shape[-2]
    dw = W - x.shape[-1]
    if dh > 0 or dw > 0:
        x = F.pad(x, (0, dw, 0, dh), mode='replicate')
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Conditioning helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Standard sinusoidal time embedding → (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / max(half - 1, 1)
    )
    x = t.float()[:, None] * freqs[None]
    return torch.cat([x.sin(), x.cos()], dim=-1)


def _num_groups(ch: int, target: int = 8) -> int:
    for g in [target, target // 2, 4, 2, 1]:
        if g > 0 and ch % g == 0:
            return g
    return 1


# ─────────────────────────────────────────────────────────────────────────────
# Context encoder
# ─────────────────────────────────────────────────────────────────────────────

class ContextEncoder(nn.Module):
    """
    Lightweight spatiotemporal context encoder.

    (B, C, T, H, W)  →  (B, cond_dim)

    1. Spatial mean-pool each frame      →  (B, C, T)
    2. Linear projection                 →  (B, T, cond_dim)
    3. Learned query cross-attends       →  (B, 1, cond_dim)
    4. LayerNorm + linear output         →  (B, cond_dim)

    Uses .mean() for spatial pooling — avoids adaptive_avg_pool2d,
    which is broken on MPS for non-divisible sizes.
    """

    def __init__(
        self,
        context_channels : int,
        n_context_frames : int,
        cond_dim         : int,
        n_heads          : int = 4,
    ):
        super().__init__()
        self.frame_proj = nn.Linear(context_channels, cond_dim)
        self.query      = nn.Parameter(torch.randn(1, 1, cond_dim) * 0.02)
        self.attn       = nn.MultiheadAttention(cond_dim, num_heads=n_heads,
                                                batch_first=True)
        self.norm       = nn.LayerNorm(cond_dim)
        self.out_proj   = nn.Linear(cond_dim, cond_dim)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        # ctx : (B, C, T, H, W)
        B = ctx.shape[0]
        x = ctx.mean(dim=(-2, -1))           # (B, C, T)
        x = x.permute(0, 2, 1)              # (B, T, C)
        x = self.frame_proj(x)              # (B, T, cond_dim)
        q = self.query.expand(B, -1, -1)    # (B, 1, cond_dim)
        out, _ = self.attn(q, x, x)         # (B, 1, cond_dim)
        out = self.norm(out.squeeze(1))      # (B, cond_dim)
        return self.out_proj(out)            # (B, cond_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    Pre-activation ResBlock with AdaGN conditioning.
    The second conv is zero-initialised so the block starts as identity.
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch),  in_ch)
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.cond  = nn.Linear(cond_dim, out_ch * 2)   # scale + shift
        self.skip  = (nn.Conv2d(in_ch, out_ch, 1, bias=False)
                      if in_ch != out_ch else nn.Identity())
        self.act   = nn.SiLU()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        sc, sh = self.cond(self.act(c)).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + sc[:, :, None, None]) + sh[:, :, None, None]
        h = self.conv2(self.act(h))
        return h + self.skip(x)


def _downsample(in_ch: int, out_ch: int) -> nn.Conv2d:
    """Stride-2 3×3 conv — avoids aliasing, no transposed conv."""
    return nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)


def _upsample(in_ch: int, out_ch: int) -> nn.Sequential:
    """
    Bilinear ×2 followed by a 3×3 conv.
    Avoids the 2×-frequency checkerboard pattern of transposed conv.
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
    )


def _out_head(feat_ch: int, out_ch: int) -> nn.Sequential:
    """Output projection: GroupNorm + SiLU + Conv."""
    return nn.Sequential(
        nn.GroupNorm(_num_groups(feat_ch), feat_ch),
        nn.SiLU(),
        nn.Conv2d(feat_ch, out_ch, 3, padding=1, bias=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main network
# ─────────────────────────────────────────────────────────────────────────────

class WaveletFlowUNet(nn.Module):
    """
    Multi-scale wavelet flow UNet with decoupled n_scales / n_levels.

    Parameters
    ----------
    scale_shapes     : {i: {'H', 'W', 'n_coeff'}} from the wavelet probe.
                       Length must equal n_scales.
    in_channels      : number of physical fields C.
    n_steps_output   : T — number of predicted future frames.  Used to
                       compute correct stem input sizes: each stem takes
                       cat(curr[i], x_tau[i]) with curr[i] and x_tau[i]
                       both shaped (B, C*T*n_coeff_i, H_i, W_i).
    n_scales         : number of wavelet scales; controls injection points
                       and output heads.
    n_levels         : UNet depth (number of down/upsamplings).
                       Constraint: n_levels >= n_scales.
    context_channels : C of past-context frames (0 = disabled).
    n_context_frames : T of past-context frames (0 = disabled).
    cond_dim         : width of the time / scalar conditioning vector.
    num_scalar_conds : number of extra scalar conditions (e.g. Ra, Pr).
    base_ch          : channel width at the finest encoder level.
    n_res            : ResBlocks per encoder / decoder level.
    """

    def __init__(
        self,
        scale_shapes    : Dict[int, dict],
        in_channels     : int,
        n_steps_output  : int,
        n_scales        : int,
        n_levels        : int,
        context_channels: int = 0,
        n_context_frames: int = 0,
        cond_dim        : int = 128,
        num_scalar_conds: int = 0,
        base_ch         : int = 32,
        n_res           : int = 1,
    ):
        super().__init__()
        assert n_scales >= 1, "n_scales must be >= 1"
        assert n_levels >= n_scales, (
            f"n_levels ({n_levels}) must be >= n_scales ({n_scales}). "
            f"Every wavelet scale needs a corresponding UNet level."
        )

        self.n_scales       = n_scales
        self.n_levels       = n_levels
        self.scale_shapes   = scale_shapes
        self.in_channels    = in_channels
        self.n_steps_output = n_steps_output

        # Channel widths at UNet levels 0 .. n_levels (n_levels+1 entries).
        # level 0 = finest (base_ch), level n_levels = bottleneck.
        self.chs = [min(base_ch * (2 ** lvl), base_ch * 8)
                    for lvl in range(n_levels + 1)]

        # Wavelet coefficient channels per scale: C * T * n_coeff_i.
        # Matches _to_coeff_dict: (B,C,T,Coeff,H,W) → (B, C·T·Coeff, H, W).
        self.wav_ch = {
            i: in_channels * n_steps_output * scale_shapes[i]['n_coeff']
            for i in range(n_scales)
        }

        # ── Context encoder ───────────────────────────────────────────────────
        self.has_context = context_channels > 0 and n_context_frames > 0
        self.context_encoder = (
            ContextEncoder(context_channels, n_context_frames, cond_dim)
            if self.has_context else None
        )

        # ── Conditioning MLP ──────────────────────────────────────────────────
        self._cond_dim  = cond_dim
        self.time_embed = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 2), nn.SiLU(),
            nn.Linear(cond_dim * 2, cond_dim),
        )
        
        self.scalar_proj = (
            nn.Linear(num_scalar_conds, cond_dim)
            if num_scalar_conds > 0 else None
        )
        #self.scalar_proj = nn.Sequential(
        #    nn.Linear(num_scalar_conds, cond_dim), nn.SiLU(),
        #    nn.Linear(cond_dim, cond_dim),
        #) if num_scalar_conds > 0 else None

        # ── Stems (one per wavelet scale) ─────────────────────────────────────
        # stem[0]  : initial input at UNet level 0
        #            2·wav_ch[0] → chs[0]
        # stem[i>0]: provides wavelet scale-i features for injection at level i
        #            2·wav_ch[i] → chs[i]
        self.stems = nn.ModuleList([
            nn.Conv2d(2 * self.wav_ch[i], self.chs[i], 3, padding=1, bias=False)
            for i in range(n_scales)
        ])

        # ── Encoder ───────────────────────────────────────────────────────────
        # enc_blocks[lvl]: n_res ResBlocks at channel width chs[lvl],
        #                  applied before downsampling.
        self.enc_blocks = nn.ModuleList([
            nn.ModuleList([
                ResBlock(self.chs[lvl], self.chs[lvl], cond_dim)
                for _ in range(n_res)
            ])
            for lvl in range(n_levels)
        ])

        # enc_downs[lvl]: stride-2 conv chs[lvl] → chs[lvl+1].
        self.enc_downs = nn.ModuleList([
            _downsample(self.chs[lvl], self.chs[lvl + 1])
            for lvl in range(n_levels)
        ])

        # injects[i]: 1×1 conv that merges downsampled encoder features with
        #             wavelet stem features at scale i+1.
        #             Input: cat(x_down, stem[i+1]) = 2·chs[i+1] channels.
        #             Only built for i = 0 .. n_scales-2.
        self.injects = nn.ModuleList([
            nn.Conv2d(self.chs[i + 1] * 2, self.chs[i + 1], 1, bias=False)
            for i in range(n_scales - 1)
        ])

        # ── Bottleneck ────────────────────────────────────────────────────────
        # Operates at the coarsest UNet level (level n_levels, chs[n_levels]).
        # No velocity output here; all vel outputs are in the decoder.
        self.bottleneck = nn.ModuleList([
            ResBlock(self.chs[n_levels], self.chs[n_levels], cond_dim)
            for _ in range(max(n_res, 2))
        ])

        # ── Decoder ───────────────────────────────────────────────────────────
        # Decoder step j goes from level n_levels-j to level n_levels-j-1.
        #
        # dec_ups[j]   : _upsample(chs[n_levels-j], chs[n_levels-j-1])
        # dec_blocks[j]: first ResBlock reduces 2·chs → chs after skip-cat,
        #                subsequent ResBlocks maintain chs.
        self.dec_ups = nn.ModuleList([
            _upsample(self.chs[n_levels - j], self.chs[n_levels - j - 1])
            for j in range(n_levels)
        ])
        self.dec_blocks = nn.ModuleList([
            nn.ModuleList(
                [ResBlock(self.chs[n_levels - j - 1] * 2,
                          self.chs[n_levels - j - 1], cond_dim)]
                + [ResBlock(self.chs[n_levels - j - 1],
                            self.chs[n_levels - j - 1], cond_dim)
                   for _ in range(max(n_res - 1, 0))]
            )
            for j in range(n_levels)
        ])

        # ── Output heads (one per wavelet scale) ──────────────────────────────
        # out_heads[k] is applied at the decoder step where current_level == k,
        # i.e. step j = n_levels - 1 - k.  It maps chs[k] → wav_ch[k].
        self.out_heads = nn.ModuleList([
            _out_head(self.chs[k], self.wav_ch[k])
            for k in range(n_scales)
        ])

    # ── helpers ───────────────────────────────────────────────────────────────

    def _embed_cond(
        self,
        t           : torch.Tensor,
        scalar_conds: Optional[torch.Tensor],
        context     : Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compose conditioning vector (B, cond_dim) from:
          1. sinusoidal time embedding
          2. optional scalar conditions (summed)
          3. optional context frames via ContextEncoder (summed)
        """
        emb = self.time_embed(_sinusoidal(t, self._cond_dim))
        if self.scalar_proj is not None and scalar_conds is not None:
            emb = emb + self.scalar_proj(scalar_conds.float())
        if self.context_encoder is not None and context is not None:
            emb = emb + self.context_encoder(context)
        return emb   # (B, cond_dim)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        curr        : Dict[int, torch.Tensor],
        x_tau       : Dict[int, torch.Tensor],
        t           : torch.Tensor,
        scalar_conds: Optional[torch.Tensor] = None,
        context     : Optional[torch.Tensor] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Parameters
        ----------
        curr   : {i: (B, C·T·n_coeff_i, H_i, W_i)}
                 Wavelet coefficients of the current frame, tiled to T along
                 the temporal axis before calling _to_coeff_dict.
        x_tau  : {i: (B, C·T·n_coeff_i, H_i, W_i)}
                 Noisy OT-CFM interpolant at flow time τ.
        t      : (B,) flow-matching time.
        scalar_conds, context : optional conditioning signals.

        Returns
        -------
        vel    : {i: (B, C·T·n_coeff_i, H_i, W_i)}
                 Predicted velocity at each wavelet scale.
        """
        cond = self._embed_cond(t, scalar_conds, context)

        # ── 1. Stem features for every wavelet scale ──────────────────────────
        stem_feats: Dict[int, torch.Tensor] = {
            i: self.stems[i](torch.cat([curr[i], x_tau[i]], dim=1))
            for i in range(self.n_scales)
        }

        # ── 2. Encoder ────────────────────────────────────────────────────────
        x     = stem_feats[0]
        skips : List[torch.Tensor] = []

        for lvl in range(self.n_levels):
            for blk in self.enc_blocks[lvl]:
                x = blk(x, cond)

            skips.append(x)                              # save skip at level lvl
            x = self.enc_downs[lvl](x)                  # stride-2 → level lvl+1

            if lvl + 1 < self.n_scales:                 # inject wavelet scale lvl+1
                sf = stem_feats[lvl + 1]
                x  = _align_to(x, sf.shape[-2], sf.shape[-1])
                x  = self.injects[lvl](torch.cat([x, sf], dim=1))

        # ── 3. Bottleneck ─────────────────────────────────────────────────────
        for blk in self.bottleneck:
            x = blk(x, cond)

        # ── 4. Decoder ────────────────────────────────────────────────────────
        vel: Dict[int, torch.Tensor] = {}

        for j in range(self.n_levels):
            x = self.dec_ups[j](x)

            current_level = self.n_levels - j - 1
            skip = skips[current_level]
            x    = _align_to(x, skip.shape[-2], skip.shape[-1])
            x    = torch.cat([x, skip], dim=1)

            for blk in self.dec_blocks[j]:
                x = blk(x, cond)

            if current_level < self.n_scales:            # produce velocity output
                vel[current_level] = self.out_heads[current_level](x)

        return vel