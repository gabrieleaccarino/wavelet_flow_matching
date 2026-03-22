"""
wavelet_unet.py

Minimal UNet for multi-scale wavelet flow matching.

Design principles
-----------------
* ALL coefficients (LL + LH/HL/HH) are flow-matched — no deterministic LL branch.
  The network predicts a velocity field for every coefficient at every scale.

* Input  at scale i : cat(curr[i], x_tau[i])   →  (B, 2·C_i, H_i, W_i)
* Output at scale i : velocity field            →  (B,   C_i, H_i, W_i)

* Spatial alignment : ALL feature maps are aligned to wavelet spatial sizes via
  a single _align_to() helper.  This is the only alignment logic in the file.
  Non-Haar wavelets (db2, sym4 …) produce ±1-pixel sizes that the stride-2
  encoder would otherwise mishandle; we treat wavelet shape as authoritative
  and crop/pad features to match.

* Anti-checkerboard : upsampling uses bilinear interp + conv, never transposed
  conv.  Transposed conv with stride 2 is the primary source of the 2×-periodic
  grid artifacts that appear in IDWT output.

* Condition : sinusoidal time embedding + optional scalar conditions, injected
  via AdaGN (scale/shift) at every ResBlock.
"""

from typing import Dict, List, Optional
import torch.nn.functional as F
import torch.nn as nn
import torch
import math


# ─────────────────────────────────────────────────────────────────────────────
# Spatial helpers
# ─────────────────────────────────────────────────────────────────────────────

def _align_to(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Crop then replicate-pad x so its last two dims are exactly (H, W).

    A single call handles both the ±1 wavelet/stride-2 mismatch and the
    decoder upsample/skip mismatch.  No interpolation is used so the
    operation is alias-free and deterministic.
    """
    x = x[..., :H, :W]                                   # crop if too large
    dh = H - x.shape[-2]
    dw = W - x.shape[-1]
    if dh > 0 or dw > 0:
        x = F.pad(x, (0, dw, 0, dh), mode='replicate')   # pad if too small
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Conditioning
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
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """
    Pre-activation ResBlock with AdaGN conditioning.

    cond (B, cond_dim) → scale & shift applied after the second GroupNorm.
    The second conv is zero-initialised so the block starts as an identity.
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch),  in_ch)
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.cond  = nn.Linear(cond_dim, out_ch * 2)       # scale + shift
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
    """Stride-2 3×3 conv — simple, no aliasing."""
    return nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)


def _upsample(in_ch: int, out_ch: int) -> nn.Sequential:
    """
    Bilinear ×2 followed by a 3×3 conv.

    Avoids the 2×-frequency checkerboard pattern that transposed convolution
    with stride 2 produces.  The subsequent conv restores expressivity.
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
    )


def _out_head(feat_ch: int, out_ch: int) -> nn.Sequential:
    """Lightweight output projection with GroupNorm + SiLU."""
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
    Multi-scale wavelet flow UNet.

    Encoder  (fine → coarse)
    ────────────────────────
    For each scale i = 0 … n_scales-2:
      x = enc_block[i](stem_feats[i])
      skip[i] = x
      x = downsample(x)
      x = align_to(x, wavelet_shape[i+1])      ← one-liner spatial fix
      x = inject[i](cat(x, stem_feats[i+1]))   ← blend coarser wavelet input

    Bottleneck  (scale n_scales-1)
    ──────────────────────────────
      x = bottleneck(x)
      vel[n_scales-1] = coarse_head(x)

    Decoder  (coarse → fine)
    ────────────────────────
    For j = 0 … n_scales-2:
      x = upsample(x)
      x = align_to(x, skip[scale_j].shape)     ← same one-liner
      x = dec_block[j](cat(x, skip[scale_j]))
      vel[scale_j] = out_head[j](x)

    Parameters
    ----------
    scale_shapes      : {i: {'H', 'W', 'n_coeff'}} from wavelet decomposer probe
    in_channels       : number of physical fields C
    n_scales          : number of wavelet scales
    context_channels  : C of past-context frames  (0 = disabled)
    n_context_frames  : T of past-context frames  (0 = disabled)
    cond_dim          : time/scalar embedding width
    num_scalar_conds  : number of extra scalar conditions
    base_ch           : channel width at finest encoder level
    n_res             : ResBlocks per encoder/decoder level
    """

    def __init__(
        self,
        scale_shapes    : Dict[int, dict],
        in_channels     : int,
        n_scales        : int,
        context_channels: int = 0,
        n_context_frames: int = 0,
        cond_dim        : int = 128,
        num_scalar_conds: int = 0,
        base_ch         : int = 32,
        n_res           : int = 1,
    ):
        super().__init__()
        assert n_scales >= 1

        self.n_scales     = n_scales
        self.scale_shapes = scale_shapes
        self.in_channels  = in_channels

        # Feature channels at each UNet level; capped to avoid explosion
        self.chs = [min(base_ch * (2 ** i), base_ch * 8) for i in range(n_scales)]

        # Wavelet coefficient channels per scale
        self.wav_ch = {i: in_channels * scale_shapes[i]['n_coeff'] for i in range(n_scales)}

        # ── Context ──────────────────────────────────────────────────────────
        self.has_context = context_channels > 0 and n_context_frames > 0
        ctx_ch = context_channels * n_context_frames if self.has_context else 0

        # ── Condition MLP ─────────────────────────────────────────────────────
        self._cond_dim  = cond_dim
        self.time_embed = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 2), nn.SiLU(),
            nn.Linear(cond_dim * 2, cond_dim),
        )
        self.scalar_proj = (
            nn.Linear(num_scalar_conds, cond_dim) if num_scalar_conds > 0 else None
        )

        # ── Stems (one per scale) ─────────────────────────────────────────────
        # Project wavelet input (curr + x_tau + optional context) → feature space.
        # Context is pooled to the finest-scale spatial size and only added there.
        print(f'[STEM BUILD] context_channels={context_channels}, n_context_frames={n_context_frames}, ctx_ch={context_channels*n_context_frames}')
        self.stems = nn.ModuleList()
        for i in range(n_scales):
            in_c = 2 * self.wav_ch[i] + (ctx_ch if i == 0 else 0)
            self.stems.append(nn.Conv2d(in_c, self.chs[i], 3, padding=1, bias=False))

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc_blocks = nn.ModuleList([
            nn.ModuleList([ResBlock(self.chs[i], self.chs[i], cond_dim)
                           for _ in range(n_res)])
            for i in range(n_scales - 1)
        ])
        self.enc_downs  = nn.ModuleList([
            _downsample(self.chs[i], self.chs[i + 1])
            for i in range(n_scales - 1)
        ])
        # Injection: merge downsampled encoder features + coarser stem features
        self.injects = nn.ModuleList([
            nn.Conv2d(self.chs[i + 1] * 2, self.chs[i + 1], 1, bias=False)
            for i in range(n_scales - 1)
        ])

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = nn.ModuleList([
            ResBlock(self.chs[-1], self.chs[-1], cond_dim)
            for _ in range(max(n_res, 2))
        ])
        self.coarse_head = _out_head(self.chs[-1], self.wav_ch[n_scales - 1])

        # ── Decoder ───────────────────────────────────────────────────────────
        # Indexed j = 0 … n_scales-2, targeting scale i = n_scales-2 … 0
        self._dec_scales = list(range(n_scales - 2, -1, -1))
        self.dec_ups     = nn.ModuleList()
        self.dec_blocks  = nn.ModuleList()
        self.out_heads   = nn.ModuleList()

        for i in self._dec_scales:
            self.dec_ups.append(_upsample(self.chs[i + 1], self.chs[i]))
            self.dec_blocks.append(
                nn.ModuleList(
                    [ResBlock(self.chs[i] * 2, self.chs[i], cond_dim)]      # first: merges skip
                    + [ResBlock(self.chs[i],   self.chs[i], cond_dim)
                       for _ in range(max(n_res - 1, 0))]
                )
            )
            self.out_heads.append(_out_head(self.chs[i], self.wav_ch[i]))

    # ── helpers ───────────────────────────────────────────────────────────────

    def _embed_cond(
        self,
        t           : torch.Tensor,
        scalar_conds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        emb = self.time_embed(_sinusoidal(t, self._cond_dim))
        if self.scalar_proj is not None and scalar_conds is not None:
            emb = emb + self.scalar_proj(scalar_conds.float())
        return emb                                            # (B, cond_dim)

    def _pool_context(
        self,
        ctx: torch.Tensor,
        H  : int,
        W  : int,
    ) -> torch.Tensor:
        """Pool (B, C, T, Hf, Wf) context to (B, C·T, H, W)."""
        B, C, T, Hf, Wf = ctx.shape
        x = ctx.permute(0, 2, 1, 3, 4).reshape(B * T, C, Hf, Wf)
        x = F.adaptive_avg_pool2d(x, (H, W))
        return x.reshape(B, C * T, H, W)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        curr        : Dict[int, torch.Tensor],   # {i: (B, C_i, H_i, W_i)}
        x_tau       : Dict[int, torch.Tensor],   # {i: (B, C_i, H_i, W_i)}
        t           : torch.Tensor,              # (B,)
        scalar_conds: Optional[torch.Tensor] = None,
        context     : Optional[torch.Tensor] = None,
    ) -> Dict[int, torch.Tensor]:               # {i: (B, C_i, H_i, W_i)} velocity
        """
        Returns velocity dict at every wavelet scale.
        The spatial size of vel[i] matches the spatial size of curr[i].
        """
        cond = self._embed_cond(t, scalar_conds)

        # ── 1. Embed all scales via stems ─────────────────────────────────────
        stem_feats: Dict[int, torch.Tensor] = {}
        for i in range(self.n_scales):
            parts = [curr[i], x_tau[i]]
            if i == 0 and self.has_context and context is not None:
                H0, W0 = curr[0].shape[-2:]
                parts.append(self._pool_context(context, H0, W0))
            stem_feats[i] = self.stems[i](torch.cat(parts, dim=1))

        # ── 2. Encoder ────────────────────────────────────────────────────────
        x      = stem_feats[0]
        skips  : List[torch.Tensor] = []

        for i, (blks, down, inject) in enumerate(
            zip(self.enc_blocks, self.enc_downs, self.injects)
        ):
            for blk in blks:
                x = blk(x, cond)
            skips.append(x)                                    # save skip

            x = down(x)                                        # stride-2 downsample

            # Align to wavelet size at next scale — only spatial fix needed
            H_w = stem_feats[i + 1].shape[-2]
            W_w = stem_feats[i + 1].shape[-1]
            x = _align_to(x, H_w, W_w)

            x = inject(torch.cat([x, stem_feats[i + 1]], dim=1))

        # ── 3. Bottleneck ─────────────────────────────────────────────────────
        for blk in self.bottleneck:
            x = blk(x, cond)

        vel: Dict[int, torch.Tensor] = {self.n_scales - 1: self.coarse_head(x)}

        # ── 4. Decoder ────────────────────────────────────────────────────────
        for up, blks, head, scale_i in zip(
            self.dec_ups, self.dec_blocks, self.out_heads, self._dec_scales
        ):
            x    = up(x)                                       # bilinear ×2 + conv
            skip = skips[scale_i]
            x    = _align_to(x, skip.shape[-2], skip.shape[-1])  # align to skip

            x = torch.cat([x, skip], dim=1)
            for blk in blks:
                x = blk(x, cond)
            vel[scale_i] = head(x)

        return vel
