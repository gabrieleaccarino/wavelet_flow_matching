"""
wavelet_unet.py  —  context now encoded into the conditioning signal,
not concatenated into the stem.  Replaces avg-pool with a lightweight
cross-attention ContextEncoder.
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
    x = x[..., :H, :W]
    dh = H - x.shape[-2]
    dw = W - x.shape[-1]
    if dh > 0 or dw > 0:
        x = F.pad(x, (0, dw, 0, dh), mode='replicate')
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Conditioning
# ─────────────────────────────────────────────────────────────────────────────

def _sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
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
# Context encoder  (NEW)
# ─────────────────────────────────────────────────────────────────────────────

class ContextEncoder(nn.Module):
    """
    Lightweight spatiotemporal context encoder.

    (B, C, T, H, W)  →  (B, cond_dim)

    Steps
    -----
    1. Spatial mean-pool each frame         →  (B, T, C)
    2. Linear projection                    →  (B, T, cond_dim)
    3. Learned query cross-attends T frames →  (B, 1, cond_dim)
    4. LayerNorm + linear output            →  (B, cond_dim)

    Uses .mean() for spatial pooling — avoids adaptive_avg_pool2d
    which is broken on MPS for non-divisible sizes.
    """

    def __init__(self, context_channels: int, n_context_frames: int, cond_dim: int,
                 n_heads: int = 4):
        super().__init__()
        self.frame_proj = nn.Linear(context_channels, cond_dim)
        # Learned query token that summarises the whole context sequence
        self.query      = nn.Parameter(torch.randn(1, 1, cond_dim) * 0.02)
        self.attn       = nn.MultiheadAttention(cond_dim, num_heads=n_heads,
                                                batch_first=True)
        self.norm       = nn.LayerNorm(cond_dim)
        self.out_proj   = nn.Linear(cond_dim, cond_dim)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        # ctx : (B, C, T, H, W)
        B = ctx.shape[0]
        x = ctx.mean(dim=(-2, -1))              # (B, C, T)  — pure mean, MPS-safe
        x = x.permute(0, 2, 1)                 # (B, T, C)
        x = self.frame_proj(x)                  # (B, T, cond_dim)
        q = self.query.expand(B, -1, -1)        # (B, 1, cond_dim)
        out, _ = self.attn(q, x, x)             # (B, 1, cond_dim)
        out = self.norm(out.squeeze(1))          # (B, cond_dim)
        return self.out_proj(out)               # (B, cond_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_ch),  in_ch)
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_num_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.cond  = nn.Linear(cond_dim, out_ch * 2)
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
    return nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)


def _upsample(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
    )


def _out_head(feat_ch: int, out_ch: int) -> nn.Sequential:
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

    Context conditioning change vs previous version
    ────────────────────────────────────────────────
    Past context frames are NO LONGER concatenated into the stem.
    Instead a ContextEncoder (spatial mean-pool + cross-attention)
    encodes them into a (B, cond_dim) vector that is ADDED to the
    time+scalar conditioning signal and broadcast to every ResBlock
    via AdaGN.  This is:
      * architecturally cleaner  — stem input size is always 2·wav_ch[i]
      * more expressive          — attention > avg-pool
      * MPS-safe                 — no adaptive_avg_pool2d
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
        self.chs = [min(base_ch * (2 ** i), base_ch * 8) for i in range(n_scales)]
        self.wav_ch = {i: in_channels * scale_shapes[i]['n_coeff']
                       for i in range(n_scales)}

        # ── Context encoder  (NEW) ────────────────────────────────────────────
        self.has_context = context_channels > 0 and n_context_frames > 0
        self.context_encoder = (
            ContextEncoder(context_channels, n_context_frames, cond_dim)
            if self.has_context else None
        )

        # ── Condition MLP ─────────────────────────────────────────────────────
        self._cond_dim  = cond_dim
        self.time_embed = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 2), nn.SiLU(),
            nn.Linear(cond_dim * 2, cond_dim),
        )
        self.scalar_proj = (
            nn.Linear(num_scalar_conds, cond_dim) if num_scalar_conds > 0 else None
        )

        # ── Stems  (CHANGED: no ctx_ch — always 2·wav_ch[i]) ─────────────────
        print(f'[STEM BUILD] context_channels={context_channels}, n_context_frames={n_context_frames}, ctx_ch={context_channels*n_context_frames}')
        self.stems = nn.ModuleList()
        for i in range(n_scales):
            in_c = 2 * self.wav_ch[i]          # ← no context channels here
            self.stems.append(nn.Conv2d(in_c, self.chs[i], 3, padding=1, bias=False))

        print(f"[WaveletFlowUNet] stem[0].in_channels = {self.stems[0].in_channels}  "
              f"(expected {2 * self.wav_ch[0]})")

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc_blocks = nn.ModuleList([
            nn.ModuleList([ResBlock(self.chs[i], self.chs[i], cond_dim)
                           for _ in range(n_res)])
            for i in range(n_scales - 1)
        ])
        self.enc_downs = nn.ModuleList([
            _downsample(self.chs[i], self.chs[i + 1])
            for i in range(n_scales - 1)
        ])
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
        self._dec_scales = list(range(n_scales - 2, -1, -1))
        self.dec_ups     = nn.ModuleList()
        self.dec_blocks  = nn.ModuleList()
        self.out_heads   = nn.ModuleList()

        for i in self._dec_scales:
            self.dec_ups.append(_upsample(self.chs[i + 1], self.chs[i]))
            self.dec_blocks.append(
                nn.ModuleList(
                    [ResBlock(self.chs[i] * 2, self.chs[i], cond_dim)]
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
        context     : Optional[torch.Tensor],   # (NEW arg)
    ) -> torch.Tensor:
        """
        Compose conditioning vector from:
          1. sinusoidal time embedding
          2. optional scalar conditions
          3. optional context embedding from ContextEncoder  (NEW)
        All three are summed → (B, cond_dim).
        """
        emb = self.time_embed(_sinusoidal(t, self._cond_dim))

        if self.scalar_proj is not None and scalar_conds is not None:
            emb = emb + self.scalar_proj(scalar_conds.float())

        if self.context_encoder is not None and context is not None:
            emb = emb + self.context_encoder(context)          # (NEW)

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

        # Context is now folded into cond, not the stem
        cond = self._embed_cond(t, scalar_conds, context)

        # ── 1. Stems  (no context concatenation here anymore) ─────────────────
        stem_feats: Dict[int, torch.Tensor] = {}
        for i in range(self.n_scales):
            stem_feats[i] = self.stems[i](
                torch.cat([curr[i], x_tau[i]], dim=1)   # always 2·wav_ch[i]
            )

        # ── 2. Encoder ────────────────────────────────────────────────────────
        x     = stem_feats[0]
        skips : List[torch.Tensor] = []

        for i, (blks, down, inject) in enumerate(
            zip(self.enc_blocks, self.enc_downs, self.injects)
        ):
            for blk in blks:
                x = blk(x, cond)
            skips.append(x)
            x = down(x)
            x = _align_to(x, stem_feats[i + 1].shape[-2], stem_feats[i + 1].shape[-1])
            x = inject(torch.cat([x, stem_feats[i + 1]], dim=1))

        # ── 3. Bottleneck ─────────────────────────────────────────────────────
        for blk in self.bottleneck:
            x = blk(x, cond)

        vel: Dict[int, torch.Tensor] = {self.n_scales - 1: self.coarse_head(x)}

        # ── 4. Decoder ────────────────────────────────────────────────────────
        for up, blks, head, scale_i in zip(
            self.dec_ups, self.dec_blocks, self.out_heads, self._dec_scales
        ):
            x    = up(x)
            skip = skips[scale_i]
            x    = _align_to(x, skip.shape[-2], skip.shape[-1])
            x    = torch.cat([x, skip], dim=1)
            for blk in blks:
                x = blk(x, cond)
            vel[scale_i] = head(x)

        return vel