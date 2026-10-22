r"""
dcae.py  —  self-contained, weight-compatible Deep-Compression Auto-Encoder.

This is a faithful (structure- and parameter-name-preserving) vendoring of the
DC-AE used by LOLA ("Lost in Latent Space", Rozet et al.):

    lola/nn/layers.py   (ConvNd, LayerNorm, SelfAttentionNd, Patchify, Unpatchify)
    lola/nn/dcae.py     (ResBlock, DCEncoder, DCDecoder)
    lola/autoencoder.py (AutoEncoder, get_autoencoder)

    Reference: Deep Compression Autoencoder for Efficient High-Resolution
    Diffusion Models (Chen et al., 2024) — https://arxiv.org/abs/2410.10733

Why vendored?
-------------
The LOLA package imports ``xformers`` at ``import`` time (through
``lola.nn.__init__``), which we do not need for the DC-AE. Vendoring the exact
module hierarchy lets us load the pre-trained ``state.pth`` checkpoints
(``autoencoder.state_dict()``) with ``strict=True`` — the submodule names
(``encoder.descent.*``, ``decoder.ascent.*``) are identical to LOLA's.

Only the pieces required for a 2-D convolutional DC-AE are kept. Attention is
supported (via ``attention_heads``) but the RB autoencoders use ``{}`` so it is
never instantiated.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from typing import Any, Dict, Optional, Sequence, Tuple, Union


# ─────────────────────────────────────────────────────────────────────────────
# Layers  (faithful to lola/nn/layers.py)
# ─────────────────────────────────────────────────────────────────────────────

def ConvNd(
    in_channels: int,
    out_channels: int,
    spatial: int = 2,
    identity_init: bool = False,
    **kwargs,
) -> nn.Module:
    CONVS = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
    if spatial not in CONVS:
        raise NotImplementedError()
    conv = CONVS[spatial](in_channels, out_channels, **kwargs)

    if identity_init:
        kernel_size = conv.weight.shape[2:]
        kernel_center = [k // 2 for k in kernel_size]
        eye = torch.zeros_like(conv.weight.data)
        for i in range(out_channels):
            eye[(i, i % in_channels, *kernel_center)] = 1
        conv.weight.data.mul_(1e-2)
        conv.weight.data.add_(eye)

    return conv


class LayerNorm(nn.Module):
    r"""Standardizes features along a dimension (no affine params)."""

    def __init__(self, dim: Union[int, Sequence[int]], eps: float = 1e-5):
        super().__init__()
        self.dim = dim if isinstance(dim, int) else tuple(dim)
        self.register_buffer("eps", torch.as_tensor(eps))

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def forward(self, x: Tensor) -> Tensor:
        variance, mean = torch.var_mean(x, dim=self.dim, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.eps)


class SelfAttentionNd(nn.MultiheadAttention):
    r"""N-dimensional self-attention (kept for parity; unused when heads={})."""

    def __init__(self, channels: int, heads: int = 1, checkpointing: bool = False, **kwargs):
        super().__init__(embed_dim=channels, num_heads=heads, batch_first=True, **kwargs)
        self.checkpointing = checkpointing

    def _forward(self, x: Tensor) -> Tensor:
        y = rearrange(x, "B C ...  -> B (...) C")
        qkv = torch.nn.functional.linear(y, self.in_proj_weight, self.in_proj_bias)
        q, k, v = rearrange(qkv, "B L (n H C) -> n B H L C", n=3, H=self.num_heads)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        y = rearrange(y, "B H L C -> B L (H C)")
        y = torch.nn.functional.linear(y, self.out_proj.weight, self.out_proj.bias)
        y = rearrange(y, "B L C -> B C L").reshape(x.shape)
        return y

    def forward(self, x: Tensor) -> Tensor:
        if self.checkpointing:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


def Patchify(patch_size: Sequence[int], channel_last: bool = False) -> Rearrange:
    if len(patch_size) == 2:
        h, w = patch_size
        if channel_last:
            return Rearrange("... C (H h) (W w) -> ... H W (C h w)", h=h, w=w)
        return Rearrange("... C (H h) (W w) -> ... (C h w) H W", h=h, w=w)
    if len(patch_size) == 1:
        (l,) = patch_size
        if channel_last:
            return Rearrange("... C (L l) -> ... L (C l)", l=l)
        return Rearrange("... C (L l) -> ... (C l) L", l=l)
    if len(patch_size) == 3:
        l, h, w = patch_size
        if channel_last:
            return Rearrange("... C (L l) (H h) (W w) -> ... L H W (C l h w)", l=l, h=h, w=w)
        return Rearrange("... C (L l) (H h) (W w) -> ... (C l h w) L H W", l=l, h=h, w=w)
    raise NotImplementedError()


def Unpatchify(patch_size: Sequence[int], channel_last: bool = False) -> Rearrange:
    if len(patch_size) == 2:
        h, w = patch_size
        if channel_last:
            return Rearrange("... H W (C h w) -> ... C (H h) (W w)", h=h, w=w)
        return Rearrange("... (C h w) H W -> ... C (H h) (W w)", h=h, w=w)
    if len(patch_size) == 1:
        (l,) = patch_size
        if channel_last:
            return Rearrange("... L (C l) -> ... C (L l)", l=l)
        return Rearrange("... (C l) L -> ... C (L l)", l=l)
    if len(patch_size) == 3:
        l, h, w = patch_size
        if channel_last:
            return Rearrange("... L H W (C l h w) -> ... C (L l) (H h) (W w)", l=l, h=h, w=w)
        return Rearrange("... (C l h w) L H W -> ... C (L l) (H h) (W w)", l=l, h=h, w=w)
    raise NotImplementedError()


# ─────────────────────────────────────────────────────────────────────────────
# DC-AE blocks  (faithful to lola/nn/dcae.py)
# ─────────────────────────────────────────────────────────────────────────────

class Residual(nn.Sequential):
    def forward(self, x: Tensor) -> Tensor:
        return x + super().forward(x)


class ResBlock(nn.Module):
    r"""Residual block with LayerNorm/GroupNorm + FFN (+ optional attention)."""

    def __init__(
        self,
        channels: int,
        norm: str = "layer",
        groups: int = 16,
        attention_heads: Optional[int] = None,
        ffn_factor: int = 1,
        spatial: int = 2,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.checkpointing = checkpointing

        if norm == "layer":
            self.norm = LayerNorm(dim=-spatial - 1)
        elif norm == "group":
            self.norm = nn.GroupNorm(num_groups=min(groups, channels), num_channels=channels, affine=False)
        else:
            raise NotImplementedError()

        if attention_heads is None:
            self.attn = nn.Identity()
        else:
            self.attn = Residual(SelfAttentionNd(channels, heads=attention_heads))
            kwargs.update(kernel_size=1, padding=0)

        self.ffn = nn.Sequential(
            ConvNd(channels, ffn_factor * channels, spatial=spatial, **kwargs),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            ConvNd(ffn_factor * channels, channels, spatial=spatial, **kwargs),
        )
        self.ffn[-1].weight.data.mul_(1e-2)

    def _forward(self, x: Tensor) -> Tensor:
        y = self.norm(x)
        y = self.attn(y)
        y = self.ffn(y)
        return x + y

    def forward(self, x: Tensor) -> Tensor:
        if self.checkpointing:
            return checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class DCEncoder(nn.Module):
    r"""Deep-compressed encoder. Downsamples by ``stride`` at every depth > 0."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        pixel_shuffle: bool = True,
        norm: str = "layer",
        attention_heads: Dict[int, int] = {},  # noqa: B006
        ffn_factor: int = 1,
        spatial: int = 2,
        patch_size: Union[int, Sequence[int]] = 1,
        periodic: bool = False,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        identity_init: bool = True,
    ):
        super().__init__()
        assert len(hid_blocks) == len(hid_channels)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * spatial
        if isinstance(stride, int):
            stride = [stride] * spatial
        if isinstance(patch_size, int):
            patch_size = [patch_size] * spatial

        kwargs = dict(
            kernel_size=tuple(kernel_size),
            padding=tuple(k // 2 for k in kernel_size),
            padding_mode="circular" if periodic else "zeros",
        )

        self.patch = Patchify(patch_size=patch_size)
        self.descent = nn.ModuleList()

        for i, num_blocks in enumerate(hid_blocks):
            blocks = nn.ModuleList()

            if i > 0:
                if pixel_shuffle:
                    blocks.append(
                        nn.Sequential(
                            Patchify(patch_size=stride),
                            ConvNd(
                                hid_channels[i - 1] * math.prod(stride),
                                hid_channels[i],
                                spatial=spatial,
                                identity_init=identity_init,
                                **kwargs,
                            ),
                        )
                    )
                else:
                    blocks.append(
                        ConvNd(
                            hid_channels[i - 1], hid_channels[i], spatial=spatial,
                            stride=stride, identity_init=identity_init, **kwargs,
                        )
                    )
            else:
                blocks.append(
                    ConvNd(math.prod(patch_size) * in_channels, hid_channels[i], spatial=spatial, **kwargs)
                )

            for _ in range(num_blocks):
                blocks.append(
                    ResBlock(
                        hid_channels[i], norm=norm, attention_heads=attention_heads.get(i, None),
                        ffn_factor=ffn_factor, spatial=spatial, dropout=dropout,
                        checkpointing=checkpointing, **kwargs,
                    )
                )

            if i + 1 == len(hid_blocks):
                blocks.append(
                    ConvNd(hid_channels[i], out_channels, spatial=spatial, identity_init=identity_init, **kwargs)
                )

            self.descent.append(blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch(x)
        for blocks in self.descent:
            for block in blocks:
                x = block(x)
        return x


class DCDecoder(nn.Module):
    r"""Deep-compressed decoder. Upsamples by ``stride`` at every depth > 0."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: Union[int, Sequence[int]] = 3,
        stride: Union[int, Sequence[int]] = 2,
        pixel_shuffle: bool = True,
        norm: str = "layer",
        attention_heads: Dict[int, int] = {},  # noqa: B006
        ffn_factor: int = 1,
        spatial: int = 2,
        patch_size: Union[int, Sequence[int]] = 1,
        periodic: bool = False,
        dropout: Optional[float] = None,
        checkpointing: bool = False,
        identity_init: bool = True,
    ):
        super().__init__()
        assert len(hid_blocks) == len(hid_channels)

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * spatial
        if isinstance(stride, int):
            stride = [stride] * spatial
        if isinstance(patch_size, int):
            patch_size = [patch_size] * spatial

        kwargs = dict(
            kernel_size=tuple(kernel_size),
            padding=tuple(k // 2 for k in kernel_size),
            padding_mode="circular" if periodic else "zeros",
        )

        self.unpatch = Unpatchify(patch_size=patch_size)
        self.ascent = nn.ModuleList()

        for i, num_blocks in reversed(list(enumerate(hid_blocks))):
            blocks = nn.ModuleList()

            if i + 1 == len(hid_blocks):
                blocks.append(
                    ConvNd(in_channels, hid_channels[i], spatial=spatial, identity_init=identity_init, **kwargs)
                )

            for _ in range(num_blocks):
                blocks.append(
                    ResBlock(
                        hid_channels[i], norm=norm, attention_heads=attention_heads.get(i, None),
                        ffn_factor=ffn_factor, spatial=spatial, dropout=dropout,
                        checkpointing=checkpointing, **kwargs,
                    )
                )

            if i > 0:
                if pixel_shuffle:
                    blocks.append(
                        nn.Sequential(
                            ConvNd(
                                hid_channels[i], hid_channels[i - 1] * math.prod(stride),
                                spatial=spatial, identity_init=identity_init, **kwargs,
                            ),
                            Unpatchify(patch_size=stride),
                        )
                    )
                else:
                    blocks.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=tuple(stride), mode="nearest"),
                            ConvNd(
                                hid_channels[i], hid_channels[i - 1], spatial=spatial,
                                identity_init=identity_init, **kwargs,
                            ),
                        )
                    )
            else:
                blocks.append(
                    ConvNd(hid_channels[i], math.prod(patch_size) * out_channels, spatial=spatial, **kwargs)
                )

            self.ascent.append(blocks)

    def forward(self, x: Tensor) -> Tensor:
        for blocks in self.ascent:
            for block in blocks:
                x = block(x)
        x = self.unpatch(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# AutoEncoder wrapper  (faithful to lola/autoencoder.py)
# ─────────────────────────────────────────────────────────────────────────────

class AutoEncoder(nn.Module):
    r"""Encoder + decoder with a deterministic saturating latent.

    ``saturate`` structures the latent space (mimicking a standard Gaussian
    of range ``B=5``) in place of a KL penalty — no post-hoc latent
    standardization is needed. ``softclip2`` is the default used by the RB
    autoencoders.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 saturation: str = "softclip2", noise: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.saturation = saturation
        self.noise = noise

    def saturate(self, x: Tensor) -> Tensor:
        if self.saturation is None:
            return x
        if self.saturation == "softclip":
            return x / (1 + abs(x) / 5)
        if self.saturation == "softclip2":
            return x * torch.rsqrt(1 + torch.square(x / 5))
        if self.saturation == "tanh":
            return torch.tanh(x / 5) * 5
        if self.saturation == "arcsinh":
            return torch.arcsinh(x)
        if self.saturation == "rmsnorm":
            return x * torch.rsqrt(torch.mean(torch.square(x), dim=1, keepdim=True) + 1e-5)
        raise ValueError(f"unknown saturation '{self.saturation}'")

    def encode(self, x: Tensor) -> Tensor:
        return self.saturate(self.encoder(x))

    def decode(self, z: Tensor, noisy: bool = True) -> Tensor:
        if noisy and self.noise > 0:
            z = z + self.noise * torch.randn_like(z)
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encode(x)
        y = self.decode(z)
        return y, z


def get_autoencoder(
    pix_channels: int,
    lat_channels: int,
    spatial: int = 2,
    arch: Optional[str] = None,
    saturation: str = "softclip2",
    encoder_only: Dict[str, Any] = {},  # noqa: B006
    decoder_only: Dict[str, Any] = {},  # noqa: B006
    latent_noise: float = 0.0,
    # Ignored keys present in LOLA's config.yaml (name, loss, ...)
    name: Optional[str] = None,
    loss: Any = None,
    **kwargs,
) -> AutoEncoder:
    r"""Instantiates a DC-AE from a LOLA ``config.yaml`` ``ae`` block.

    Only ``arch in {None, 'dcae'}`` is supported here (the RB autoencoders).
    ``vit`` autoencoders are intentionally out of scope for this baseline.
    """
    if arch not in (None, "dcae"):
        raise NotImplementedError(
            f"vendored get_autoencoder only supports the DC-AE (arch='dcae'); got arch={arch!r}"
        )

    encoder = DCEncoder(
        in_channels=pix_channels, out_channels=lat_channels, spatial=spatial,
        **encoder_only, **kwargs,
    )
    decoder = DCDecoder(
        in_channels=lat_channels, out_channels=pix_channels, spatial=spatial,
        **decoder_only, **kwargs,
    )
    return AutoEncoder(encoder, decoder, saturation=saturation, noise=latent_noise)


def compression_factor(hid_blocks: Sequence[int], stride: int = 2,
                       patch_size: int = 1, spatial: int = 2) -> int:
    r"""Spatial downsampling factor r = patch_size * stride^(len(hid_blocks)-1)."""
    return int(patch_size * stride ** (len(hid_blocks) - 1)) ** 1
