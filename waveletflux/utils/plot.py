"""
Plotting and image helpers. From LOLA (TODO properly acknowledge)
"""

import math
import matplotlib.animation as ani
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import pandas as pd
import torch

from IPython.display import Video
from moviepy import ImageSequenceClip
from numpy.typing import ArrayLike
from PIL import Image, ImageDraw, ImageFont
from tempfile import mkstemp
from typing import Optional, Sequence, Tuple, Union

from waveletflux.utils.fourier import isotropic_power_spectrum






def plot_psd(
    x: ArrayLike,
    y: Optional[ArrayLike] = None,
    fields: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (3.2, 3.2),
) -> plt.Figure:
    C, *shape = x.shape

    if torch.is_tensor(x):
        x = x.numpy(force=True)

    if torch.is_tensor(y):
        y = y.numpy(force=True)

    fig, axs = plt.subplots(
        nrows=1,
        ncols=C,
        figsize=(figsize[0] * C, figsize[1]),
        squeeze=False,
    )

    for i in range(C):
        if fields:
            axs[0, i].set_title(f"{fields[i]}")

        p_x, k = isotropic_power_spectrum(x[i], spatial=len(shape))

        axs[0, i].loglog(1 / k, p_x, base=2, label="$x$", color="black")

        if y is not None:
            p_y, _ = isotropic_power_spectrum(y[i], spatial=len(shape))

            axs[0, i].loglog(1 / k, p_y, base=2, label="$y$", color="C0", alpha=0.75)

        axs[0, i].invert_xaxis()
        axs[0, i].set_xticks([2**i for i in range(1, math.ceil(math.log2(1 / k[0].item())))])

    axs[0, 0].set_ylabel("power spectrum density")
    axs[0, 0].legend()

    fig.align_labels()
    fig.tight_layout()

    return fig


def field2rgb(
    x: ArrayLike,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    contrast: float = 1.5,
    cmap: str = "RdBu_r",
    bad: str = "whitesmoke",
) -> ArrayLike:
    if torch.is_tensor(x):
        x = x.numpy(force=True)

    if vmin is None:
        vmin = np.nanquantile(x, 0.01) - 1e-2
    if vmax is None:
        vmax = np.nanquantile(x, 0.99) + 1e-2

    palette = sb.color_palette(cmap, as_cmap=True)
    palette.set_bad(bad)

    x = (x - vmin) / (vmax - vmin)
    x = 2 * x - 1
    x = np.sign(x) * np.abs(x) ** (1 / contrast)
    x = (x + 1) / 2
    x = np.clip(x, a_min=0.0, a_max=1.0)
    x = palette(x)
    x = 255 * x[..., :3]
    x = x.astype(np.uint8)

    return x


def draw(
    x: ArrayLike,  # (M, N, H, W)
    pad: Union[float, int] = 1 / 64,
    background: str = "white",
    isolate: Sequence[int] = (),
    zoom: int = 1,
    titles: Sequence[str] = (),
    fontsize: int = 32,
    **kwargs,
) -> Image.Image:
    if torch.is_tensor(x):
        x = x.numpy(force=True)

    axes = tuple(i for i in range(x.ndim) if i not in isolate)

    kwargs.setdefault("vmin", np.nanquantile(x, 0.01, axis=axes, keepdims=True) - 1e-2)
    kwargs.setdefault("vmax", np.nanquantile(x, 0.99, axis=axes, keepdims=True) + 1e-2)

    x = field2rgb(x, **kwargs)

    while x.ndim < 5:
        x = x[None]

    M, N, H, W, _ = x.shape

    if isinstance(pad, float):
        pad = int(pad * max(H, W))

    img = Image.new(
        mode="RGB",
        size=(
            N * (W + pad) + pad,
            M * (H + pad) + pad,
        ),
        color=background,
    )

    for i in range(M):
        for j in range(N):
            offset = (
                j * (W + pad) + pad,
                i * (H + pad) + pad,
            )

            img.paste(Image.fromarray(x[i][j]), offset)

    if zoom > 1:
        img = img.resize((img.width * zoom, img.height * zoom), Image.NEAREST)
        pad, H, W = pad * zoom, H * zoom, W * zoom

    if titles:
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default(size=fontsize)

        _, _, w, h = draw.textbbox((0, 0), text="".join(titles), font=font)

        new = Image.new(mode="RGB", size=(img.width, img.height + h + pad), color=background)
        new.paste(img, (0, h + pad))

        img = new
        draw = ImageDraw.Draw(img)

        for j, title in enumerate(titles):
            offset = (
                j * (W + pad) + pad + W // 2,
                h + pad,
            )

            draw.text(offset, text=title, anchor="md", fill="black", font=font)

    return img


def draw_movie(
    x: ArrayLike,  # (T, M, N, H, W)
    file: Optional[str] = None,
    fps: float = 4.0,
    display: bool = False,
    embed: bool = False,
    isolate: Sequence[int] = (),
    **kwargs,
) -> Union[str, Video]:
    if torch.is_tensor(x):
        x = x.numpy(force=True)

    axes = tuple(i for i in range(x.ndim) if i not in isolate)

    kwargs.setdefault("vmin", np.nanquantile(x, 0.01, axis=axes, keepdims=True).squeeze(0) - 1e-2)
    kwargs.setdefault("vmax", np.nanquantile(x, 0.99, axis=axes, keepdims=True).squeeze(0) + 1e-2)

    imgs = [draw(xi, **kwargs) for i, xi in enumerate(x)]
    imgs = [np.asarray(img) for img in imgs]

    clip = ImageSequenceClip(imgs, fps=fps)

    if file is None:
        _, file = mkstemp(suffix=".mp4")

    if str(file).endswith(".gif"):
        clip.write_gif(file, loop=0, logger=None)
    else:
        clip.write_videofile(file, codec="libx264", logger=None)

    if display:
        return Video(file, embed=embed, width=1280)
    else:
        return file


