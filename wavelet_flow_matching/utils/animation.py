"""Rollout animations (GIF) for the project page and talks.

Only needs numpy, matplotlib and Pillow: no ffmpeg / moviepy. Titles are typeset
with matplotlib's mathtext in Computer Modern (LaTeX look, no TeX install needed;
pass ``usetex=True`` to use a real LaTeX installation). Typical use, see
``notebooks/07_rollout_animations.ipynb``::

    from wavelet_flow_matching.utils.animation import rollout_gif
    rollout_gif({"Ground truth": gt, r"FM$_{pixel}$": fm}, "rollout.gif", fps=10)

where every value is a ``(T, H, W)`` array (numpy or torch) for one field.
"""
from __future__ import annotations

import io
from functools import lru_cache
from typing import Mapping, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from PIL import Image

_RESAMPLE = {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR,
             "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):                       # torch tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _colorize(x: np.ndarray, vmin: float, vmax: float, cmap: str, contrast: float) -> np.ndarray:
    """(H, W) float -> (H, W, 3) uint8, with the same contrast curve as utils.plot.field2rgb."""
    x = (x - vmin) / (vmax - vmin)
    x = 2 * x - 1
    x = np.sign(x) * np.abs(x) ** (1 / contrast)
    x = np.clip((x + 1) / 2, 0.0, 1.0)
    x = np.nan_to_num(x, nan=0.5)
    return (255 * colormaps[cmap](x)[..., :3]).astype(np.uint8)


@lru_cache(maxsize=512)
def render_text(text: str, fontsize: float = 16, dpi: int = 150, usetex: bool = False,
                color: str = "black") -> Image.Image:
    """Typeset ``text`` (mathtext / LaTeX allowed, e.g. r"FM$_{pixel}$") on a white background."""
    rc = {"text.usetex": usetex, "mathtext.fontset": "cm", "font.family": "serif",
          "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
          "axes.unicode_minus": False, "axes.formatter.use_mathtext": True}
    with matplotlib.rc_context(rc):
        fig = plt.figure(figsize=(0.01, 0.01), dpi=dpi)
        fig.text(0, 0, text, fontsize=fontsize, color=color)
        buf = io.BytesIO()
        fig.savefig(buf, dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor="white")
        plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def rollout_frames(
    fields: Mapping[str, object],
    *,
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    quantiles: tuple[float, float] = (0.01, 0.99),
    contrast: float = 1.5,
    transpose: bool = False,
    flip_vertical: bool = True,
    stride: int = 1,
    max_frames: Optional[int] = None,
    panel_width: Optional[int] = None,
    panel_height: Optional[int] = None,
    scale: float = 1.0,
    resample: str = "bicubic",
    pad: int = 10,
    fontsize: float = 16,
    dpi: int = 150,
    usetex: bool = False,
    show_time: bool = True,
    time_offset: int = 1,
    background: str = "white",
) -> list[Image.Image]:
    """Lay out one panel per entry of ``fields`` side by side, one image per time step.

    fields       : {label: (T, H, W) array}; labels may use LaTeX math, e.g. r"FM$_{pixel}$".
                   The first entry (ground truth) sets the colour range unless vmin/vmax are
                   given, so every panel shares one scale.
    transpose    : swap H and W (e.g. Rayleigh-Benard, stored as (x, y)).
    flip_vertical: put row 0 at the bottom, like ``imshow(origin="lower")`` in the paper figures.
    stride / max_frames: keep every ``stride``-th step, at most ``max_frames`` of them.
    panel_width  : width of each panel in pixels (overrides ``scale``).
    panel_height : height of each panel in pixels. With ``panel_width`` too, the field is
                   stretched to exactly that size (like ``imshow(aspect="auto")`` in the paper
                   figures); alone, it scales the panel keeping the aspect ratio.
    fontsize/dpi : title size in points and rendering resolution (fontsize * dpi / 72 = pixels).
    """
    labels = list(fields)
    arrs = [_to_numpy(fields[k]) for k in labels]
    T = min(a.shape[0] for a in arrs)
    steps = list(range(0, T, stride))[:max_frames]

    ref = arrs[0][steps]
    if vmin is None:
        vmin = float(np.nanquantile(ref, quantiles[0]))
    if vmax is None:
        vmax = float(np.nanquantile(ref, quantiles[1]))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    titles = [render_text(lab, fontsize, dpi, usetex) for lab in labels]
    title_h = max(t.height for t in titles)
    rs = _RESAMPLE[resample]

    frames = []
    for t in steps:
        panels = []
        for a in arrs:
            x = a[t]
            if transpose:
                x = x.T
            if flip_vertical:
                x = x[::-1]
            img = Image.fromarray(_colorize(x, vmin, vmax, cmap, contrast))
            if panel_width and panel_height:
                size = (panel_width, panel_height)
            else:
                s = (panel_width / img.width if panel_width else
                     panel_height / img.height if panel_height else scale)
                size = (max(1, round(img.width * s)), max(1, round(img.height * s)))
            if size != img.size:
                img = img.resize(size, rs)
            panels.append(img)

        pw, ph = panels[0].size
        time_img = render_text(rf"$t = {t + time_offset}$", fontsize, dpi, usetex) if show_time else None
        W = len(panels) * (pw + pad) + pad
        H = pad + title_h + pad + ph + pad + ((time_img.height + pad) if show_time else 0)
        canvas = Image.new("RGB", (W, H), background)
        for j, (title, img) in enumerate(zip(titles, panels)):
            x0 = pad + j * (pw + pad)
            canvas.paste(title, (x0 + (pw - title.width) // 2, pad + (title_h - title.height)))
            canvas.paste(img, (x0, pad + title_h + pad))
        if show_time:
            canvas.paste(time_img, ((W - time_img.width) // 2, H - pad - time_img.height))
        frames.append(canvas)
    return frames


def save_gif(frames: Sequence[Image.Image], file, fps: float = 8.0,
             loop: int = 0, colors: int = 256, hold_last: int = 0) -> str:
    """Write PIL frames to an animated GIF with one shared palette (smaller, no flicker)."""
    # Build the palette from a mosaic of a few frames spread over the rollout
    picks = [frames[i] for i in np.linspace(0, len(frames) - 1, min(8, len(frames))).astype(int)]
    w, h = picks[0].size
    mosaic = Image.new("RGB", (w, h * len(picks)))
    for i, f in enumerate(picks):
        mosaic.paste(f, (0, i * h))
    palette = mosaic.quantize(colors=colors, method=Image.MEDIANCUT)
    pal = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    pal += [pal[-1]] * hold_last
    pal[0].save(file, save_all=True, append_images=pal[1:], duration=int(round(1000 / fps)),
                loop=loop, optimize=True, disposal=1)
    return str(file)


def rollout_gif(fields: Mapping[str, object], file, fps: float = 8.0,
                hold_last: int = 0, **kwargs) -> str:
    """``rollout_frames`` + ``save_gif`` in one call. Returns the file path."""
    return save_gif(rollout_frames(fields, **kwargs), file, fps=fps, hold_last=hold_last)


def preview(file, n: int = 3, width: int = 1200) -> Image.Image:
    """A small static strip (first / middle / last frame) of a GIF, for notebook display
    without embedding the whole animation in the .ipynb."""
    im = Image.open(file)
    idx = np.linspace(0, im.n_frames - 1, n).astype(int)
    shots = []
    for i in idx:
        im.seek(int(i))
        shots.append(im.convert("RGB"))
    w, h = shots[0].size
    strip = Image.new("RGB", (w, h * n), "white")
    for k, s in enumerate(shots):
        strip.paste(s, (0, k * h))
    if strip.width > width:
        strip = strip.resize((width, round(strip.height * width / strip.width)), Image.LANCZOS)
    return strip
