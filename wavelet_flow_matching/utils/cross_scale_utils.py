"""
cross_scale_utils.py
====================

Utilities for demonstrating that Wavelet Flow Matching defines a single joint
flow on the concatenated wavelet vector, not J independent per-scale flows.

Three groups of tools:

1. Representation audit
   ``check_representation``   — is the DWT orthonormal / non-redundant?

2. Mechanism: does each velocity head read the other scales?
   ``build_probe_batch``      — assemble real inputs, no rollout needed
   ``sensitivity_matrix``     — J x J relative sensitivity at one tau
   ``sensitivity_sweep``      — the same across several tau, aggregated
   ``plot_sensitivity``       — heatmap figure for the rebuttal

3. Outcome: does the sampled field carry cross-scale structure?
   ``wavelet_stats``          — per-scale detail kurtosis + parent-child
                                magnitude correlation
   ``plot_wavelet_stats``     — truth vs. model comparison figure

Everything here is inference-only and runs comfortably on a laptop.

Conventions
-----------
Scale index 0 is the FINEST scale (largest spatial grid), matching
``WaveletFlowModule.scale_shapes`` and the UNet stem ordering.

Subband ordering within a scale is assumed to be (LL, LH, HL, HH) when
``n_coeff == 4`` and (LH, HL, HH) when ``n_coeff == 3``. Override with the
``detail_idx`` argument if your ``WaveletDecomposer2D`` differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

__all__ = [
    "ProbeBatch",
    "check_representation",
    "build_probe_batch",
    "make_interpolant",
    "velocity_heads",
    "sensitivity_matrix",
    "sensitivity_sweep",
    "plot_sensitivity",
    "wavelet_stats",
    "plot_wavelet_stats",
    "clamp_report",
]


# ═════════════════════════════════════════════════════════════════════════════
# 1. Representation audit
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def check_representation(model, n_trials: int = 3, verbose: bool = True) -> dict:
    """
    Audit the wavelet transform actually used by ``model``.

    Checks three things that the rebuttal's equivalence argument depends on:

      * perfect reconstruction   ||W^-1 W x - x|| / ||x||   ~ 0
      * energy preservation      ||W x|| / ||x||            ~ 1   (Parseval)
      * non-redundancy           #coefficients / #pixels    ~ 1

    If redundancy > 1 the representation is a frame, not an orthonormal basis.
    That does not invalidate the coupling argument, but it *does* invalidate
    the claim that i.i.d. wavelet-space noise equals W(eps) for pixel-space
    eps, so state the proposition accordingly.

    Returns a dict with the measured quantities.
    """
    if not model.use_wavelet:
        raise ValueError("model was trained in data space (use_wavelet=False).")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    C, T = model.in_channels, model.n_steps_output
    H, W = model.data_shape

    recon_err, energy_ratio = [], []
    for _ in range(n_trials):
        x = torch.randn(1, C, T, H, W, device=device, dtype=dtype)
        coeffs = model.wavelet_decomposer.forward_transform(x)
        x_rec = model.wavelet_decomposer.inverse_transform(coeffs)

        recon_err.append(
            ((x_rec - x).float().norm() / x.float().norm()).item()
        )
        e_coeff = sum(c.float().pow(2).sum() for c in coeffs.values())
        energy_ratio.append((e_coeff / x.float().pow(2).sum()).sqrt().item())

    n_coeff_total = sum(
        s["n_coeff"] * s["H"] * s["W"] for s in model.scale_shapes.values()
    )
    redundancy = n_coeff_total / float(H * W)

    out = {
        "reconstruction_error": float(np.mean(recon_err)),
        "energy_ratio": float(np.mean(energy_ratio)),
        "redundancy": redundancy,
        "n_coeff_total": n_coeff_total,
        "n_pixels": H * W,
        "orthonormal": abs(float(np.mean(energy_ratio)) - 1.0) < 1e-3
        and abs(redundancy - 1.0) < 1e-6,
        "scale_shapes": dict(model.scale_shapes),
    }

    if verbose:
        print("Representation audit")
        print("-" * 52)
        for i, s in model.scale_shapes.items():
            print(
                f"  scale {i}: {s['n_coeff']} subbands x "
                f"{s['H']:>4d} x {s['W']:>4d}"
            )
        print(f"  coefficients / pixels : {redundancy:.4f}")
        print(f"  ||W x|| / ||x||       : {out['energy_ratio']:.6f}")
        print(f"  reconstruction error  : {out['reconstruction_error']:.2e}")
        verdict = (
            "ORTHONORMAL - equivalence-to-pixel-space proposition applies "
            "verbatim."
            if out["orthonormal"]
            else "REDUNDANT / non-isometric - state the proposition for the "
            "non-redundant case and note the deviation."
        )
        print(f"  verdict               : {verdict}")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 2. Mechanism: cross-scale sensitivity of the velocity heads
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ProbeBatch:
    """Inputs to ``model.net`` for a batch of real time windows."""

    c0: Dict[int, torch.Tensor]        # current-state coefficients
    c1: Dict[int, torch.Tensor]        # target-state coefficients
    past: torch.Tensor                 # (B, C, Ti-1, H, W)
    scalars: Optional[torch.Tensor]    # (B, n_cond) or None
    B: int
    n_scales: int
    device: torch.device
    dtype: torch.dtype

    def summary(self) -> str:
        lines = [f"ProbeBatch: B={self.B}, J={self.n_scales}"]
        for i in range(self.n_scales):
            lines.append(f"  scale {i}: {tuple(self.c1[i].shape)}")
        return "\n".join(lines)


def build_probe_batch(
    model,
    x_full: torch.Tensor,
    label: Optional[torch.Tensor] = None,
    n_windows: int = 8,
    n_steps_input: int = 4,
) -> ProbeBatch:
    """
    Assemble a batch of ``n_windows`` time windows drawn from one trajectory.

    Parameters
    ----------
    x_full : (T, H, W, C) tensor, physical units, as returned by
             ``get_well_inputs``.
    label  : (n_cond,) scalar conditions for that trajectory, or None.

    No rollout and no ODE integration is performed; this only prepares the
    arguments of a single ``model.net`` forward pass.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    T_out = model.n_steps_output

    x_full = x_full.to(device=device, dtype=dtype)
    T_avail = x_full.shape[0]
    span = n_steps_input + T_out
    if T_avail < span + 1:
        raise ValueError(
            f"trajectory has {T_avail} frames, need at least {span + 1}"
        )

    starts = torch.linspace(0, T_avail - span - 1, n_windows).long()
    win = torch.stack([x_full[s : s + n_steps_input] for s in starts])
    tgt = torch.stack(
        [x_full[s + n_steps_input : s + span] for s in starts]
    )

    win_n = model.normalizer.normalize_flattened(win, mode="variable")
    tgt_n = model.normalizer.normalize_flattened(tgt, mode="variable")

    curr = rearrange(win_n[:, -1:], "B 1 H W C -> B C 1 H W")
    past = rearrange(win_n[:, :-1], "B T H W C -> B C T H W")
    x1 = rearrange(tgt_n, "B T H W C -> B C T H W")

    scalars = None
    if label is not None:
        scalars = (
            label.to(device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(n_windows, -1)
            .contiguous()
        )

    return ProbeBatch(
        c0=model._to_coeff_dict_x0(curr),
        c1=model._to_coeff_dict(x1),
        past=past,
        scalars=scalars,
        B=n_windows,
        n_scales=model.n_scales,
        device=device,
        dtype=dtype,
    )


def make_interpolant(
    batch: ProbeBatch,
    tau: float,
    generator: Optional[torch.Generator] = None,
) -> Dict[int, torch.Tensor]:
    """OT-CFM interpolant w^tau = tau * w1 + (1 - tau) * eps, per scale."""
    out = {}
    for i in range(batch.n_scales):
        eps = torch.empty_like(batch.c1[i]).normal_(generator=generator)
        out[i] = tau * batch.c1[i] + (1.0 - tau) * eps
    return out


@torch.no_grad()
def velocity_heads(
    model, batch: ProbeBatch, x_tau: Dict[int, torch.Tensor], tau: float
) -> Dict[int, torch.Tensor]:
    """One forward pass through the velocity network."""
    t = torch.full((batch.B,), tau, device=batch.device, dtype=batch.dtype)
    return model.net(
        curr=batch.c0,
        x_tau=x_tau,
        t=t,
        scalar_conds=batch.scalars,
        context=batch.past,
    )


def _rms(z: torch.Tensor) -> float:
    return z.float().pow(2).mean().sqrt().item()


@torch.no_grad()
def sensitivity_matrix(
    model,
    batch: ProbeBatch,
    tau: float,
    delta: float = 1e-2,
    n_probe: int = 4,
    generator: Optional[torch.Generator] = None,
) -> Tuple[np.ndarray, float]:
    """
    Relative cross-scale sensitivity of the velocity heads at flow time ``tau``.

    ``S[j, jp]`` is the relative change in head ``j`` produced by a relative
    perturbation of magnitude ``delta`` applied to the interpolant at scale
    ``jp``:

        S[j, jp] = RMS( u_j(w + delta*v_jp) - u_j(w) )
                   / ( delta * RMS( u_j(w) ) )

    where ``v_jp`` is a random direction rescaled to the RMS of ``w_jp`` so the
    perturbation is dimensionless and comparable across scales.

    Under J genuinely independent flows every off-diagonal entry is exactly
    zero. The returned ``floor`` is the numerical noise level, obtained by
    evaluating the network twice on identical input; it is the null against
    which off-diagonal entries should be judged.

    Returns
    -------
    S     : (J, J) array, rows = affected head, cols = perturbed scale
    floor : float, determinism floor
    """
    J = batch.n_scales
    x_tau = make_interpolant(batch, tau, generator=generator)

    base = velocity_heads(model, batch, x_tau, tau)
    base_rms = {j: _rms(base[j]) for j in range(J)}
    if min(base_rms.values()) <= 0:
        raise RuntimeError("a velocity head returned an all-zero tensor")

    repeat = velocity_heads(model, batch, x_tau, tau)
    floor = max(
        _rms(repeat[j] - base[j]) / (delta * base_rms[j]) for j in range(J)
    )

    S = np.zeros((J, J), dtype=np.float64)
    for jp in range(J):
        acc = np.zeros(J)
        for _ in range(n_probe):
            v = torch.empty_like(x_tau[jp]).normal_(generator=generator)
            v = v / _rms(v) * _rms(x_tau[jp])
            perturbed = dict(x_tau)
            perturbed[jp] = x_tau[jp] + delta * v
            out = velocity_heads(model, batch, perturbed, tau)
            for j in range(J):
                acc[j] += _rms(out[j] - base[j]) / (delta * base_rms[j])
        S[:, jp] = acc / n_probe

    return S, floor


def sensitivity_sweep(
    model,
    batch: ProbeBatch,
    taus: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
    delta: float = 1e-2,
    n_probe: int = 4,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Run ``sensitivity_matrix`` across several flow times and aggregate.

    Returns a dict with per-tau matrices, the tau-averaged matrix, the
    off-diagonal mass fraction, and the determinism floor.
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    # Generators are device-specific; fall back to global seeding on accel.
    if batch.device.type != "cpu":
        torch.manual_seed(seed)
        gen = None

    mats, floors, offs = [], [], []
    for tau in taus:
        S, floor = sensitivity_matrix(
            model, batch, tau, delta=delta, n_probe=n_probe, generator=gen
        )
        off = (S.sum() - np.trace(S)) / S.sum()
        mats.append(S)
        floors.append(floor)
        offs.append(off)
        if verbose:
            print(f"tau = {tau:.2f}   off-diagonal mass = {off:6.1%}")

    S_mean = np.mean(mats, axis=0)
    res = {
        "taus": list(taus),
        "matrices": np.stack(mats),
        "mean": S_mean,
        "off_diagonal_fraction": float(
            (S_mean.sum() - np.trace(S_mean)) / S_mean.sum()
        ),
        "off_diagonal_per_tau": np.array(offs),
        "floor": float(max(floors)),
        "delta": delta,
        "n_probe": n_probe,
    }

    if verbose:
        J = batch.n_scales
        print("\nMean over tau (rows = affected head, cols = perturbed scale)")
        print("          " + "".join(f"   w{jp}   " for jp in range(J)))
        for j in range(J):
            print(
                f"  u_{j}    "
                + "".join(f" {S_mean[j, jp]:7.3f}" for jp in range(J))
            )
        print(f"\noff-diagonal mass  : {res['off_diagonal_fraction']:.1%}")
        print(f"determinism floor  : {res['floor']:.2e}")
        ratio = res["off_diagonal_fraction"] / max(res["floor"], 1e-12)
        print(
            "verdict            : "
            + (
                "heads are coupled across scales "
                f"(off-diagonal exceeds the floor by ~{ratio:.0e}x)"
                if res["off_diagonal_fraction"] > 100 * res["floor"]
                else "inconclusive - off-diagonal is near the numerical floor"
            )
        )
    return res


#   def plot_sensitivity(
#       result: dict,
#       title: str = "Cross-scale coupling of velocity heads",
#       savepath: Optional[str] = None,
#   ):
#       """Heatmap of the tau-averaged sensitivity matrix."""
#       import matplotlib.pyplot as plt
#       from matplotlib.colors import LogNorm
#   
#       S = result["mean"]
#       J = S.shape[0]
#       vmin = max(S[S > 0].min(), 1e-5)
#   
#       fig, ax = plt.subplots(figsize=(4.4, 3.8))
#       im = ax.imshow(S, cmap="magma", norm=LogNorm(vmin=vmin, vmax=S.max()))
#       ax.set_xticks(range(J), [rf"$w^\tau_{{{i}}}$" for i in range(J)])
#       ax.set_yticks(range(J), [rf"$\hat u_{{{i}}}$" for i in range(J)])
#       ax.set_xlabel("perturbed scale (0 = finest)")
#       ax.set_ylabel("affected velocity head")
#       thresh = S.max() / 3.0
#       for j in range(J):
#           for jp in range(J):
#               ax.text(
#                   jp,
#                   j,
#                   f"{S[j, jp]:.3f}",
#                   ha="center",
#                   va="center",
#                   fontsize=9,
#                   color="white" if S[j, jp] < thresh else "black",
#               )
#       fig.colorbar(im, ax=ax, label="relative sensitivity")
#       ax.set_title(title, fontsize=10)
#       fig.tight_layout()
#       if savepath:
#           fig.savefig(savepath, dpi=300, bbox_inches="tight")
#       return fig

def plot_sensitivity(
    result: dict,
    title: str = "Cross-scale coupling of velocity heads",
    savepath: Optional[str] = None,
):
    """Heatmap of the tau-averaged sensitivity matrix."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    S = result["mean"]
    J = S.shape[0]
    vmin = max(S[S > 0].min(), 1e-5)

    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(S, cmap="magma", norm=LogNorm(vmin=vmin, vmax=S.max()))
    ax.set_xticks(range(J), [rf"$w^\tau_{{{i}}}$" for i in range(J)], fontsize=12)
    ax.set_yticks(range(J), [rf"$\hat u_{{{i}}}$" for i in range(J)], fontsize=12)
    ax.set_xlabel("Perturbed scale (0 = finest scale)", fontsize=12)
    ax.set_ylabel("Affected velocity head", fontsize=12)
    thresh = S.max() / 3.0
    for j in range(J):
        for jp in range(J):
            ax.text(
                jp,
                j,
                f"{S[j, jp]:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if S[j, jp] < thresh else "black",
            )
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.99, pad=0.02)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(label="Relative sensitivity", fontsize=12)

    ax.set_title(title, fontsize=14)
#    Change label font size
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig

# ═════════════════════════════════════════════════════════════════════════════
# 3. Outcome: cross-scale structure of the generated fields
# ═════════════════════════════════════════════════════════════════════════════

def _detail_indices(n_coeff: int, detail_idx: Optional[Sequence[int]]) -> List[int]:
    if detail_idx is not None:
        return list(detail_idx)
    if n_coeff == 4:
        return [1, 2, 3]          # LL, LH, HL, HH -> drop the approximation
    if n_coeff == 3:
        return [0, 1, 2]          # LH, HL, HH
    return list(range(n_coeff))


@torch.no_grad()
def wavelet_stats(
    model,
    fields: torch.Tensor,
    normalize: bool = True,
    detail_idx: Optional[Sequence[int]] = None,
    max_frames: int = 64,
) -> dict:
    """
    Cross-scale statistics of a field, computed in the model's own basis.

    Two diagnostics, both invisible to per-scale marginal metrics:

    * ``kurtosis[i]``  -- <d_i^4> / <d_i^2>^2 of the detail coefficients at
      scale i. Gaussian fields give 3; intermittent fields give values that
      grow toward the fine scales.

    * ``parent_child[i]`` -- Pearson correlation between |d_i| pooled onto the
      grid of scale i+1 and |d_{i+1}|, at matched orientation. This is the
      scale-to-scale magnitude dependence that a scale-factorized generator
      cannot produce.

    Parameters
    ----------
    fields : (N, T, H, W, C) or (T, H, W, C) tensor in physical units.
    """
    if fields.ndim == 4:
        fields = fields.unsqueeze(0)
    N, T, H, W, C = fields.shape

    flat = fields.reshape(N * T, 1, H, W, C)[:max_frames]
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    flat = flat.to(device=device, dtype=dtype)

    if normalize:
        flat = model.normalizer.normalize_flattened(
            flat.squeeze(1), mode="variable"
        ).unsqueeze(1)

    x = rearrange(flat, "B T H W C -> B C T H W")
    coeffs = model.wavelet_decomposer.forward_transform(x)
    J = len(coeffs)

    mags = {}
    kurt = {}
    for i in range(J):
        c = coeffs[i].detach().float().cpu()            # (B, C, T, K, H, W)
        idx = _detail_indices(c.shape[3], detail_idx)
        d = c[:, :, :, idx]                             # (B, C, T, 3, H, W)
        m2 = d.pow(2).mean(dim=(-2, -1))
        m4 = d.pow(4).mean(dim=(-2, -1))
        kurt[i] = float((m4 / m2.clamp(min=1e-12).pow(2)).mean())
        mags[i] = d.abs()

    parent_child = {}
    for i in range(J - 1):
        fine = mags[i]
        coarse = mags[i + 1]
        Bd, Cd, Td, Kd = fine.shape[:4]
        Hc, Wc = coarse.shape[-2], coarse.shape[-1]
        pooled = F.interpolate(
            fine.reshape(-1, Kd, fine.shape[-2], fine.shape[-1]),
            size=(Hc, Wc),
            mode="area",
        ).reshape(Bd, Cd, Td, Kd, Hc, Wc)

        rho = []
        for k in range(min(Kd, coarse.shape[3])):
            a = pooled[:, :, :, k].reshape(-1)
            b = coarse[:, :, :, k].reshape(-1)
            a = a - a.mean()
            b = b - b.mean()
            denom = a.norm() * b.norm()
            rho.append(float((a @ b) / denom.clamp(min=1e-12)))
        parent_child[i] = float(np.mean(rho))

    return {
        "kurtosis": kurt,
        "parent_child": parent_child,
        "n_frames": int(flat.shape[0]),
    }


def plot_wavelet_stats(
    stats_by_name: Dict[str, dict],
    savepath: Optional[str] = None,
):
    """
    Two panels comparing several sources (e.g. ground truth, WFM, FM-pixel).

    ``stats_by_name`` maps a label to the dict returned by ``wavelet_stats``.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))

    for name, st in stats_by_name.items():
        ks = sorted(st["kurtosis"])
        axes[0].plot(
            ks,
            [st["kurtosis"][i] for i in ks],
            marker="o",
            label=name,
        )
        pcs = sorted(st["parent_child"])
        if pcs:
            axes[1].plot(
                pcs,
                [st["parent_child"][i] for i in pcs],
                marker="s",
                label=name,
            )

    axes[0].axhline(3.0, ls="--", lw=0.8, color="grey")
    axes[0].text(
        0.02, 0.06, "Gaussian", transform=axes[0].transAxes,
        fontsize=8, color="grey",
    )
    axes[0].set_xlabel("scale (0 = finest)")
    axes[0].set_ylabel(r"kurtosis  $\langle d^4\rangle/\langle d^2\rangle^2$")
    axes[0].set_title("Intermittency of detail coefficients", fontsize=10)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("scale pair  (i, i+1)")
    axes[1].set_ylabel(r"$\mathrm{Corr}(|d_i|, |d_{i+1}|)$")
    axes[1].set_title("Parent-child magnitude coupling", fontsize=10)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Rollout hygiene
# ═════════════════════════════════════════════════════════════════════════════

def clamp_report(preds_norm: torch.Tensor, clamp_sigma: float = 5.0) -> dict:
    """
    How often would ``rollout``'s ``clamp_sigma`` have fired?

    ``rollout`` stores predictions unclamped but feeds the clamped state back
    into the next step. If this fraction is non-zero, the autoregressive
    trajectory was produced by a mechanism that truncates extreme values --
    precisely the values that carry intermittency. Rerun with
    ``clamp_sigma=0.0`` before making any intermittency claim.
    """
    a = preds_norm.detach().float().abs()
    frac = float((a > clamp_sigma).float().mean())
    return {
        "clamp_sigma": clamp_sigma,
        "fraction_clipped": frac,
        "max_abs": float(a.max()),
        "safe": frac == 0.0,
    }
