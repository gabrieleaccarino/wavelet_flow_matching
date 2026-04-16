"""
Fourier-domain metrics for WaveletFlow evaluation.

Changelog vs. the original LOLA-derived code
---------------------------------------------
FIX 1  Spread: Bessel correction n/(n-1) instead of (n+1)/(n-1).
FIX 2  CRPS:   Fair CRPS uses 1/(n(n-1)) not 1/n²  (Ferro 2014).
FIX 3  Band edges consistent between compute_stats and build_psd_df.
FIX 4  build_psd_df uses k[0] from the full grid for band edges (not
       k[_valid][0] after applying the min-bin-count mask).

NEW    Log-spectral distance  (lsd_p_*, lsd_c_*) alongside the original
       rmse_p_* / rmse_c_* metrics.

WHY rmse_p BLOWS UP FOR NEAR-ZERO-MEAN FIELDS
----------------------------------------------
The original formula:

    se_p = ( 1 − (p_v + ε) / (p_u + ε) )²        ε = 1e-6

is unbounded.  Two failure modes appear in the active-matter stats:

  Mode A — ε-dominated denominator.  When p_u ≪ ε (velocity, E tensor,
  D_xy at low-power timesteps), (p_u + ε) ≈ ε while p_v can be orders
  of magnitude larger.  Result: se_p ≈ (p_v / ε)² ~ 10⁸.

  Mode B — ratio explosion at large power.  When the model concentrates
  spectral energy in the wrong band (p_v ≫ p_u, both above ε), the
  ratio p_v / p_u can reach 10⁴, giving se_p ≈ 10⁸.

The log-spectral distance (LSD):

    lsd_p = ( log(p_v + ε_log) − log(p_u + ε_log) )²

is bounded by (log(p_v / p_u))² and is the standard in audio / speech
signal-processing literature (Itakura–Saito family).  ε_log is chosen
adaptively:  ε_log = ε_log_scale × max(p_u), clamped to a tiny absolute
floor.  This keeps the additive stabiliser sub-dominant across the 6-
decade range of field powers in active-matter (velocity m2 ~ 3e-7 to
concentration m2 ~ 1.0).

Note on rmse_c: the coherence denominator is sqrt(p_u·p_v), so by
Cauchy–Schwarz |S_xy|² ≤ S_xx S_yy the ratio c_uv / sqrt(p_u·p_v)
stays in [0, 1].  This makes se_c bounded in [0, 4] and robust — it
does not need the fix that rmse_p does.  The lsd_c metric is provided
for completeness.

LOLA NORMALISATION NOTE
-----------------------
LOLA (and several follow-up works) compute PSD metrics in the per-
channel normalised space:

    u_norm = (u − μ_field) / σ_field          (dataset-level statistics)

After normalisation every field has unit variance, so p_u bins are
O(1 / n_bins) ≈ 0.02, well above ε = 1e-6.  This eliminates Mode A
for the original rmse_p formula and makes cross-field comparison fair.
Mode B (wrong spectral distribution) still causes large rmse_p values.

To replicate LOLA's metrics, pass `norm_mean` and `norm_std` (both
shape (C,)) to `build_stats_df`.  This adds a second family of metrics
with the suffix `_n` in the column names:
    rmse_p_low_n, lsd_p_low_n, rmse_c_low_n, lsd_c_low_n, …

Without those arrays the _n columns are omitted.  The lsd_p family
without normalisation is already robust, so normalisation matters most
if you need exact comparability to LOLA-reported numbers.
"""

from __future__ import annotations

import ot
import pandas as pd
import torch

from waveletflux.utils.fourier import (
    isotropic_binning,
    isotropic_cross_correlation,
    isotropic_power_spectrum,
)

# ── Constants ─────────────────────────────────────────────────────────────────

_BAND_NAMES = ["low", "mid", "high", "vhigh"]


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _band_edges(k: torch.Tensor) -> torch.Tensor:
    """Four log-spaced edges from k[0] (lowest non-DC bin) to Nyquist."""
    return torch.logspace(k[0].log2().item(), -1.0, steps=4, base=2.0)


def _adaptive_eps_log(p_u: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Data-adaptive floor for log-spectral distance.

    ε_log = scale × max(p_u),  clamped to [1e-30, ∞)

    Keeps ε_log sub-dominant even when max(p_u) ~ 1e-7 (velocity / E
    tensors), while guaranteeing a positive argument to torch.log.

    Returns a scalar tensor on the same device as p_u.
    """
    return (scale * p_u.amax()).clamp(min=1e-30)


def _spectral_band_metrics(
    p_u:      torch.Tensor,   # (B,)  truth PSD
    p_v_mean: torch.Tensor,   # (B,)  predicted PSD (ensemble mean)
    c_uv_mean:torch.Tensor,   # (B,)  cross-spectrum magnitude (ensemble mean)
    k:        torch.Tensor,   # (B,)  wavenumber bins
    eps:      float,          # absolute stabiliser for original formula
    eps_log:  torch.Tensor,   # adaptive stabiliser for log formula (scalar)
    suffix:   str = "",       # e.g. "_n" for normalised-space metrics
) -> dict:
    """
    Compute rmse_p, lsd_p, rmse_c, lsd_c for each of the four frequency bands.

    The `suffix` parameter appends a string to all column names, allowing
    normalised-space and raw-space metrics to coexist in the same row.
    """
    # ── Original relative-error metrics ───────────────────────────────────────
    # rmse_p can blow up (see module docstring).
    # rmse_c is bounded in [0, 2] by Cauchy–Schwarz → robust.
    se_p = torch.square(1.0 - (p_v_mean + eps) / (p_u + eps))
    se_c = torch.square(
        1.0 - (c_uv_mean + eps) / torch.sqrt(p_u * p_v_mean + eps ** 2)
    )

    # ── Log-spectral distance (robust) ────────────────────────────────────────
    lsd_p = torch.square(
        torch.log(p_v_mean + eps_log) - torch.log(p_u + eps_log)
    )
    # Coherence log version: log(|S_xy|) vs log(sqrt(S_xx · S_yy))
    denom_c = torch.sqrt(p_u * p_v_mean + eps_log ** 2) + eps_log
    lsd_c = torch.square(
        torch.log(c_uv_mean + eps_log) - torch.log(denom_c)
    )

    edges = _band_edges(k)
    out   = {}

    for i, band in enumerate(_BAND_NAMES):
        mask = (
            (edges[i] <= k) & (k <= edges[i + 1])
            if i < len(_BAND_NAMES) - 1
            else (edges[i] <= k)
        )
        if mask.any():
            out[f"rmse_p_{band}{suffix}"] = torch.sqrt(torch.mean(se_p[mask])).item()
            out[f"lsd_p_{band}{suffix}"]  = torch.sqrt(torch.mean(lsd_p[mask])).item()
            out[f"rmse_c_{band}{suffix}"] = torch.sqrt(torch.mean(se_c[mask])).item()
            out[f"lsd_c_{band}{suffix}"]  = torch.sqrt(torch.mean(lsd_c[mask])).item()
        else:
            for prefix in ("rmse_p", "lsd_p", "rmse_c", "lsd_c"):
                out[f"{prefix}_{band}{suffix}"] = float("nan")

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Core metric computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats(
    u:             torch.Tensor,    # ground-truth single frame  (H, W)
    v:             torch.Tensor,    # predicted frames           (N_ENS, H, W)
    spatial:       int   = 2,
    eps:           float = 1e-6,    # stabiliser for original rmse_p / rmse_c
    eps_log_scale: float = 1e-8,    # adaptive floor scale for lsd_p / lsd_c
    compute_emd:   bool  = False,
    # Optional LOLA-style normalisation for a second PSD metric family.
    # Pass scalar tensors (or Python floats) representing per-field
    # dataset statistics.  If both are provided, rmse_p_*_n / lsd_p_*_n
    # columns are added computed on (u - norm_mean) / norm_std.
    norm_mean: float | torch.Tensor | None = None,
    norm_std:  float | torch.Tensor | None = None,
) -> dict:
    """
    Compute all metrics for a single (field, timestep) pair.

    Args
    ----
    u            : Ground-truth frame          (H, W)
    v            : Ensemble/predicted frames   (N_ENS, H, W)
    spatial      : Number of spatial dims.
    eps          : Absolute stabiliser for the original rmse_p / rmse_c
                   formulas (unchanged from the original implementation).
    eps_log_scale: Relative scale for the adaptive log stabiliser:
                       ε_log = eps_log_scale × max(p_u)
                   A value of 1e-8 keeps ε_log ≲ 1 % of the minimum
                   significant PSD bin across all active-matter fields.
    compute_emd  : Whether to compute the expensive 2-D Sliced-EMD.
    norm_mean    : Dataset-level field mean for LOLA-style normalisation.
    norm_std     : Dataset-level field std  for LOLA-style normalisation.

    Returns
    -------
    dict of scalar metric values.  All keys:

    Scalar statistics
        m1, m2, spread, crps, mean_ens_std, ens_std_max, ens_std_min,
        std_err_corr, coverage_68, coverage_95, rmse, nrmse, vrmse,
        spread_skill, total_error, w1, emd

    Per-band spectral metrics (raw space)
        rmse_p_{low,mid,high,vhigh}   — original formula (can blow up)
        lsd_p_{low,mid,high,vhigh}    — log-spectral distance (robust)
        rmse_c_{low,mid,high,vhigh}   — original coherence (robust by C-S)
        lsd_c_{low,mid,high,vhigh}    — log coherence (for completeness)

    Per-band spectral metrics (normalised space, only if norm_mean/norm_std
    are supplied)
        rmse_p_{low,mid,high,vhigh}_n
        lsd_p_{low,mid,high,vhigh}_n
        rmse_c_{low,mid,high,vhigh}_n
        lsd_c_{low,mid,high,vhigh}_n
    """
    samples = v.shape[0]

    # ── Moments ───────────────────────────────────────────────────────────────
    m1 = torch.mean(u).item()
    m2 = torch.mean(u ** 2).item()

    # ── Ensemble mean ─────────────────────────────────────────────────────────
    v_mean = torch.mean(v, dim=0)   # (H, W)

    # ── Spread (calibration) ──────────────────────────────────────────────────
    # FIX 1: standard Bessel correction n/(n-1).
    if samples > 1:
        spread_sq = torch.mean(torch.square(v - v_mean))
        spread = torch.sqrt(samples / (samples - 1) * spread_sq).item()
    else:
        spread = float("nan")

    # ── CRPS ──────────────────────────────────────────────────────────────────
    # FIX 2: Fair CRPS 1/(n(n-1)) weight  (Ferro 2014).
    abs_err = torch.abs(v - u).mean()   # E|X - y|

    if samples > 1:
        v_sorted  = torch.sort(v.reshape(samples, -1), dim=0).values   # (N_ENS, H*W)
        n         = samples
        weights   = (2 * torch.arange(1, n + 1, device=v.device) - n - 1).float() / (n * (n - 1))
        fair_term = (weights[:, None] * v_sorted).sum(dim=0).mean()
        crps      = (abs_err - fair_term).item()
    else:
        crps = abs_err.item()

    # ── Ensemble diagnostics ──────────────────────────────────────────────────
    if samples > 1:
        ens_std      = v.std(dim=0)
        abs_err_sp   = torch.abs(v_mean - u)
        mean_ens_std = ens_std.mean().item()
        ens_std_max  = ens_std.max().item()
        ens_std_min  = ens_std.min().item()
        std_err_corr = torch.corrcoef(
            torch.stack([ens_std.flatten(), abs_err_sp.flatten()])
        )[0, 1].item()
        coverage_68  = (abs_err_sp <= 1.0 * ens_std).float().mean().item()
        coverage_95  = (abs_err_sp <= 2.0 * ens_std).float().mean().item()
    else:
        mean_ens_std = ens_std_max = ens_std_min = float("nan")
        std_err_corr = coverage_68 = coverage_95 = float("nan")

    # ── Skill ─────────────────────────────────────────────────────────────────
    mse   = torch.mean(torch.square(u - v_mean))
    rmse  = torch.sqrt(mse).item()
    nrmse = torch.sqrt(mse / (torch.mean(u ** 2) + eps)).item()
    vrmse = torch.sqrt(mse / (torch.var(u)       + eps)).item()

    spread_skill = spread / (rmse + eps) if samples > 1 else float("nan")

    # ── Total conservation ────────────────────────────────────────────────────
    # NOTE: for zero-mean fields sum(u) ≈ 0 and this becomes meaningless.
    # Guard the division — a large |total_error| for velocity / E tensors
    # is an artefact of near-zero denominator, not a real conservation error.
    u_sum = u.sum()
    if u_sum.abs() > eps:
        total_error = (1.0 - v_mean.sum() / u_sum).item()
    else:
        total_error = float("nan")

    # ── Wasserstein-1 ─────────────────────────────────────────────────────────
    w1 = float(
        ot.lp.wasserstein_1d(
            u.flatten().cpu(),
            v.flatten().cpu(),
            p=1.0,
        )
    )

    # ── Sliced EMD ────────────────────────────────────────────────────────────
    if compute_emd:
        coo = torch.cartesian_prod(
            *(torch.linspace(0, 1, s, device=u.device) for s in u.shape)
        )
        emd = float(
            ot.sliced.sliced_wasserstein_distance(
                coo.cpu(),
                coo.cpu(),
                a=(u.flatten() / u.sum()).cpu(),
                b=(v_mean.flatten() / (v_mean.sum() + eps)).cpu(),
                p=1.0,
                n_projections=16,
                seed=42,
            )
        )
    else:
        emd = float("nan")

    # ── Power spectrum + cross-correlation ────────────────────────────────────
    p_u,   k  = isotropic_power_spectrum(u, spatial=spatial)        # (B,)
    p_v,   _  = isotropic_power_spectrum(v, spatial=spatial)        # (N_ENS, B)
    c_uv,  _  = isotropic_cross_correlation(u, v, spatial=spatial)  # (N_ENS, B)

    p_v_mean   = p_v.mean(dim=0)    # (B,)
    c_uv_mean  = c_uv.mean(dim=0)   # (B,)

    eps_log = _adaptive_eps_log(p_u, eps_log_scale)

    # Raw-space metrics
    band_stats = _spectral_band_metrics(
        p_u, p_v_mean, c_uv_mean, k, eps, eps_log, suffix=""
    )

    # ── LOLA-style normalised-space metrics (optional) ────────────────────────
    # Normalise u and v by dataset-level per-field statistics before computing
    # PSD.  After normalisation every field has unit variance so all PSD bins
    # are O(1/n_bins) ≈ 0.02, well above ε = 1e-6.  This eliminates Mode A
    # blow-ups in the original rmse_p formula and makes cross-field comparison
    # numerically fair.
    #
    # Important caveats:
    #   • Use dataset-level (not per-sample) statistics; per-sample
    #     normalisation would force unit variance on every frame, destroying
    #     temporal dynamics.
    #   • Normalisation changes the metric's physical interpretation: rmse_p_n
    #     measures spectral shape error relative to the field's natural
    #     variability, not its absolute power.
    #   • lsd_p (without normalisation) already handles scale variation
    #     through the adaptive ε_log, so normalisation is optional when using
    #     lsd_p.
    if norm_mean is not None and norm_std is not None:
        _mean = torch.as_tensor(norm_mean, dtype=u.dtype, device=u.device)
        _std  = torch.as_tensor(norm_std,  dtype=u.dtype, device=u.device).clamp(min=1e-12)

        u_n = (u - _mean) / _std                       # (H, W)
        v_n = (v - _mean) / _std                       # (N_ENS, H, W)

        p_u_n,  kn  = isotropic_power_spectrum(u_n, spatial=spatial)
        p_v_n,  _   = isotropic_power_spectrum(v_n, spatial=spatial)
        c_uvn,  _   = isotropic_cross_correlation(u_n, v_n, spatial=spatial)

        p_v_mean_n  = p_v_n.mean(dim=0)
        c_uv_mean_n = c_uvn.mean(dim=0)
        eps_log_n   = _adaptive_eps_log(p_u_n, eps_log_scale)

        norm_band_stats = _spectral_band_metrics(
            p_u_n, p_v_mean_n, c_uv_mean_n, kn, eps, eps_log_n, suffix="_n"
        )
        band_stats.update(norm_band_stats)

    return {
        "m1":           m1,
        "m2":           m2,
        "spread":       spread,
        "crps":         crps,
        "mean_ens_std": mean_ens_std,
        "ens_std_max":  ens_std_max,
        "ens_std_min":  ens_std_min,
        "std_err_corr": std_err_corr,
        "coverage_68":  coverage_68,
        "coverage_95":  coverage_95,
        "rmse":         rmse,
        "nrmse":        nrmse,
        "vrmse":        vrmse,
        "spread_skill": spread_skill,
        "total_error":  total_error,
        "w1":           w1,
        "emd":          emd,
        **band_stats,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Build the statistics DataFrame
# ══════════════════════════════════════════════════════════════════════════════

def build_stats_df(
    truth:        torch.Tensor,              # (N_TRAJ, STEPS, H, W, C)
    predictions:  dict[str, torch.Tensor],   # name -> (N_TRAJ, N_ENS, STEPS, H, W, C)
                                             #      or (N_TRAJ,        STEPS, H, W, C)
    field_names:  list[str],
    Ti:           int = 0,
    lead_times:   list[int] | None = None,
    stride:       int = 1,
    traj_indices: list[int] | None = None,
    spatial:      int = 2,
    emd_fields:   set[str] | None = None,
    eps:          float = 1e-6,
    eps_log_scale:float = 1e-8,
    # LOLA-style normalisation statistics.
    # Shape: (C,) — one value per field, same ordering as field_names.
    # If provided, a second family of PSD metrics (suffix _n) is added.
    norm_mean: torch.Tensor | None = None,
    norm_std:  torch.Tensor | None = None,
) -> pd.DataFrame:
    """
    Build a long-format DataFrame of all scalar metrics for every
    combination of (model, trajectory, field, lead-time).

    Args
    ----
    truth        : Ground-truth tensor  (N_TRAJ, STEPS, H, W, C)
    predictions  : dict mapping model name → prediction tensor.
                   Ensemble:     (N_TRAJ, N_ENS, STEPS, H, W, C)
                   Deterministic:(N_TRAJ,        STEPS, H, W, C)
    field_names  : List of field names, length == C.
    Ti           : Context frames to skip in truth.
    lead_times   : Rollout steps to evaluate (0-indexed).  None = all.
    stride       : Multiplier step → physical time.
    traj_indices : Subset of trajectories.  None = all.
    spatial      : Number of spatial dimensions (default 2).
    emd_fields   : Fields for which to compute the expensive Sliced-EMD.
    eps          : Absolute stabiliser for the original rmse_p formula.
    eps_log_scale: Relative scale for the adaptive log stabiliser.
    norm_mean    : Per-field dataset mean  (C,).  Optional.
    norm_std     : Per-field dataset std   (C,).  Optional.

    Returns
    -------
    pd.DataFrame — one row per (model, traj, field, step).
    Columns include all keys returned by compute_stats plus identifiers.
    """
    if emd_fields is None:
        emd_fields = set()

    N_TRAJ, STEPS = truth.shape[:2]

    if traj_indices is None:
        traj_indices = list(range(N_TRAJ))

    pred_steps = [
        (p.shape[2] if p.ndim == 6 else p.shape[1])
        for p in predictions.values()
    ]
    max_steps = min(STEPS - Ti, *pred_steps)

    if lead_times is None:
        lead_times = list(range(max_steps))
    else:
        oob = [t for t in lead_times if t >= max_steps]
        if oob:
            raise ValueError(
                f"lead_times {oob} exceed available steps ({max_steps}). "
                f"truth has {STEPS} steps, Ti={Ti}, "
                f"shortest pred has {min(pred_steps)} steps."
            )

    records = []

    for model_name, pred in predictions.items():
        is_ensemble = pred.ndim == 6

        for traj_idx in traj_indices:
            for field_idx, field_name in enumerate(field_names):

                # Per-field norm statistics (scalars), if provided
                fnorm_mean = (
                    norm_mean[field_idx].item()
                    if norm_mean is not None else None
                )
                fnorm_std = (
                    norm_std[field_idx].item()
                    if norm_std is not None else None
                )

                for step in lead_times:
                    u = truth[traj_idx, Ti + step, :, :, field_idx]
                    u = torch.as_tensor(u, dtype=torch.float32)          # (H, W)

                    if is_ensemble:
                        v = pred[traj_idx, :, step, :, :, field_idx]
                    else:
                        v = pred[traj_idx, step, :, :, field_idx].unsqueeze(0)
                    v = torch.as_tensor(v, dtype=torch.float32)          # (N_ENS, H, W)

                    stats = compute_stats(
                        u, v,
                        spatial=spatial,
                        eps=eps,
                        eps_log_scale=eps_log_scale,
                        compute_emd=(field_name in emd_fields),
                        norm_mean=fnorm_mean,
                        norm_std=fnorm_std,
                    )

                    records.append({
                        "model": model_name,
                        "traj" : traj_idx,
                        "field": field_name,
                        "step" : step,
                        "time" : step * stride,
                        **stats,
                    })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# Build the PSD DataFrame
# ══════════════════════════════════════════════════════════════════════════════

def build_psd_df(
    truth:         torch.Tensor,             # (N_TRAJ, STEPS, H, W, C)
    predictions:   dict[str, torch.Tensor],  # name -> (N_TRAJ, N_ENS, STEPS, H, W, C)
                                             #      or (N_TRAJ,        STEPS, H, W, C)
    field_names:   list[str],
    Ti:            int = 0,
    lead_times:    list[int] | None = None,
    stride:        int = 1,
    traj_indices:  list[int] | None = None,
    spatial:       int = 2,
    min_bin_count: int = 3,
    # Optional: normalise fields before computing PSD for the PSD curves
    # (matches LOLA plotting convention).  Shape (C,) each.
    norm_mean:     torch.Tensor | None = None,
    norm_std:      torch.Tensor | None = None,
) -> pd.DataFrame:
    """
    Compute the isotropic power spectrum for every combination of:
        (model, trajectory, ensemble-member, field, lead-time, wavenumber)

    Returns a long-format DataFrame — one row per wavenumber bin.

    Columns
    -------
    model, traj, ens, field, step, time   — identifiers
    k                                     — spatial frequency  (cycles / pixel)
    wavelength                            — 1 / k              (pixels / cycle)
    psd                                   — power spectral density
    psd_n                                 — PSD in normalised space
                                            (only if norm_mean / norm_std given)
    band                                  — "low" | "mid" | "high" | "vhigh"

    Notes
    -----
    * Ground-truth is added automatically (model="Ground-truth", ens=0).
    * Frequency bins with < min_bin_count contributing pixels are dropped.
    * FIX 4: band edges are derived from k[0] of the full grid (before
      validity masking), matching compute_stats — labels are consistent.
    """
    N_TRAJ, STEPS = truth.shape[:2]

    if traj_indices is None:
        traj_indices = list(range(N_TRAJ))

    pred_steps = [
        (p.shape[2] if p.ndim == 6 else p.shape[1])
        for p in predictions.values()
    ]
    max_steps = min(STEPS - Ti, *pred_steps)

    if lead_times is None:
        lead_times = list(range(max_steps))
    else:
        oob = [t for t in lead_times if t >= max_steps]
        if oob:
            raise ValueError(
                f"lead_times {oob} exceed available steps ({max_steps}). "
                f"truth has {STEPS} steps, Ti={Ti}, "
                f"shortest pred has {min(pred_steps)} steps."
            )

    # ── Pre-compute validity mask (shared across all tensors) ─────────────────
    _sample_shape = tuple(int(x) for x in truth.shape[2:2 + spatial])
    _edges, _counts, _ = isotropic_binning(_sample_shape, device=torch.device("cpu"))
    _valid = (_counts[1:] >= min_bin_count).numpy()

    # FIX 4: band edges use the full k grid BEFORE validity masking.
    _k_full  = _edges[1:]
    _bins    = torch.logspace(
        float(_k_full[0].log2().item()), -1.0, steps=4, base=2.0
    ).tolist()

    def _assign_bands(k_arr: list[float]) -> list[str]:
        out = []
        for k_val in k_arr:
            label = _BAND_NAMES[-1]
            for i in range(len(_BAND_NAMES) - 1):
                if _bins[i] <= k_val <= _bins[i + 1]:
                    label = _BAND_NAMES[i]
                    break
            out.append(label)
        return out

    def _rows_from_tensor(
        v:          torch.Tensor,   # (N_ENS, H, W)
        model_name: str,
        traj_idx:   int,
        field_idx:  int,
        field_name: str,
        step:       int,
    ) -> list[dict]:
        v = torch.as_tensor(v, dtype=torch.float32)
        p_v, k = isotropic_power_spectrum(v, spatial=spatial)  # (N_ENS, B), (B,)

        k_np   = k.numpy(force=True)[_valid]
        wl_np  = 1.0 / k_np
        p_np   = p_v.numpy(force=True)[:, _valid]              # (N_ENS, B')

        k_list    = k_np.tolist()
        wl_list   = wl_np.tolist()
        band_list = _assign_bands(k_list)

        # Normalised PSD (optional)
        p_np_n: "np.ndarray | None" = None
        if norm_mean is not None and norm_std is not None:
            _m = float(norm_mean[field_idx])
            _s = float(norm_std[field_idx])
            if _s > 1e-12:
                v_n = (v - _m) / _s
                p_vn, _ = isotropic_power_spectrum(v_n, spatial=spatial)
                p_np_n  = p_vn.numpy(force=True)[:, _valid]

        n_ens = p_np.shape[0]
        rows  = []
        for ens_idx in range(n_ens):
            row = {
                "model":      model_name,
                "traj":       traj_idx,
                "ens":        ens_idx,
                "field":      field_name,
                "step":       step,
                "time":       step * stride,
                "k":          k_list,
                "wavelength": wl_list,
                "band":       band_list,
                "psd":        p_np[ens_idx].tolist(),
            }
            if p_np_n is not None:
                row["psd_n"] = p_np_n[ens_idx].tolist()
            rows.append(row)
        return rows

    # ── Accumulate records ────────────────────────────────────────────────────
    records = []

    # Ground truth
    for traj_idx in traj_indices:
        for field_idx, field_name in enumerate(field_names):
            for step in lead_times:
                u = truth[traj_idx, Ti + step, :, :, field_idx]
                records.extend(
                    _rows_from_tensor(
                        u.unsqueeze(0), "Ground-truth",
                        traj_idx, field_idx, field_name, step,
                    )
                )

    # Predictions
    for model_name, pred in predictions.items():
        is_ensemble = pred.ndim == 6
        for traj_idx in traj_indices:
            for field_idx, field_name in enumerate(field_names):
                for step in lead_times:
                    if is_ensemble:
                        v = pred[traj_idx, :, step, :, :, field_idx]
                    else:
                        v = pred[traj_idx, step, :, :, field_idx].unsqueeze(0)
                    records.extend(
                        _rows_from_tensor(
                            v, model_name,
                            traj_idx, field_idx, field_name, step,
                        )
                    )

    # ── Explode list columns → long format ────────────────────────────────────
    df      = pd.DataFrame(records)
    explode_cols = ["k", "wavelength", "band", "psd"]
    if "psd_n" in df.columns:
        explode_cols.append("psd_n")

    df = df.explode(explode_cols).reset_index(drop=True)
    df["k"]          = df["k"].astype(float)
    df["wavelength"] = df["wavelength"].astype(float)
    df["psd"]        = df["psd"].astype(float)
    if "psd_n" in df.columns:
        df["psd_n"]  = df["psd_n"].astype(float)
    df["band"] = pd.Categorical(df["band"], categories=_BAND_NAMES, ordered=True)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Utility: compute dataset-level normalisation statistics
# ══════════════════════════════════════════════════════════════════════════════

def compute_norm_stats(
    truth: torch.Tensor,    # (N_TRAJ, STEPS, H, W, C)
    Ti:    int = 0,         # skip context frames
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-field dataset-level mean and std for LOLA-style normalisation.

    Uses the unbiased std (Bessel-corrected) across trajectories, timesteps,
    and spatial positions — the same convention as the original LOLA code.

    Returns
    -------
    norm_mean : (C,) tensor of per-field means
    norm_std  : (C,) tensor of per-field stds  (clamped ≥ 1e-12)
    """
    data = truth[:, Ti:, ...]              # (N_TRAJ, STEPS, H, W, C)
    # Flatten everything except the channel dimension
    flat = data.reshape(-1, data.shape[-1])   # (N_TRAJ * STEPS * H * W, C)
    norm_mean = flat.mean(dim=0)
    norm_std  = flat.std(dim=0, correction=1).clamp(min=1e-12)
    return norm_mean, norm_std
