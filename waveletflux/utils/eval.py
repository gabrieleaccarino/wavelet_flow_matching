"""
Fourier-domain metrics for WaveletFlow evaluation.

Changelog vs. the original LOLA-derived code
---------------------------------------------
FIX 1  Spread: Bessel correction n/(n-1) instead of (n+1)/(n-1).
FIX 2  CRPS:   Fair CRPS uses 1/(n(n-1)) not 1/n²  (Ferro 2014).
FIX 3  Band edges consistent between compute_stats and build_psd_df.
FIX 4  build_psd_df uses k[0] from the full grid for band edges (not
       k[_valid][0] after applying the min-bin-count mask).
FIX 5  rmse_lp: log-ratio spectral metric — symmetric and robust.

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

WHY rmse_lp IS ROBUST
----------------------
The log-ratio formula replaces the relative error with its log-domain
equivalent:

    se_lp = ( log(p_v + ε) − log(p_u + ε) )²
           = ( log( (p_v + ε) / (p_u + ε) ) )²

Properties:
  • Symmetric:   swapping u ↔ v changes sign only, same magnitude.
  • Bounded:     a ratio of 10⁴ gives se_lp = (log 10⁴)² ≈ 85, not 10⁸.
  • Zero-safe:   ε regularises both sides identically; Mode A is tamed
                 because log(p_v/ε) is large but finite (~18 for p_v~1).
  • Scale-free:  measures multiplicative deviation, natural for power
                 spectra whose dynamic range spans many decades.
  • Interpretable units: rmse_lp ≈ 1 means roughly e-fold (≈2.7×) error
                 in spectral power; rmse_lp ≈ ln 2 ≈ 0.69 means ~2× error.

Note on rmse_c: the coherence denominator is sqrt(p_u·p_v), so by
Cauchy–Schwarz |S_xy|² ≤ S_xx S_yy the ratio c_uv / sqrt(p_u·p_v)
stays in [0, 1].  This makes se_c bounded in [0, 4] and robust.
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


def _spectral_band_metrics(
    p_u:       torch.Tensor,   # (B,)  truth PSD
    p_v_mean:  torch.Tensor,   # (B,)  predicted PSD (ensemble mean)
    c_uv_mean: torch.Tensor,   # (B,)  cross-spectrum magnitude (ensemble mean)
    k:         torch.Tensor,   # (B,)  wavenumber bins
    eps:       float,
) -> dict:
    """
    Compute rmse_p, rmse_lp, and rmse_c for each of the four frequency bands.

    rmse_p   — original relative formula; can blow up (see module docstring).
    rmse_lp  — log-ratio formula; symmetric, bounded, robust to near-zero PSDs.
    rmse_c   — coherence metric; bounded in [0, 2] by Cauchy–Schwarz.
    """
    # ── Original relative error (kept for backward compatibility) ─────────────
    se_p = torch.square(1.0 - (p_v_mean + eps) / (p_u + eps))

    # ── Log-ratio error (FIX 5) ───────────────────────────────────────────────
    # se_lp = ( log(p_v + ε) − log(p_u + ε) )²
    # Equivalent to ( log((p_v + ε) / (p_u + ε)) )² but numerically cleaner
    # to compute as the difference of two log calls.
    se_lp = torch.square(
        torch.log(p_v_mean + eps) - torch.log(p_u + eps)
    )

    # ── Coherence error ───────────────────────────────────────────────────────
    se_c = torch.square(
        1.0 - (c_uv_mean + eps) / torch.sqrt(p_u * p_v_mean + eps ** 2)
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
            out[f"rmse_p_{band}"]  = torch.sqrt(torch.mean(se_p[mask])).item()
            out[f"rmse_lp_{band}"] = torch.sqrt(torch.mean(se_lp[mask])).item()
            out[f"rmse_c_{band}"]  = torch.sqrt(torch.mean(se_c[mask])).item()
        else:
            out[f"rmse_p_{band}"]  = float("nan")
            out[f"rmse_lp_{band}"] = float("nan")
            out[f"rmse_c_{band}"]  = float("nan")

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Core metric computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats(
    u:             torch.Tensor,    # ground-truth single frame  (H, W)
    v:             torch.Tensor,    # predicted frames           (N_ENS, H, W)
    spatial:       int   = 2,
    eps:           float = 1e-6,
    compute_emd:   bool  = False,
) -> dict:
    """
    Compute all metrics for a single (field, timestep) pair.

    Args
    ----
    u           : Ground-truth frame          (H, W)
    v           : Ensemble/predicted frames   (N_ENS, H, W)
    spatial     : Number of spatial dims.
    eps         : Absolute stabiliser for the rmse_p / rmse_lp / rmse_c formulas.
    compute_emd : Whether to compute the expensive 2-D Sliced-EMD.

    Returns
    -------
    dict of scalar metric values.  All keys:

    Scalar statistics
        m1, m2, spread, crps, mean_ens_std, ens_std_max, ens_std_min,
        std_err_corr, coverage_68, coverage_95, rmse, nrmse, vrmse,
        spread_skill, total_error, w1, emd

    Per-band spectral metrics
        rmse_p_{low,mid,high,vhigh}   — original formula (can blow up)
        rmse_lp_{low,mid,high,vhigh}  — log-ratio formula (robust, symmetric)
        rmse_c_{low,mid,high,vhigh}   — coherence metric (robust by C-S)
    """
    samples = v.shape[0]

    # ── Moments ───────────────────────────────────────────────────────────────
    m1 = torch.mean(u).item()
    m2 = torch.mean(u ** 2).item()

    # ── Ensemble mean ─────────────────────────────────────────────────────────
    v_mean = torch.mean(v, dim=0)   # (H, W)

    # ── Spread (calibration) ──────────────────────────────────────────────────
    # Bessel correction n/(n-1).
    if samples > 1:
        spread_sq = torch.mean(torch.square(v - v_mean))
        spread = torch.sqrt(samples / (samples - 1) * spread_sq).item()
    else:
        spread = float("nan")

    # ── CRPS — fair CRPS with 1/(n(n-1)) weight  (Ferro 2014) ────────────────
    abs_err = torch.abs(v - u).mean()

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
    u_sum = u.sum()
    total_error = (
        (1.0 - v_mean.sum() / u_sum).item()
        if u_sum.abs() > eps else float("nan")
    )

    # ── Wasserstein-1 ─────────────────────────────────────────────────────────
    w1 = float(
        ot.lp.wasserstein_1d(
            u.flatten().cpu(),
            v.flatten().cpu(),
            p=1.0,
        )
    )

    # ── Sliced EMD (optional — expensive) ─────────────────────────────────────
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
    p_u,  k   = isotropic_power_spectrum(u, spatial=spatial)        # (B,)
    p_v,  _   = isotropic_power_spectrum(v, spatial=spatial)        # (N_ENS, B)
    c_uv, _   = isotropic_cross_correlation(u, v, spatial=spatial)  # (N_ENS, B)

    p_v_mean  = p_v.mean(dim=0)    # (B,)
    c_uv_mean = c_uv.mean(dim=0)   # (B,)

    band_stats = _spectral_band_metrics(p_u, p_v_mean, c_uv_mean, k, eps)

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
    traj_indices: list[int] | None = None,
    spatial:      int = 2,
    emd_fields:   set[str] | None = None,
    eps:          float = 1e-6,
) -> pd.DataFrame:
    """
    Build a long-format DataFrame of all scalar metrics for every
    combination of (model, traj, field, step).

    Args
    ----
    truth        : Ground-truth tensor  (N_TRAJ, STEPS, H, W, C)
    predictions  : dict mapping model name → prediction tensor.
                   Ensemble:      (N_TRAJ, N_ENS, STEPS, H, W, C)
                   Deterministic: (N_TRAJ,        STEPS, H, W, C)
    field_names  : List of field names, length == C.
    Ti           : Context frames to skip in truth.
    lead_times   : Rollout steps to evaluate (0-indexed).  None = all.
    traj_indices : Subset of trajectories.  None = all.
    spatial      : Number of spatial dimensions (default 2).
    emd_fields   : Fields for which to compute the expensive Sliced-EMD.
    eps          : Absolute stabiliser for rmse_p / rmse_lp / rmse_c.

    Returns
    -------
    pd.DataFrame — one row per (model, traj, field, step).
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
                        compute_emd=(field_name in emd_fields),
                    )

                    records.append({
                        "model": model_name,
                        "traj" : traj_idx,
                        "field": field_name,
                        "step" : step,
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
    traj_indices:  list[int] | None = None,
    spatial:       int = 2,
    min_bin_count: int = 3,
) -> pd.DataFrame:
    """
    Compute the isotropic power spectrum for every combination of
    (model, traj, ensemble-member, field, step, wavenumber).

    Returns a long-format DataFrame — one row per wavenumber bin.

    Columns
    -------
    model, traj, ens, field, step   — identifiers
    k                               — spatial frequency  (cycles / pixel)
    wavelength                      — 1 / k              (pixels / cycle)
    psd                             — power spectral density
    band                            — "low" | "mid" | "high" | "vhigh"

    Notes
    -----
    * Ground-truth is included automatically (model="Ground-truth", ens=0).
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
    _k_full = _edges[1:]
    _bins   = torch.logspace(
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
        field_name: str,
        step:       int,
    ) -> list[dict]:
        v     = torch.as_tensor(v, dtype=torch.float32)
        p_v, k = isotropic_power_spectrum(v, spatial=spatial)  # (N_ENS, B), (B,)

        k_np  = k.numpy(force=True)[_valid]
        p_np  = p_v.numpy(force=True)[:, _valid]               # (N_ENS, B')

        k_list    = k_np.tolist()
        wl_list   = (1.0 / k_np).tolist()
        band_list = _assign_bands(k_list)

        return [
            {
                "model":      model_name,
                "traj":       traj_idx,
                "ens":        ens_idx,
                "field":      field_name,
                "step":       step,
                "k":          k_list,
                "wavelength": wl_list,
                "band":       band_list,
                "psd":        p_np[ens_idx].tolist(),
            }
            for ens_idx in range(p_np.shape[0])
        ]

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
                        traj_idx, field_name, step,
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
                            traj_idx, field_name, step,
                        )
                    )

    # ── Explode list columns → long format ────────────────────────────────────
    df = pd.DataFrame(records)
    df = df.explode(["k", "wavelength", "band", "psd"]).reset_index(drop=True)
    df["k"]          = df["k"].astype(float)
    df["wavelength"] = df["wavelength"].astype(float)
    df["psd"]        = df["psd"].astype(float)
    df["band"]       = pd.Categorical(df["band"], categories=_BAND_NAMES, ordered=True)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Utility: compute dataset-level normalisation statistics
# ══════════════════════════════════════════════════════════════════════════════

