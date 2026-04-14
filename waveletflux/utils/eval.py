import ot
import pandas as pd
import torch

from waveletflux.utils.fourier import isotropic_power_spectrum, isotropic_cross_correlation, isotropic_binning

# ══════════════════════════════════════════════════════════════════════════════
# Core metric computation
# ══════════════════════════════════════════════════════════════════════════════

# Frequency-band labels.  Four bands, matching eval.py's `for i in range(4)`.
_BAND_NAMES = ["low", "mid", "high", "vhigh"]


def compute_stats(
    u: torch.Tensor,           # ground-truth single frame  (H, W)
    v: torch.Tensor,           # predicted frames           (N_ENS, H, W)  or  (1, H, W) for baselines
    spatial: int = 2,
    eps: float = 1e-6,
    compute_emd: bool = False,  # Sliced-EMD is expensive; enable only for density fields
) -> dict:
    """
    Compute all metrics for a single (field, timestep) pair.

    Args:
        u           : Ground-truth frame          (H, W)
        v           : Ensemble/predicted frames   (N_ENS, H, W)
        spatial     : number of spatial dims
        eps         : numerical stabiliser
        compute_emd : whether to compute the (expensive) 2-D Sliced-EMD.
                      Set to True only for density-like fields.

    Returns:
        dict of scalar metric values

    Metric notes
    ------------
    crps
        Fair CRPS for ensemble models (samples > 1):
            CRPS = E|X - y| - (1/2) E|X - X'|
        The (1/2)E|X-X'| term (always ≥ 0) is computed via the sorted-ensemble
        identity in O(n log n).  For deterministic models (samples == 1) it
        reduces exactly to the MAE.

    spread
        Unbiased ensemble standard deviation (Bessel-corrected):
            sqrt( n/(n-1) * mean_sq_deviation )
        NaN for deterministic models.

    spread_skill
        spread / RMSE.  Perfect calibration → 1.
        NaN for deterministic models.

    vrmse
        RMSE normalised by sqrt(Var(u)).  Equivalent to the standard deviation
        of the error normalised by the signal's own variability.  A value of 1
        means the model error equals the natural variability of the field.
        Guarded by eps to handle nearly-uniform fields.

    nrmse
        RMSE normalised by sqrt(E[u²]).  Similar to vrmse but uses the second
        moment instead of the variance, so it is sensitive to the mean level.

    total_error
        Relative mass / integral conservation error:
            1 - sum(v_mean) / sum(u)
        Zero means the ensemble mean conserves the field integral exactly.

    w1
        1-D Wasserstein-1 distance between the empirical marginal distributions
        of u and all ensemble members stacked.

    emd
        2-D Sliced Wasserstein distance between the spatial mass distributions of
        u and v_mean, each normalised to a probability measure.
        Only meaningful for density-like (non-negative, integrable) fields.
        NaN when compute_emd=False.

    rmse_p_{band} / rmse_c_{band}
        RMS relative error in isotropic power (p) and coherence (c) within
        each of the four log-spaced frequency bands.
    """
    samples = v.shape[0]

    # ── Moments ───────────────────────────────────────────────────────────────
    m1 = torch.mean(u).item()
    m2 = torch.mean(u ** 2).item()

    # ── Ensemble mean ─────────────────────────────────────────────────────────
    v_mean = torch.mean(v, dim=0)   # (H, W)

    # ── Spread (calibration) ──────────────────────────────────────────────────
    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  FIX 1: Use standard Bessel correction n/(n-1) instead of (n+1)/(n-1)
    # ║  Old:  torch.sqrt((samples + 1) / (samples - 1) * spread)
    # ║  New:  torch.sqrt( samples      / (samples - 1) * spread)
    # ║
    # ║  The (n+1)/(n-1) factor inflated spread by ~12% at n=4, giving
    # ║  optimistic spread/skill ratios.  Standard unbiased variance uses
    # ║  Bessel's correction: Var = (1/(n-1)) * Σ(x_i - x̄)²
    # ║  which is equivalent to  n/(n-1) * mean_sq_deviation.
    # ╚═══════════════════════════════════════════════════════════════════════╝
    if samples > 1:
        spread = torch.mean(torch.square(v - v_mean))
        spread = torch.sqrt(samples / (samples - 1) * spread).item()       # FIX 1
    else:
        spread = float("nan")

    # ── CRPS ──────────────────────────────────────────────────────────────────
    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  FIX 2: Fair CRPS uses 1/(n(n-1)) not 1/n² (Ferro 2014)
    # ║  Old:  weights = (2*arange(1,n+1) - n - 1) / (n * n)
    # ║  New:  weights = (2*arange(1,n+1) - n - 1) / (n * (n - 1))
    # ║
    # ║  The difference is a factor n/(n-1).  For n=4 that's 33% undercount
    # ║  of the ensemble-spread correction, inflating CRPS and making
    # ║  generative models look worse vs deterministic baselines.
    # ╚═══════════════════════════════════════════════════════════════════════╝
    abs_err = torch.abs(v - u).mean()   # E|X - y|, averaged over space & ensemble

    if samples > 1:
        v_sorted  = torch.sort(v.reshape(samples, -1), dim=0).values   # (N_ENS, H*W)
        n         = samples
        weights   = (2 * torch.arange(1, n + 1, device=v.device) - n - 1).float() / (n * (n - 1))   # FIX 2
        fair_term = (weights[:, None] * v_sorted).sum(dim=0).mean()    # (1/2)E|X-X'| ≥ 0
        crps      = (abs_err - fair_term).item()
    else:
        # Deterministic: CRPS = MAE
        crps = abs_err.item()

    if samples > 1:
        ens_std = v.std(dim=0)                         # (H, W) pointwise std
        abs_err = torch.abs(v_mean - u)                # (H, W)

        mean_ens_std = ens_std.mean().item()
        ens_std_max  = ens_std.max().item()
        ens_std_min  = ens_std.min().item()

        std_err_corr = torch.corrcoef(
            torch.stack([ens_std.flatten(), abs_err.flatten()])
        )[0, 1].item()

        # coverage: fraction of pixels where truth is within ±k*std of ensemble mean
        coverage_68 = ((abs_err <= 1.0 * ens_std).float().mean()).item()
        coverage_95 = ((abs_err <= 2.0 * ens_std).float().mean()).item()
    else:
        mean_ens_std = float("nan")
        ens_std_max  = float("nan")
        ens_std_min  = float("nan")
        std_err_corr = float("nan")
        coverage_68  = float("nan")
        coverage_95  = float("nan")

    # ── Skill ─────────────────────────────────────────────────────────────────
    se    = torch.square(u - v_mean)
    mse   = torch.mean(se)
    rmse  = torch.sqrt(mse).item()
    nrmse = torch.sqrt(mse / (torch.mean(u ** 2) + eps)).item()
    vrmse = torch.sqrt(mse / (torch.var(u)       + eps)).item()

    # ── Spread / Skill ────────────────────────────────────────────────────────
    # Perfect calibration → 1.  Only defined for ensemble models.
    spread_skill = spread / (rmse + eps) if samples > 1 else float("nan")

    # ── Total conservation ────────────────────────────────────────────────────
    total_error = (1 - v_mean.sum() / (u.sum() + eps)).item()

    # ── Wasserstein-1 (1-D marginal) ─────────────────────────────────────────
    # Treat all ensemble members as empirical samples of the predicted
    # distribution — all pixels × all members form the empirical CDF.
    w1 = float(
        ot.lp.wasserstein_1d(
            u.flatten().cpu(),
            v.flatten().cpu(),   # (N_ENS * H * W,)
            p=1.0,
        )
    )

    # ── Sliced Earth Mover's Distance (2-D spatial) ───────────────────────────
    # Only meaningful for density / mass-like fields; guard behind the flag.
    # Uses the ensemble mean as the predicted distribution, normalised to a
    # probability measure.
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
    p_u, k   = isotropic_power_spectrum(u, spatial=spatial)        # (B,)
    p_v, _   = isotropic_power_spectrum(v, spatial=spatial)        # (N_ENS, B)
    c_uv, _  = isotropic_cross_correlation(u, v, spatial=spatial)  # (N_ENS, B)

    p_v_mean  = torch.mean(p_v,  dim=0)   # (B,)
    c_uv_mean = torch.mean(c_uv, dim=0)   # (B,)

    # Relative error in power and coherence:
    #   se_p uses eps in numerator + denominator  (additive stabilisation)
    #   se_c uses eps² under the sqrt             (eps² = 1e-12 at default eps)
    se_p = torch.square(1 - (p_v_mean + eps) / (p_u + eps))
    se_c = torch.square(1 - (c_uv_mean + eps) / torch.sqrt(p_u * p_v_mean + eps ** 2))

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  FIX 3: Use min_bin_count-consistent band edges
    # ║  Old: bins = torch.logspace(k[0].log2(), ...)
    # ║  New: bins = torch.logspace(k[0].log2(), ...)
    # ║       (same formula, but now both compute_stats and build_psd_df
    # ║        use k[0] consistently — see build_psd_df fix below)
    # ╚═══════════════════════════════════════════════════════════════════════╝
    bins = torch.logspace(k[0].log2().item(), -1.0, steps=4, base=2.0)

    band_stats = {}
    for i, band in enumerate(_BAND_NAMES):
        if i < 3:
            mask = (bins[i] <= k) & (k <= bins[i + 1])
        else:                          # i == 3: everything above the last edge
            mask = bins[i] <= k

        # Guard against empty bands (can happen on small grids)
        if mask.any():
            band_stats[f"rmse_p_{band}"] = torch.sqrt(torch.mean(se_p[mask])).item()
            band_stats[f"rmse_c_{band}"] = torch.sqrt(torch.mean(se_c[mask])).item()
        else:
            band_stats[f"rmse_p_{band}"] = float("nan")
            band_stats[f"rmse_c_{band}"] = float("nan")

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
    truth:        torch.Tensor,             # (N_TRAJ, STEPS, H, W, C)
    predictions:  dict[str, torch.Tensor],  # name -> (N_TRAJ, N_ENS, STEPS, H, W, C)
                                            #      or (N_TRAJ,        STEPS, H, W, C)
    field_names:  list[str],
    Ti:           int = 0,                  # context frames to skip
    lead_times:   list[int] | None = None,  # if None, evaluate all available steps
    stride:       int = 1,
    traj_indices: list[int] | None = None,
    spatial:      int = 2,
    emd_fields:   set[str] | None = None,   # field names for which to compute EMD
                                            # (expensive — omit for non-density fields)
) -> pd.DataFrame:
    """
    Build a long-format DataFrame of all scalar metrics for every combination
    of (model, trajectory, field, lead-time).

    Ti aligns everything automatically:
      - truth  is indexed at  Ti + step
      - preds  are indexed at step          (rollout step 0 = first prediction)

    Args:
        truth        : Ground-truth tensor  (N_TRAJ, STEPS, H, W, C)
        predictions  : dict mapping model name to prediction tensor.
                       Ensemble models:     (N_TRAJ, N_ENS, STEPS, H, W, C)
                       Deterministic models:(N_TRAJ,        STEPS, H, W, C)
        field_names  : List of field names, length == C.
        Ti           : Number of context frames to skip in truth.
        lead_times   : Which rollout steps to evaluate (0-indexed).  None = all.
        stride       : Multiplier to convert step index to physical time units.
        traj_indices : Subset of trajectories to evaluate.  None = all.
        spatial      : Number of spatial dimensions (default 2 for H×W).
        emd_fields   : Field names for which to compute the expensive Sliced-EMD.
                       Defaults to the empty set.

    Returns:
        pd.DataFrame with columns:
            model, traj, field, step, time,
            m1, m2, spread, crps, rmse, nrmse, vrmse, spread_skill,
            total_error, w1, emd,
            rmse_p_{low,mid,high,vhigh}, rmse_c_{low,mid,high,vhigh}
    """
    if emd_fields is None:
        emd_fields = set()

    N_TRAJ, STEPS = truth.shape[:2]

    if traj_indices is None:
        traj_indices = list(range(N_TRAJ))

    # Max number of usable rollout steps across all models
    pred_steps = []
    for pred in predictions.values():
        s = pred.shape[2] if pred.ndim == 6 else pred.shape[1]
        pred_steps.append(s)

    max_steps = min(STEPS - Ti, *pred_steps)

    if lead_times is None:
        lead_times = list(range(max_steps))
    else:
        oob = [t for t in lead_times if t >= max_steps]
        if oob:
            raise ValueError(
                f"lead_times {oob} exceed available steps ({max_steps}). "
                f"truth has {STEPS} steps, Ti={Ti}, shortest pred has {min(pred_steps)} steps."
            )

    records = []

    for model_name, pred in predictions.items():
        is_ensemble = pred.ndim == 6

        for traj_idx in traj_indices:
            for field_idx, field_name in enumerate(field_names):
                for step in lead_times:

                    # Ground truth: skip the context window
                    u = truth[traj_idx, Ti + step, :, :, field_idx]
                    u = torch.as_tensor(u, dtype=torch.float32)        # (H, W)

                    # Prediction: 0-indexed from rollout start.
                    # Deterministic predictions are unsqueezed to (1, H, W) so
                    # compute_stats always receives an (N_ENS, H, W) tensor.
                    if is_ensemble:
                        v = pred[traj_idx, :, step, :, :, field_idx]
                    else:
                        v = pred[traj_idx, step, :, :, field_idx].unsqueeze(0)
                    v = torch.as_tensor(v, dtype=torch.float32)        # (N_ENS, H, W)

                    stats = compute_stats(
                        u, v,
                        spatial=spatial,
                        compute_emd=(field_name in emd_fields),
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
# Build the DataFrame for power spectral density (PSD) curves
# ══════════════════════════════════════════════════════════════════════════════

def build_psd_df(
    truth:         torch.Tensor,             # (N_TRAJ, STEPS, H, W, C)
    predictions:   dict[str, torch.Tensor],  # name -> (N_TRAJ, N_ENS, STEPS, H, W, C)
                                             #      or (N_TRAJ,        STEPS, H, W, C)
    field_names:   list[str],
    Ti:            int = 0,                  # context frames to skip
    lead_times:    list[int] | None = None,  # 0-indexed rollout steps; None = all
    stride:        int = 1,
    traj_indices:  list[int] | None = None,
    spatial:       int = 2,
    min_bin_count: int = 3,                  # drop frequency bins with fewer contributing
                                             # pixels than this — suppresses Nyquist noise
) -> pd.DataFrame:
    """
    Compute the isotropic power spectrum for every combination of:
        (model, trajectory, ensemble-member, field, lead-time, wavenumber)

    The returned DataFrame is in **long format** — one row per wavenumber bin —
    so that seaborn can aggregate freely (mean over traj / ens / step, CI bands, …).

    Columns
    -------
    model, traj, ens, field, step, time   – identifiers
    k                                     – spatial frequency  (cycles / pixel)
    wavelength                            – 1 / k              (pixels / cycle)
    psd                                   – power spectral density
    band                                  – categorical: "low" | "mid" | "high" | "vhigh"

    Notes
    -----
    * "Ground-truth" is added automatically as a model with a single ensemble
      member (ens=0).
    * For deterministic baselines (5-D pred tensor) ens is always 0.
    * Frequency bins with fewer than `min_bin_count` contributing pixels are
      dropped.  These are the isotropic corners of the 2-D spectrum (near Nyquist)
      where the radial average is based on only 1–2 pixels and produces noisy,
      model-dependent spikes.  Default = 3 is conservative; increase to 8–16
      for cleaner plots on small grids.
    * The `band` column uses the same 4 log-spaced edges as `compute_stats`
      so that band-level filtering is consistent between scalar metrics and PSD
      curves.
    * All PSD arrays are stored as Python lists before being passed to pandas,
      which prevents silent k↔psd misalignment when explode() is called on
      object-dtype columns.
    """
    N_TRAJ, STEPS = truth.shape[:2]

    if traj_indices is None:
        traj_indices = list(range(N_TRAJ))

    # ── derive max usable rollout steps ───────────────────────────────────────
    pred_steps = []
    for pred in predictions.values():
        s = pred.shape[2] if pred.ndim == 6 else pred.shape[1]
        pred_steps.append(s)

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

    # ── pre-compute low-count bin mask (shared across all tensors) ────────────
    _sample_shape = tuple(int(x) for x in truth.shape[2:2 + spatial])  # (H, W)
    _edges, _counts, _ = isotropic_binning(_sample_shape, device=torch.device("cpu"))
    _valid = (_counts[1:] >= min_bin_count).numpy()   # (B,) bool

    # ╔═══════════════════════════════════════════════════════════════════════╗
    # ║  FIX 4: Use k[0] (BEFORE validity mask) for band edges, matching
    # ║  compute_stats which also uses k[0].
    # ║  Old: _bins used k[_valid][0] — the first VALID bin
    # ║  New: _bins uses k[0] — the first bin (same as compute_stats)
    # ║  This ensures the "low", "mid", "high", "vhigh" labels in the PSD
    # ║  DataFrame correspond to the same frequency ranges as the
    # ║  rmse_p_low, rmse_p_mid, etc. columns in the stats DataFrame.
    # ╚═══════════════════════════════════════════════════════════════════════╝
    # We compute bin edges from the full k grid (edges[1:]), then apply
    # the validity mask only for data output.
    _k_full = _edges[1:]  # full k grid before validity masking
    _bins_tensor = torch.logspace(
        float(_k_full[0].log2().item()),
        -1.0,
        steps=4,
        base=2.0,
    )
    _bins: list[float] = _bins_tensor.tolist()

    def _assign_bands(k_arr: list[float]) -> list[str]:
        """Map each wavenumber to its band label using the shared bin edges."""
        out = []
        for k_val in k_arr:
            label = _BAND_NAMES[-1]               # default: "vhigh"
            for i in range(len(_BAND_NAMES) - 1):
                if _bins[i] <= k_val <= _bins[i + 1]:
                    label = _BAND_NAMES[i]
                    break
            out.append(label)
        return out

    # ── helper: collect PSD rows for one (traj, field, step) slice ───────────
    def _rows_from_tensor(
        v:          torch.Tensor,   # (N_ENS, H, W)  already sliced
        model_name: str,
        traj_idx:   int,
        field_name: str,
        step:       int,
    ) -> list[dict]:

        v = torch.as_tensor(v, dtype=torch.float32)
        p_v, k = isotropic_power_spectrum(v, spatial=spatial)  # (N_ENS, B), (B,)

        # ── owned copies, not views ──────────────────────────────────────────
        k_np  = k.numpy(force=True)[_valid]           # (B',) via bool-index → owned
        wl_np = 1.0 / k_np                            # (B',) all k > 0 after mask
        p_np  = p_v.numpy(force=True)[:, _valid]      # (N_ENS, B') owned copy

        k_list    = k_np.tolist()
        wl_list   = wl_np.tolist()
        band_list = _assign_bands(k_list)

        n_ens = p_np.shape[0]
        rows  = []
        for ens_idx in range(n_ens):
            rows.append({
                "model":      model_name,
                "traj":       traj_idx,
                "ens":        ens_idx,
                "field":      field_name,
                "step":       step,
                "time":       step * stride,
                "k":          k_list,
                "wavelength": wl_list,
                "band":       band_list,
                "psd":        p_np[ens_idx].tolist(),   # .tolist() → independent copy
            })
        return rows

    # ── accumulate records ────────────────────────────────────────────────────
    records = []

    # Ground truth (model="Ground-truth", ens=0)
    for traj_idx in traj_indices:
        for field_idx, field_name in enumerate(field_names):
            for step in lead_times:
                u = truth[traj_idx, Ti + step, :, :, field_idx]
                records.extend(
                    _rows_from_tensor(u.unsqueeze(0), "Ground-truth", traj_idx, field_name, step)
                )

    # Predictions
    for model_name, pred in predictions.items():
        is_ensemble = pred.ndim == 6

        for traj_idx in traj_indices:
            for field_idx, field_name in enumerate(field_names):
                for step in lead_times:
                    if is_ensemble:
                        v = pred[traj_idx, :, step, :, :, field_idx]    # (N_ENS, H, W)
                    else:
                        v = pred[traj_idx, step, :, :, field_idx].unsqueeze(0)
                    records.extend(
                        _rows_from_tensor(v, model_name, traj_idx, field_name, step)
                    )

    # ── build DataFrame: each row holds arrays → explode to long format ───────
    df = pd.DataFrame(records)
    df = df.explode(["k", "wavelength", "band", "psd"]).reset_index(drop=True)
    df["k"]          = df["k"].astype(float)
    df["wavelength"] = df["wavelength"].astype(float)
    df["psd"]        = df["psd"].astype(float)
    df["band"]       = pd.Categorical(df["band"], categories=_BAND_NAMES, ordered=True)

    return df
