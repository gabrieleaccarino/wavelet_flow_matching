import matplotlib.pyplot as plt
import pandas as pd
import torch

from waveletflux.utils.fourier import isotropic_power_spectrum, isotropic_cross_correlation


# ══════════════════════════════════════════════════════════════════════════════
# Core metric computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats(
    u: torch.Tensor,          # ground-truth single frame  (H, W)
    v: torch.Tensor,          # predicted frames           (N_ENS, H, W)  or  (1, H, W) for baselines
    spatial: int = 2,
    eps: float = 1e-6,
) -> dict:
    """
    Compute all metrics for a single (field, timestep) pair.

    Args:
        u : Ground-truth frame          (H, W)
        v : Ensemble/predicted frames   (N_ENS, H, W)
        spatial : number of spatial dims

    Returns:
        dict of scalar metric values
    """
    samples = v.shape[0]

    # ── Moments ───────────────────────────────────────────────────────────────
    m1 = torch.mean(u).item()
    m2 = torch.mean(u ** 2).item()

    # ── Ensemble mean ─────────────────────────────────────────────────────────
    v_mean = torch.mean(v, dim=0)   # (H, W)

    # ── Spread (calibration) ──────────────────────────────────────────────────
    if samples > 1:
        spread = torch.mean(torch.square(v - v_mean))
        spread = torch.sqrt((samples + 1) / (samples - 1) * spread).item()
    else:
        spread = float("nan")

    # ── Skill ─────────────────────────────────────────────────────────────────
    se    = torch.square(u - v_mean)
    mse   = torch.mean(se)
    rmse  = torch.sqrt(mse).item()
    nrmse = torch.sqrt(mse / (torch.mean(u ** 2) + eps)).item()
    vrmse = torch.sqrt(mse / (torch.var(u)       + eps)).item()

    # ── Spread / Skill ────────────────────────────────────────────────────────
    spread_skill = spread / (rmse + eps) if samples > 1 else float("nan")

    # ── Total conservation ────────────────────────────────────────────────────
    total_error = (1 - v_mean.sum() / (u.sum() + eps)).item()

    # ── Power spectrum + cross-correlation ────────────────────────────────────
    p_u, k   = isotropic_power_spectrum(u, spatial=spatial)        # (B,)
    p_v, _   = isotropic_power_spectrum(v, spatial=spatial)        # (N_ENS, B)
    c_uv, _  = isotropic_cross_correlation(u, v, spatial=spatial)  # (N_ENS, B)

    p_v_mean   = torch.mean(p_v,  dim=0)   # (B,)
    c_uv_mean  = torch.mean(c_uv, dim=0)   # (B,)

    se_p = torch.square(1 - (p_v_mean + eps) / (p_u + eps))
    se_c = torch.square(1 - (c_uv_mean + eps) / torch.sqrt(p_u * p_v_mean + eps ** 2))

    # 3 log-spaced bands matching reference implementation
    bins = torch.logspace(k[0].log2(), torch.tensor(-1.0), steps=4, base=2)

    band_stats = {}
    for i, band in enumerate(["low", "mid", "high"]):
        if i < 3:
            mask = (bins[i] <= k) & (k <= bins[i + 1])
        else:
            mask = bins[i] <= k

        band_stats[f"rmse_p_{band}"] = torch.sqrt(torch.mean(se_p[mask])).item()
        band_stats[f"rmse_c_{band}"] = torch.sqrt(torch.mean(se_c[mask])).item()

    return {
        "m1":           m1,
        "m2":           m2,
        "spread":       spread,
        "rmse":         rmse,
        "nrmse":        nrmse,
        "vrmse":        vrmse,
        "spread_skill": spread_skill,
        "total_error":  total_error,
        **band_stats,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DataFrame builder
# ══════════════════════════════════════════════════════════════════════════════

"""
def build_stats_df(
    truth:        torch.Tensor,                   # (N_TRAJ, STEPS, H, W, C)
    predictions:  dict[str, torch.Tensor],        # name -> (N_TRAJ, N_ENS, STEPS, H, W, C)
                                                  #      or (N_TRAJ,        STEPS, H, W, C)
    field_names:  list[str],
    lead_times:   list[int],                      # 1-indexed steps to evaluate
    stride:       int = 1,                        # physical time per step
    traj_indices: list[int] | None = None,
    spatial:      int = 2,
) -> pd.DataFrame:
    #
    #Compute all metrics for every combination of:
    #    model × trajectory × field × lead_time
#
    #Args:
    #    truth         : Ground-truth tensor    (N_TRAJ, STEPS, H, W, C)
    #    predictions   : dict of model tensors:
    #                        generative  -> (N_TRAJ, N_ENS, STEPS, H, W, C)
    #                        baseline    -> (N_TRAJ,        STEPS, H, W, C)
    #    field_names   : field labels, length C
    #    lead_times    : 1-indexed rollout steps to evaluate
    #    stride        : multiplier to convert step index -> physical time
    #    traj_indices  : subset of trajectory indices (default: all)
#
    #Returns:
    #    pd.DataFrame with one row per (model, traj, field, lead_time)
    #
    N_TRAJ = truth.shape[0]
    if traj_indices is None:
        traj_indices = list(range(N_TRAJ))

    records = []

    for model_name, pred in predictions.items():
        is_ensemble = pred.ndim == 6   # (N_TRAJ, N_ENS, STEPS, H, W, C)

        for traj_idx in traj_indices:
            for field_idx, field_name in enumerate(field_names):
                for lead in lead_times:
                    step = lead - 1   # 0-indexed

                    # Ground truth frame: (H, W)
                    u = truth[traj_idx, step, :, :, field_idx]
                    u = torch.as_tensor(u, dtype=torch.float32)

                    # Predicted frames: (N_ENS, H, W)
                    if is_ensemble:
                        v = pred[traj_idx, :, step, :, :, field_idx]
                    else:
                        v = pred[traj_idx, step, :, :, field_idx].unsqueeze(0)
                    v = torch.as_tensor(v, dtype=torch.float32)

                    stats = compute_stats(u, v, spatial=spatial)

                    records.append({
                        "model":      model_name,
                        "traj":       traj_idx,
                        "field":      field_name,
                        "lead_time":  lead,
                        "time":       lead * stride,
                        **stats,
                    })

    return pd.DataFrame(records)
"""



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
) -> pd.DataFrame:
    """
    Ti aligns everything automatically:
      - truth  is indexed at  Ti + step
      - preds  are indexed at step          (rollout step 0 = first prediction)

    This means you pass raw tensors with no pre-slicing.
    """
    N_TRAJ, STEPS = truth.shape[:2]

    if traj_indices is None:
        traj_indices = list(range(N_TRAJ))

    # Max number of usable rollout steps across all models
    pred_steps = []
    for pred in predictions.values():
        # ensemble: (N_TRAJ, N_ENS, STEPS, H, W, C) -> axis 2
        # baseline: (N_TRAJ,        STEPS, H, W, C) -> axis 1
        s = pred.shape[2] if pred.ndim == 6 else pred.shape[1]
        pred_steps.append(s)

    max_steps = min(STEPS - Ti, *pred_steps)   # respect both truth and pred lengths

    if lead_times is None:
        lead_times = list(range(max_steps))    # 0-indexed steps
    else:
        # Validate no step goes out of bounds
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

                    # Prediction: 0-indexed from rollout start
                    if is_ensemble:
                        v = pred[traj_idx, :, step, :, :, field_idx]
                    else:
                        v = pred[traj_idx, step, :, :, field_idx].unsqueeze(0)
                    v = torch.as_tensor(v, dtype=torch.float32)        # (N_ENS, H, W)

                    stats = compute_stats(u, v, spatial=spatial)

                    records.append({
                        "model": model_name,
                        "traj" : traj_idx,
                        "field": field_name,
                        "step" : step,
                        "time" : step * stride,
                        **stats,
                    })

    return pd.DataFrame(records)