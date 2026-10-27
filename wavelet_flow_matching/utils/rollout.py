
from einops import rearrange
import torch

def get_gt_frame(tensor, traj_idx, lead, n_steps_input, field):
    return tensor[traj_idx, n_steps_input + lead - 1, :, :, field]


def get_pred_frame(tensor, traj_idx, ens_idx, lead, field):
    if tensor.ndim == 6:
        return tensor[traj_idx, ens_idx, lead - 1, :, :, field]
    return tensor[traj_idx, lead - 1, :, :, field]


def normalize_x(x: torch.Tensor, normalizer) -> torch.Tensor:
    """Normalize a (B, T, H, W, C) field tensor in-place over the C dimension."""
    # ZScoreNormalization.normalize_flattened expects (B, T, C) or (B, C) depending on impl;
    # reshape to match what the Well normalizer expects.
    B, T, *spatial, C = x.shape
    flat = x.reshape(B * T, *spatial, C)          # merge batch & time → treat as batch
    flat = normalizer.normalize_flattened(flat, "variable")
    return flat.reshape(B, T, *spatial, C)


def denormalize_x(x: torch.Tensor, normalizer) -> torch.Tensor:
    """Inverse of normalize_x."""
    B, T, *spatial, C = x.shape
    flat = x.reshape(B * T, *spatial, C)
    flat = normalizer.denormalize_flattened(flat, "variable")
    return flat.reshape(B, T, *spatial, C)


@torch.inference_mode()
def rollout_normalized(
    model,
    x_full: torch.Tensor,        # (T_total, H, W, C) — raw physical space
    normalizer,
    n_steps_input:     int,
    rollout_steps:     int,       # explicit arg — no magic T_total-n_steps_input inside
    device: torch.device = 'cpu',
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns
    -------
    y_pred_norm : (rollout_steps, H, W, C)  normalized  ← use for VRMSE
    y_ref_norm  : (rollout_steps, H, W, C)  normalized  ← use for VRMSE
    y_pred_phys : (rollout_steps, H, W, C)  physical    ← use for plots
    y_ref_phys  : (rollout_steps, H, W, C)  physical    ← use for plots
    """
    x_full = x_full.to(device)
    _, H, W, C = x_full.shape

    # Normalize once up front
    x_norm_full = normalize_x(x_full.clone(), normalizer)   # (T_total, H, W, C)

    # Ground truth window: frames [n_steps_input, n_steps_input + rollout_steps)
    y_ref_norm = x_norm_full[n_steps_input : n_steps_input + rollout_steps]
    y_ref_phys = x_full     [n_steps_input : n_steps_input + rollout_steps]

    window_norm = x_norm_full[:n_steps_input].clone()   # (n_in, H, W, C)
    y_preds_norm = []

    for step in range(rollout_steps):
        inp = rearrange(window_norm, 'T H W C -> 1 (T C) H W')
        y_pred_norm = model(inp)                            # (1, T_out*C, H, W)
        y_pred_norm = rearrange(y_pred_norm, '1 (T C) H W -> T H W C', C=C)
        y_preds_norm.append(y_pred_norm)

        if step < rollout_steps - 1:
            window_norm = torch.cat([window_norm[1:], y_pred_norm], dim=0)

    y_pred_norm_out = torch.cat(y_preds_norm, dim=0)       # (rollout_steps, H, W, C)
    y_pred_phys = denormalize_x(y_pred_norm_out.clone(), normalizer)

    return (
        y_pred_norm_out.cpu(),
        y_ref_norm.cpu(),
        y_pred_phys.cpu(),
        y_ref_phys.cpu(),
    )