

def get_gt_frame(tensor, traj_idx, lead, n_steps_input, field):
    return tensor[traj_idx, n_steps_input + lead - 1, :, :, field]

def get_pred_frame(tensor, traj_idx, ens_idx, lead, field):
    if tensor.ndim == 6:
        return tensor[traj_idx, ens_idx, lead - 1, :, :, field]
    return tensor[traj_idx, lead - 1, :, :, field]