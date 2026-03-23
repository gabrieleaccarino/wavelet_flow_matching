from typing import Dict, Optional, Sequence, Tuple
from the_well.data import WellDataset
import torch
import os

from waveletflux.modules.augmentations import compose_augmentation


def get_well_dataset(
    path: str,
    physics: str,
    split: str,
    steps: int = 1,
    include_filters: Sequence[str] = (),
    exclude_filters: Sequence[str] = (),
    augment: Sequence[str] = (),
    **kwargs,
) -> WellDataset:
    r"""Instantiates a dataset from the Well.

    Arguments:
        path: A path to a directory of HDF5 datasets.
        split: The name of the data split to load. Options are "train", "valid", and "test".
            If the path does not contain the split directory, an exception is raised.
        steps: The number of time steps in the trajectories. If negative, load all steps.
        include_filters: Include files whose name contains any of these strings.
        exclude_filters: Exclude files whose name contains any of these strings
        kwargs: Keyword arguments passed to :class:`the_well.data.WellDataset`.

    Returns:
        A dataset from the Well.
    """

    path = os.path.realpath(os.path.expanduser(path), strict=True)

    if os.path.exists(os.path.join(path, physics)):
        path = os.path.join(path, physics)
    elif os.path.exists(os.path.join(path, "datasets", physics)):
        path = os.path.join(path, "datasets", physics)
    else:
        raise NotADirectoryError(f"{os.path.join(path, physics)} does not exist.")

    if os.path.exists(os.path.join(path, split)):
        path = os.path.join(path, split)
    elif os.path.exists(os.path.join(path, "data", split)):
        path = os.path.join(path, "data", split)
    else:
        raise NotADirectoryError(f"{os.path.join(path, split)} does not exist.")

    if augment:
        augmentation = compose_augmentation(*augment)
    else:
        augmentation = None

    if steps < 0:
        dataset = WellDataset(
            path=path,
            include_filters=include_filters,
            exclude_filters=exclude_filters,
        )
        steps = dataset.metadata.n_steps_per_trajectory[0]

    return WellDataset(
        path=path,
        n_steps_input=steps,
        n_steps_output=0,
        include_filters=include_filters,
        exclude_filters=exclude_filters,
        use_normalization=False,
        transform=augmentation,
        **kwargs,
    )


def get_well_inputs(
    item: Dict[str, torch.Tensor],
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Returns the fields and label of a :class:`WellDataset` item.

    Returns:
        The field tensor :math:`x` and label vector, with shape :math:`(*, L, H, W, C)`
        and :math:`(*, D)`, respectively.
    """

    if "input_fields" in item:
        scalars = item["constant_scalars"]
        scalars = scalars.to(device=device, non_blocking=True)

        #bcs = item["boundary_conditions"]
        #bcs = bcs.to(device=device, non_blocking=True)
        #bcs = bcs.flatten(start_dim=scalars.ndim - 1)  # ... H W -> ... (H W)

        #label = torch.cat((scalars, bcs), dim=-1)
        label = scalars

        x = item["input_fields"]
        x = x.to(device=device, non_blocking=True)
        
    elif "state" in item:
        label = item["label"]
        label = label.to(device=device, non_blocking=True)

        x = item["state"]
        x = x.to(device=device, non_blocking=True)
        x = x.movedim(label.ndim, -1)  # L C H W -> L H W C
    else:
        raise KeyError("missing inputs")

    return x, label