from the_well.data.augmentation import Augmentation, Compose
from the_well.data.normalization import ZScoreNormalization
from typing import Dict, Optional, Sequence, Tuple, Any
from torch.utils.data import DataLoader
from the_well.data import WellDataset
import pytorch_lightning as pl
import fsspec
import torch
import yaml
import os

from waveletflux.utils.augmentations import compose_augmentation

def _build_transform(*transforms):
    """Compose only non-None transforms."""
    ops = [t for t in transforms if t is not None]
    if not ops:
        return None
    return Compose(*ops) if len(ops) > 1 else ops[0]


class WellDataModuleLightning(pl.LightningDataModule):

    def __init__(
        self,
        well_base_path        : str,
        well_dataset          : str,
        stats_path            : str,
        n_steps_input         : int,
        n_steps_output        : int,
        max_rollout_steps     : int,
        batch_size            : int,
        num_workers           : int,
        train_shuffle         : bool                   = True,
        val_shuffle           : bool                   = False,
        test_shuffle          : bool                   = False,
        rollout_val_shuffle   : bool                   = False,
        rollout_test_shuffle  : bool                   = False,
        target_dtype          : torch.dtype            = torch.float32,
        augmentation          : Optional[Augmentation] = None,
        transformation        : Optional[Augmentation] = None,
    ):
        super().__init__()
        self.well_base_path       = well_base_path
        self.well_dataset         = well_dataset
        self.stats_path           = stats_path
        self.n_steps_input        = n_steps_input
        self.n_steps_output       = n_steps_output
        self.max_rollout_steps    = max_rollout_steps
        self.batch_size           = batch_size
        self.num_workers          = num_workers
        self.train_shuffle        = train_shuffle
        self.val_shuffle          = val_shuffle
        self.test_shuffle         = test_shuffle
        self.rollout_val_shuffle  = rollout_val_shuffle
        self.rollout_test_shuffle = rollout_test_shuffle
        self.target_dtype         = target_dtype
        self.augmentation         = augmentation
        self.transformation       = transformation

        self.train_dataset        = None
        self.val_dataset          = None
        self.test_dataset         = None
        self.rollout_val_dataset  = None
        self.rollout_test_dataset = None
        self.normalizer           = None

    def _make_dataset(self, split: str, is_train: bool = False, is_rollout: bool = False) -> WellDataset:
        transform = _build_transform(
            self.transformation,
            self.augmentation if is_train else None,
        )
        # Rollout datasets need to predict max_rollout_steps ahead
        # Standard datasets predict n_steps_output ahead
        n_steps_output = self.max_rollout_steps if is_rollout else self.n_steps_output

        return WellDataset(
            well_base_path    = self.well_base_path,
            well_dataset_name = self.well_dataset,
            well_split_name   = split,
            n_steps_input     = self.n_steps_input,
            n_steps_output    = n_steps_output,
            use_normalization = False,
            transform         = transform,
        )

    def setup(self, stage: Optional[str] = None):
        if self.train_dataset is not None:
            return  # already set up — prevents double rebuild when Lightning
                    # calls setup() after manual data_module.setup() in train.py
                    
        self.train_dataset        = self._make_dataset("train", is_train=True)
        self.val_dataset          = self._make_dataset("valid")
        self.test_dataset         = self._make_dataset("test")
        self.rollout_val_dataset  = self._make_dataset("valid", is_rollout=True)
        self.rollout_test_dataset = self._make_dataset("test",  is_rollout=True)

        with fsspec.open(self.stats_path, mode="r") as f:
            stats = yaml.safe_load(f)

        self.normalizer = ZScoreNormalization(
            stats,
            self.train_dataset.core_field_names,
            self.train_dataset.core_constant_field_names,
        )

    # ------------------------------------------------------------------
    # Dtype casting + dataloaders (unchanged from working version)
    # ------------------------------------------------------------------

    def _cast(self, batch: Any) -> Any:
        if isinstance(batch, dict):
            return {k: v.to(self.target_dtype) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            return type(batch)(
                v.to(self.target_dtype) if isinstance(v, torch.Tensor) else v
                for v in batch
            )
        if isinstance(batch, torch.Tensor):
            return batch.to(self.target_dtype)
        return batch

    def _make_loader(self, dataset: WellDataset, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size  = self.batch_size,
            num_workers = self.num_workers,
            shuffle     = shuffle,
            pin_memory  = True,
            drop_last   = drop_last,
            collate_fn  = lambda batch: self._cast(
                torch.utils.data.dataloader.default_collate(batch)
            ),
        )

    def train_dataloader(self):
        return self._make_loader(self.train_dataset,        self.train_shuffle,        drop_last=True)

    def val_dataloader(self):
        return self._make_loader(self.val_dataset,          self.val_shuffle,          drop_last=False)

    def test_dataloader(self):
        return self._make_loader(self.test_dataset,         self.test_shuffle,         drop_last=False)

    def rollout_val_dataloader(self):
        return self._make_loader(self.rollout_val_dataset,  self.rollout_val_shuffle,  drop_last=False)

    def rollout_test_dataloader(self):
        return self._make_loader(self.rollout_test_dataset, self.rollout_test_shuffle, drop_last=False)
    

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