from the_well.data.normalization import ZScoreNormalization
from typing import Dict, Optional, Sequence, Tuple, Any
from the_well.data import WellDataModule, WellDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import fsspec
import torch
import yaml
import os

from waveletflux.utils.augmentations import compose_augmentation


class WellDataModuleLightning(pl.LightningDataModule):
    """
    A Lightning wrapper around the existing WellDataModule.
    Allows optional overriding of shuffle, batch_size, and num_workers
    for all dataloaders.
    """
    def __init__(
        self,
        well_dm: WellDataModule,
        train_shuffle: Optional[bool] = None,
        val_shuffle: Optional[bool] = None,
        test_shuffle: Optional[bool] = None,
        rollout_shuffle: Optional[bool] = None,
        override_batch_size: Optional[int] = None,
        override_num_workers: Optional[int] = None,
        target_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.well_dm = well_dm
        
        # Optional overrides
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        self.test_shuffle = test_shuffle
        self.rollout_shuffle = rollout_shuffle
        self.override_batch_size = override_batch_size
        self.override_num_workers = override_num_workers
        self.target_dtype = target_dtype
    
    def setup(self, stage: Optional[str] = None):
        """
        Called by Lightning AFTER self.trainer is attached.
        Sets up distributed training parameters if needed.
        """
        if hasattr(self, "trainer") and self.trainer is not None:
            if torch.distributed.is_initialized():
                self.well_dm.world_size = self.trainer.world_size
                self.well_dm.rank = self.trainer.global_rank
            else:
                self.well_dm.world_size = 1
                self.well_dm.rank = 0
        
        # Call the original WellDataModule setup
        if hasattr(self.well_dm, "setup") and callable(self.well_dm.setup):
            self.well_dm.setup(stage)

    def _convert_batch_dtype(self, batch: Any) -> Any:
        """Convert batch tensors to target dtype."""
        if isinstance(batch, dict):
            return {
                key: val.to(dtype=self.target_dtype) if isinstance(val, torch.Tensor) else val
                for key, val in batch.items()
            }
        elif isinstance(batch, (list, tuple)):
            converted = [
                item.to(dtype=self.target_dtype) if isinstance(item, torch.Tensor) else item
                for item in batch
            ]
            return type(batch)(converted)
        elif isinstance(batch, torch.Tensor):
            return batch.to(dtype=self.target_dtype)
        else:
            return batch
    
    def _wrap_collate_fn(self, original_collate_fn):
        """Wrap the collate function to convert dtypes."""
        def collate_wrapper(batch):
            # Call original collate
            result = original_collate_fn(batch) if original_collate_fn else batch
            # Convert dtypes
            return self._convert_batch_dtype(result)
        return collate_wrapper
    
    def _create_dataloader(
        self, 
        original_dl: DataLoader, 
        shuffle: Optional[bool] = None
    ) -> DataLoader:
        """
        Create a new DataLoader with optional overrides and dtype conversion.
        Lightning will inject DistributedSampler automatically if needed.
        """
        # Get original collate function
        original_collate = original_dl.collate_fn
        
        # Determine parameters
        batch_size = self.override_batch_size or original_dl.batch_size
        num_workers = self.override_num_workers or original_dl.num_workers
        use_shuffle = shuffle if shuffle is not None else original_dl.shuffle
        
        # Create new dataloader with wrapped collate function
        return DataLoader(
            dataset=original_dl.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=use_shuffle,
            pin_memory=original_dl.pin_memory,
            drop_last=original_dl.drop_last,
            collate_fn=self._wrap_collate_fn(original_collate),
            sampler=None,  # Let Lightning inject DistributedSampler
        )
    
    def train_dataloader(self):
        dl = self.well_dm.train_dataloader()
        return self._create_dataloader(dl, self.train_shuffle)
    
    def val_dataloader(self):
        dl = self.well_dm.val_dataloader()
        return self._create_dataloader(dl, self.val_shuffle)
    
    def test_dataloader(self):
        dl = self.well_dm.test_dataloader()
        return self._create_dataloader(dl, self.test_shuffle)
    
    def rollout_dataloader(self):
        dl = self.well_dm.rollout_val_dataloader()
        return self._create_dataloader(dl, self.rollout_shuffle)
    

class WellDataModule(pl.LightningDataModule):
    """DataModule for The Well datasets."""

    def __init__(
        self,
        well_base_path:  str,
        well_dataset:    str,
        stats_path:      str,
        n_steps_input:   int,
        n_steps_output:  int,
        batch_size:      int,
        num_workers:     int,
    ):
        super().__init__()
        self.well_base_path  = well_base_path
        self.well_dataset    = well_dataset
        self.stats_path      = stats_path
        self.n_steps_input   = n_steps_input
        self.n_steps_output  = n_steps_output
        self.batch_size      = batch_size
        self.num_workers     = num_workers
        self.normalizer      = None  # set in setup()

    def _make_dataset(self, split: str):
        return WellDataset(
            well_base_path=self.well_base_path,
            well_dataset_name=self.well_dataset,
            well_split_name=split,
            n_steps_input=self.n_steps_input,
            n_steps_output=self.n_steps_output,
            use_normalization=False,
        )

    def setup(self, stage=None):
        self.train_dataset = self._make_dataset("train")
        self.val_dataset   = self._make_dataset("valid")

        with fsspec.open(self.stats_path, mode="r") as f:
            stats = yaml.safe_load(f)

        self.normalizer = ZScoreNormalization(
            stats,
            self.train_dataset.core_field_names,
            self.train_dataset.core_constant_field_names,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
    


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