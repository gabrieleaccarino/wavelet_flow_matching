from the_well.data.augmentation import Augmentation, Compose
from the_well.data.normalization import ZScoreNormalization
from typing import Dict, Optional, Sequence, Tuple, Any
from the_well.data import WellDataModule, WellDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import fsspec
import torch
import yaml
import os

from waveletflux.utils.augmentations import compose_augmentation, compose_transformation


class WellDataModuleLightning(pl.LightningDataModule):
    """
    Lightning wrapper around the library's WellDataModule.

    Responsibilities:
      - Inject distributed sampler params before setup
      - Apply deterministic `transformation` to ALL dataset splits
      - Apply stochastic `augmentation`  to TRAIN split only
      - Cast batch tensors to `target_dtype`

    The library's WellDataModule is responsible for:
      - Dataset construction, rollout splits, dataset_kws, etc.
    """

    def __init__(
        self,
        well_dm:         WellDataModule,
        train_shuffle:   bool            = True,
        val_shuffle:     bool            = False,
        test_shuffle:    bool            = False,
        rollout_shuffle: bool            = False,
        target_dtype:    torch.dtype     = torch.float32,
        augmentation:    Optional[Augmentation] = None,   # stochastic, train only
        transformation:  Optional[Augmentation] = None,   # deterministic, all splits
    ):
        super().__init__()
        self.well_dm         = well_dm
        self.train_shuffle   = train_shuffle
        self.val_shuffle     = val_shuffle
        self.test_shuffle    = test_shuffle
        self.rollout_shuffle = rollout_shuffle
        self.target_dtype    = target_dtype
        self.augmentation    = augmentation
        self.transformation  = transformation

    # ------------------------------------------------------------------
    # Transform helpers
    # ------------------------------------------------------------------

    def _make_transform(self, split: str) -> Optional[Augmentation]:
        """
        Compose the correct transform per split:
          - train : transformation → augmentation  (det. first, stoch. second)
          - others: transformation only
        """
        ops = []
        if self.transformation is not None:
            ops.append(self.transformation)
        if split == "train" and self.augmentation is not None:
            ops.append(self.augmentation)

        if not ops:
            return None
        return Compose(*ops) if len(ops) > 1 else ops[0]

    def _apply_transforms(self):
        """
        Patch transforms onto every dataset split AFTER the library has
        constructed them in setup(). Maps split name → dataset attribute.
        """
        split_to_attr = {
            "train"        : "train_dataset",
            "val"          : "val_dataset",
            "test"         : "test_dataset",
            "rollout_val"  : "rollout_val_dataset",
            "rollout_test" : "rollout_test_dataset",
        }
        for split, attr in split_to_attr.items():
            ds = getattr(self.well_dm, attr, None)
            if ds is None:
                continue
            transform = self._make_transform(split)
            if transform is not None:
                ds.transform = transform

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def setup(self, stage: Optional[str] = None):
        # 1. Inject distributed params before the library builds datasets
        if hasattr(self, "trainer") and self.trainer is not None:
            if torch.distributed.is_initialized():
                self.well_dm.world_size = self.trainer.world_size
                self.well_dm.rank       = self.trainer.global_rank
            else:
                self.well_dm.world_size = 1
                self.well_dm.rank       = 0

        # 2. Let the library build all dataset splits
        if callable(getattr(self.well_dm, "setup", None)):
            self.well_dm.setup(stage)

        # 3. Patch transforms onto every split (the only place we touch them)
        self._apply_transforms()

    # ------------------------------------------------------------------
    # Dtype conversion
    # ------------------------------------------------------------------

    def _convert_dtype(self, batch: Any) -> Any:
        if isinstance(batch, dict):
            return {k: v.to(dtype=self.target_dtype) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            converted = [v.to(dtype=self.target_dtype) if isinstance(v, torch.Tensor) else v
                         for v in batch]
            return type(batch)(converted)
        if isinstance(batch, torch.Tensor):
            return batch.to(dtype=self.target_dtype)
        return batch

    def _wrap_collate(self, original_collate_fn):
        def collate_wrapper(batch):
            result = original_collate_fn(batch) if original_collate_fn else batch
            return self._convert_dtype(result)
        return collate_wrapper

    def _make_loader(self, original_dl: DataLoader, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset     = original_dl.dataset,
            batch_size  = original_dl.batch_size,
            num_workers = original_dl.num_workers,
            shuffle     = shuffle,
            pin_memory  = original_dl.pin_memory,
            drop_last   = original_dl.drop_last,
            collate_fn  = self._wrap_collate(original_dl.collate_fn),
            sampler     = None,   # Lightning injects DistributedSampler
        )

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def train_dataloader(self):
        return self._make_loader(self.well_dm.train_dataloader(), self.train_shuffle)

    def val_dataloader(self):
        return self._make_loader(self.well_dm.val_dataloader(), self.val_shuffle)

    def test_dataloader(self):
        return self._make_loader(self.well_dm.test_dataloader(), self.test_shuffle)

    def rollout_dataloader(self):
        return self._make_loader(self.well_dm.rollout_val_dataloader(), self.rollout_shuffle)
    

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