from the_well.data import WellDataModule
from torch.utils.data import DataLoader
from typing import Optional, Any
import pytorch_lightning as pl
import torch


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