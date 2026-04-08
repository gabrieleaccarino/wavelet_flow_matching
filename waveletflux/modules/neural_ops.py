from neuralop.models import FNO, TFNO
import pytorch_lightning as pl
from einops import rearrange
import torch


class NeuralOperatorModule(pl.LightningModule):
    """LightningModule wrapping FNO or TFNO."""

    def __init__(
        self,
        model_type:   str,
        modes:        int,
        n_blocks:     int,
        hidden:       int,
        in_channels:  int,
        out_channels: int,
        lr:           float,
        tucker_rank:  float = 0.42,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr

        # Build model
        if model_type == "fno":
            self.model = FNO(
                n_modes=(modes, modes),
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden,
                n_layers=n_blocks,
            )
        elif model_type == "tfno":
            self.model = TFNO(
                n_modes=(modes, modes),
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden,
                n_layers=n_blocks,
                factorization="tucker",
                rank=tucker_rank,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, normalizer):
        x = normalizer.normalize_flattened(batch["input_fields"],  mode="variable")
        y = normalizer.normalize_flattened(batch["output_fields"], mode="variable")
        x = rearrange(x, "B Ti Lx Ly F -> B (Ti F) Lx Ly")
        y = rearrange(y, "B To Lx Ly F -> B (To F) Lx Ly")
        return (self(x) - y).square().mean()

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, self.trainer.datamodule.normalizer)
        self.log("train_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, self.trainer.datamodule.normalizer)
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.lr * 1e-2,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval":  "epoch",
            },
        }
    
    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "NeuralOperatorModule":
        """Load a portable checkpoint saved by PortableCheckpoint.

        Usage:
            fno = NeuralOperatorModule.from_pretrained("path/to/fno_portable-....pt")
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)

        model = cls(**ckpt["hparams"])

        # PortableCheckpoint saves pl_module.state_dict() → keys already have
        # the "model." prefix and match NeuralOperatorModule.load_state_dict().
        # If the file was saved with pl_module.model.state_dict() (old format),
        # keys will be missing the prefix — patch transparently.
        state_dict = ckpt["model_state_dict"]
        sample_key = next(iter(state_dict))
        if not sample_key.startswith("model."):
            state_dict = {"model." + k: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        return model
