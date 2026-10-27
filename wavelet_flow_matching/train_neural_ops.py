#!/usr/bin/env python
r"""FNO / TFNO baseline training script — PyTorch Lightning + DDP.

Launch:
    torchrun --nproc_per_node=2 train_neural_ops.py \
        --model fno --modes 16 --n_blocks 4 --hidden 128 \
        --epochs 500 --batch_size 16 --lr 5e-3               \
        --num_workers 16 --save_every 50                     \
        --n_steps_input 4 --n_steps_output 1                 \
        --patience 50                                        \
        --well_base_path /glade/work/gaccarino/data/datasets \
        --well_dataset   turbulent_radiative_layer_2D        \
        --checkpoint_dir /glade/work/gaccarino/wavelet_flow_matching/experiments/fno_v0.1
"""
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
import argparse
import datetime
import warnings
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from wavelet_flow_matching.modules.neural_ops import NeuralOperatorModule
from wavelet_flow_matching.utils.well_data import WellDataModuleLightning

# Silence tltorch / neuralop deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tltorch")
warnings.filterwarnings("ignore", category=UserWarning, module="neuralop")


# ── Portable checkpoint callback ──────────────────────────────────────────────

class PortableCheckpoint(Callback):
    """Saves a plain PyTorch state-dict checkpoint whenever val_loss improves.

    The saved file is completely independent of Lightning / DeepSpeed and can
    be loaded with:
        ckpt = torch.load("best.pt")
        model = NeuralOperatorModule(**ckpt["hparams"])
        model.load_state_dict(ckpt["model_state_dict"])
    """

    def __init__(self, dirpath: str, model_type: str, top_k: int = 3):
        super().__init__()
        self.dirpath   = dirpath
        self.model_type = model_type
        self.top_k     = top_k
        self._history: list[tuple[float, str]] = []   # (val_loss, path)
        self.best_score: float | None = None
        self.best_path:  str   | None = None

    # ------------------------------------------------------------------
    def on_validation_epoch_end(self, trainer, pl_module):
        # Only rank-0 writes files
        if trainer.global_rank != 0:
            return

        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is None:
            return
        val_loss = float(val_loss)

        if self.best_score is None or val_loss < self.best_score:
            self.best_score = val_loss
            epoch = trainer.current_epoch

            filename = (
                f"{self.model_type}_portable"
                f"-epoch={epoch:04d}"
                f"-val_loss={val_loss:.6f}.pt"
            )
            path = os.path.join(self.dirpath, filename)

            # Collect hparams so the file is fully self-contained
            hparams = dict(pl_module.hparams) if hasattr(pl_module, "hparams") else {}

            torch.save(
                {
                    "epoch":            epoch,
                    "val_loss":         val_loss,
                    "model_state_dict": pl_module.model.state_dict(),
                    "hparams":          hparams,
                    "model_type":       self.model_type,
                },
                path,
            )
            print(
                f"\n[PortableCheckpoint] saved → {os.path.basename(path)}"
                f"  (val_loss={val_loss:.6f})",
                flush=True,
            )

            self._history.append((val_loss, path))
            self.best_path = path

            # Prune to top-k (remove worst)
            self._history.sort(key=lambda x: x[0])
            while len(self._history) > self.top_k:
                _, old_path = self._history.pop()   # worst is last after sort
                if os.path.exists(old_path):
                    os.remove(old_path)

    # ------------------------------------------------------------------
    @property
    def best_model_path(self) -> str | None:
        return self.best_path

    @property
    def best_model_score(self) -> float | None:
        return self.best_score


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        prog="train_neural_ops.py",
        description="FNO / TFNO baseline — PyTorch Lightning + DDP",
    )

    # Model
    parser.add_argument("--model",       type=str,   default="fno",
                        choices=["fno", "tfno"])
    parser.add_argument("--modes",       type=int,   default=16)
    parser.add_argument("--n_blocks",    type=int,   default=4)
    parser.add_argument("--hidden",      type=int,   default=128)
    parser.add_argument("--tucker_rank", type=float, default=0.42,
                        help="Tucker rank fraction (TFNO only)")

    # Training
    parser.add_argument("--epochs",         type=int,   default=500)
    parser.add_argument("--batch_size",     type=int,   default=4,
                        help="Per-GPU batch size")
    parser.add_argument("--lr",             type=float, default=None,
                        help="Learning rate (default: 5e-3 FNO, 1e-3 TFNO)")
    parser.add_argument("--num_workers",    type=int,   default=4)
    parser.add_argument("--save_every",     type=int,   default=50,
                        help="(kept for back-compat; portable ckpt saves on improvement)")
    parser.add_argument("--n_steps_input",  type=int,   default=4)
    parser.add_argument("--n_steps_output", type=int,   default=1)

    # Early stopping
    parser.add_argument("--patience",       type=int,   default=50,
                        help="Early-stopping patience (epochs without val_loss improvement)")
    parser.add_argument("--min_delta",      type=float, default=1e-6,
                        help="Minimum absolute improvement to reset patience counter")

    # Paths
    parser.add_argument("--well_base_path", type=str,
                        default="/glade/work/gaccarino/data/datasets")
    parser.add_argument("--well_dataset",   type=str,
                        default="turbulent_radiative_layer_2D")
    parser.add_argument("--stats_path",     type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)

    # torchrun injects LOCAL_RANK via env var
    parser.add_argument("--local_rank", type=int,
                        default=int(os.environ.get("LOCAL_RANK", 0)))

    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    rank    = int(os.environ.get("RANK", "0"))
    is_main = (rank == 0)

    if is_main:
        print(f"[RANK {rank}] starting train_neural_ops.py", flush=True)

    # Defaults
    lr = args.lr if args.lr is not None else (5e-3 if args.model == "fno" else 1e-3)

    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(
            "./experiments", f"{args.model}_{args.well_dataset}"
        )
    if args.stats_path is None:
        args.stats_path = os.path.join(
            args.well_base_path, args.well_dataset, "stats.yaml"
        )

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Reproducibility ───────────────────────────────────────────────────────
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("high")

    # ── DataModule ────────────────────────────────────────────────────────────
    data_module = WellDataModuleLightning(
        well_base_path=args.well_base_path,
        well_dataset=args.well_dataset,
        stats_path=args.stats_path,
        n_steps_input=args.n_steps_input,
        n_steps_output=args.n_steps_output,
        max_rollout_steps=args.n_steps_output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    data_module.setup()
    F = data_module.train_dataset.metadata.n_fields

    if is_main:
        print(f"\n{'='*60}")
        print(f"  Model          : {args.model.upper()}")
        print(f"  Modes          : {args.modes}")
        print(f"  Blocks         : {args.n_blocks}")
        print(f"  Hidden         : {args.hidden}")
        print(f"  LR             : {lr}")
        print(f"  Epochs         : {args.epochs}")
        print(f"  Patience       : {args.patience}")
        print(f"  Batch/GPU      : {args.batch_size}")
        print(f"  Fields (F)     : {F}")
        print(f"  In  channels   : {args.n_steps_input  * F}")
        print(f"  Out channels   : {args.n_steps_output * F}")
        print(f"  Checkpoints    : {args.checkpoint_dir}")
        print(f"{'='*60}\n")

    # ── LightningModule ───────────────────────────────────────────────────────
    model = NeuralOperatorModule(
        model_type=args.model,
        modes=args.modes,
        n_blocks=args.n_blocks,
        hidden=args.hidden,
        in_channels=args.n_steps_input  * F,
        out_channels=args.n_steps_output * F,
        lr=lr,
        tucker_rank=args.tucker_rank,
    )

    if is_main:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters     : {n_params:,}\n")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    # 1) Lightning checkpoint — keeps full training state for resuming
    lightning_ckpt = pl.callbacks.ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f"{args.model}-lightning" + "-{epoch:04d}-{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_epochs=1,
    )

    # 2) Portable checkpoint — plain state-dict, shareable without Lightning
    portable_ckpt = PortableCheckpoint(
        dirpath=args.checkpoint_dir,
        model_type=args.model,
        top_k=3,
    )

    # 3) Early stopping
    early_stop = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=args.patience,
        min_delta=args.min_delta,
        verbose=True,
    )

    # 4) LR monitor
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    # ── Logger ────────────────────────────────────────────────────────────────
    csv_logger = pl.loggers.CSVLogger(
        save_dir=args.checkpoint_dir,
        name="logs",
    )

    # ── Trainer (DDP) ─────────────────────────────────────────────────────────
    # DDPStrategy is simpler, more portable, and avoids DeepSpeed file sharding.
    # find_unused_parameters=False is a safe default when all params are used.
    strategy = pl.strategies.DDPStrategy(find_unused_parameters=False)

    trainer = pl.Trainer(
        accelerator="gpu",
        strategy=strategy,
        devices=torch.cuda.device_count(),
        num_nodes=1,
        max_epochs=args.epochs,
        precision="32-true",
        callbacks=[lightning_ckpt, portable_ckpt, early_stop, lr_monitor],
        logger=csv_logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        enable_progress_bar=is_main,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    if is_main:
        start_time = datetime.datetime.now()
        print(f"Training started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    trainer.fit(model, datamodule=data_module)

    if is_main:
        end_time = datetime.datetime.now()
        stopped_early = trainer.early_stopping_callback.stopped_epoch > 0  # type: ignore[union-attr]

        print(f"\nTraining {'stopped early' if stopped_early else 'complete'}.")
        if stopped_early:
            print(f"  Stopped at     : epoch {trainer.current_epoch}")
        print(f"  Best val loss  : {lightning_ckpt.best_model_score:.6f}")
        print(f"  Best Lightning : {lightning_ckpt.best_model_path}")
        print(f"  Best portable  : {portable_ckpt.best_model_path}")
        print(f"  Finished       : {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Total time     : {str(end_time - start_time)}")

    sys.exit(0)


if __name__ == "__main__":
    main()