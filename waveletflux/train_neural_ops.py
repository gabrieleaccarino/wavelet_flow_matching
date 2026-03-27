#!/usr/bin/env python
r"""FNO / TFNO baseline training script — PyTorch Lightning + DeepSpeed.

Mirrors the structure of train.py (same PBS script, same torchrun launcher).

Launch (identical to train.py workflow):
    torchrun --nproc_per_node=4 train_fno.py \
        --model fno --modes 16 --n_blocks 4 --hidden 128 \
        --epochs 500 --batch_size 4 --lr 5e-3               \
        --num_workers 16 --save_every 50                     \
        --n_steps_input 4 --n_steps_output 1                 \
        --well_base_path /glade/work/gaccarino/data/datasets \
        --well_dataset   turbulent_radiative_layer_2D        \
        --checkpoint_dir /glade/work/gaccarino/multi-scale-wavelet-flow-matching/experiments/fno_v0.1
"""

import pytorch_lightning as pl
import argparse
import datetime
import warnings
import torch
import sys
import os

# Silence tltorch / neuralop deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tltorch")
warnings.filterwarnings("ignore", category=UserWarning, module="neuralop")

from waveletflux.modules.neural_ops import NeuralOperatorModule
from waveletflux.utils.well_data import WellDataModule

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        prog="train_fno.py",
        description="FNO / TFNO baseline — PyTorch Lightning + DeepSpeed",
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
                        help="Save a checkpoint every N epochs")
    parser.add_argument("--n_steps_input",  type=int,   default=4)
    parser.add_argument("--n_steps_output", type=int,   default=1)

    # DeepSpeed ZeRO stage (0=disabled, 1=optimizer sharding, 2=+gradient sharding)
    parser.add_argument("--zero_stage",     type=int,   default=1,
                        choices=[0, 1, 2],
                        help="DeepSpeed ZeRO optimization stage")

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

    # Detect rank the same way as train.py
    rank    = int(os.environ.get("RANK", "0"))
    is_main = (rank == 0)

    if is_main:
        print(f"[RANK {rank}] starting train_fno.py", flush=True)

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

    # ── Seed (mirrors pl.seed_everything in train.py) ─────────────────────────
    pl.seed_everything(42)

    # ── DataModule ────────────────────────────────────────────────────────────
    data_module = WellDataModule(
        well_base_path=args.well_base_path,
        well_dataset=args.well_dataset,
        stats_path=args.stats_path,
        n_steps_input=args.n_steps_input,
        n_steps_output=args.n_steps_output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Setup early to read F (number of fields)
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
        print(f"  Batch/GPU      : {args.batch_size}")
        print(f"  ZeRO stage     : {args.zero_stage}")
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

    # ── DeepSpeed strategy ────────────────────────────────────────────────────
    # Lightning's DeepSpeedStrategy wraps DDP with ZeRO — does NOT use
    # DataParallel's setattr/replicate, so tltorch ComplexFactorList is safe.
    strategy = pl.strategies.DeepSpeedStrategy(
        stage=args.zero_stage,
        # Optional: enable mixed precision via DeepSpeed
        # precision_plugin=pl.plugins.DeepSpeedPrecisionPlugin(precision="bf16-mixed"),
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    #checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #    dirpath=args.checkpoint_dir,
    #    filename=f"{args.model}" + "-{epoch:04d}-{val_loss:.4f}",
    #    monitor="val_loss",
    #    mode="min",
    #    save_top_k=3,
    #    every_n_epochs=args.save_every,
    #    save_last=True,
    #)
    best_checkpoint = pl.callbacks.ModelCheckpoint(
    dirpath=args.checkpoint_dir,
    filename=f"{args.model}" + "-best-{epoch:04d}-{val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=3,
    save_last=True,
    every_n_epochs=1,           # check every epoch, save when val_loss improves
)




    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    # ── CSV logger (mirrors ExperimentManager CSV logging in train.py) ────────
    csv_logger = pl.loggers.CSVLogger(
        save_dir=args.checkpoint_dir,
        name="logs",
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        accelerator="gpu",
        strategy=strategy,
        devices=torch.cuda.device_count(),
        num_nodes=1,
        max_epochs=args.epochs,
        precision="32-true",          # change to "bf16-mixed" for mixed precision
        callbacks=[best_checkpoint, lr_monitor],
        logger=csv_logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0,       # mirrors train.py
        enable_progress_bar=is_main,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    if is_main:
        start_time = datetime.datetime.now()
        print(f"Training started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    trainer.fit(model, datamodule=data_module)

    if is_main:
        end_time = datetime.datetime.now()
        print(f"\nTraining complete.")
        print(f"  Best val loss  : {best_checkpoint.best_model_score:.6f}")
        print(f"  Best ckpt      : {best_checkpoint.best_model_path}")
        print(f"  Finished       : {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Total time     : {str(end_time - start_time)}")
        
        # Save a single-file checkpoint usable without DeepSpeed
        trainer.save_checkpoint(
            os.path.join(args.checkpoint_dir, f"{args.model}_final.ckpt")
        )
        
    sys.exit(0)


if __name__ == "__main__":
    main()