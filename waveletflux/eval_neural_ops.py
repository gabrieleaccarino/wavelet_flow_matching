#!/usr/bin/env python

import argparse
import os
import pickle
import sys
from pathlib import Path

import munch
import toml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from waveletflux.modules.neural_ops import NeuralOperatorModule
from waveletflux.utils.well_data import get_well_dataset, get_well_inputs


def parse_args():
    parser = argparse.ArgumentParser(
        prog="eval_neural_ops.py",
        description="Autoregressive rollout evaluation for neural-operator and diffusion baselines",
    )
    parser.add_argument("-c", "--config", type=str, required=True, help="TOML config file")
    parser.add_argument("-k", "--ckpt", type=str, required=True, help="Portable checkpoint")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--traj_limit",
        type=int,
        default=-1,
        help="Number of trajectories to evaluate (-1 = all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Evaluation device (default: cuda if available else cpu)",
    )
    parser.add_argument(
        "--save_metrics",
        action="store_true",
        help="Also save `build_stats_df` and `build_psd_df` outputs as CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for rollout artifacts (default: experiments_path/results)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = munch.munchify(toml.load(args.config))

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    transform_names = config.train.transformations if getattr(config.train, "transformations", None) else []

    dataset = get_well_dataset(
        path=config.path.well_base_path,
        physics=config.path.well_dataset_name,
        split=args.split,
        steps=-1,
        transform=transform_names,
        augment=[],
    )

    model = NeuralOperatorModule.from_checkpoint(args.ckpt, map_location=device)
    model = model.to(device).eval()

    num_traj = len(dataset) if args.traj_limit < 0 else min(args.traj_limit, len(dataset))
    truths = []
    preds = []

    with torch.no_grad():
        for traj_idx in range(num_traj):
            item = dataset[traj_idx]
            x_full, label = get_well_inputs(item, device=device)
            truths.append(x_full.cpu())

            seed = x_full[: config.train.n_steps_input].unsqueeze(0)
            scalar_conds = label.unsqueeze(0) if label is not None else None
            rollout_steps = max(x_full.shape[0] - config.train.n_steps_input, 0)

            pred = model.rollout(
                seed=seed,
                scalar_conds=scalar_conds,
                rollout_steps=rollout_steps,
                verbose=False,
            )
            preds.append(pred.squeeze(0).cpu())

    truth_traj = torch.stack(truths, dim=0)
    pred_traj = torch.stack(preds, dim=0)

    output_dir = Path(args.output_dir or (Path(config.path.experiments_path) / "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    result_stem = (
        config.main.exp_name
        if args.split == "test"
        else f"{config.main.exp_name}_{args.split}"
    )
    gt_name = (
        f"ground_truth_{config.path.well_dataset_name}_traj.pkl"
        if args.split == "test"
        else f"ground_truth_{config.path.well_dataset_name}_{args.split}_traj.pkl"
    )

    with open(output_dir / f"{result_stem}_rollout_traj.pkl", "wb") as f:
        pickle.dump(pred_traj, f)
    with open(output_dir / gt_name, "wb") as f:
        pickle.dump(truth_traj, f)

    if args.save_metrics:
        from waveletflux.utils.eval import build_psd_df, build_stats_df

        meta_fields = dataset.metadata.field_names
        field_names = [item for sublist in meta_fields.values() for item in sublist]
        predictions = {config.main.exp_name: pred_traj}

        stats_df = build_stats_df(
            truth=truth_traj,
            predictions=predictions,
            field_names=field_names,
            Ti=config.train.n_steps_input,
            lead_times=None,
            stride=1,
        )
        psd_df = build_psd_df(
            truth=truth_traj,
            predictions=predictions,
            field_names=field_names,
            Ti=config.train.n_steps_input,
            lead_times=None,
            stride=1,
            min_bin_count=3,
        )

        stats_df.to_csv(output_dir / f"{result_stem}_stats.csv", index=False)
        psd_df.to_csv(output_dir / f"{result_stem}_psd.csv", index=False)

    print(f"Saved rollout predictions to {output_dir / f'{result_stem}_rollout_traj.pkl'}")


if __name__ == "__main__":
    main()
