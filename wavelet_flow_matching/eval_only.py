"""
eval_only.py
------------
Recompute statistics and PSD CSVs from an existing inference pickle,
without re-running model inference.

This is useful when:
  - The eval metrics have been updated (e.g. CRPS sign fix)
  - A new metric has been added
  - You want to re-evaluate with different lead_times / stride

The script is single-process (no torchrun needed) and CPU-only.
It reads the full *_traj.pkl produced by inference.py and writes:

  <stem>_stats.csv   per-(model, traj, field, step) scalar metrics
  <stem>_psd.csv     isotropic PSD in long format

Usage
-----
    python eval_only.py --config configs/trl2d/inference_trl2d.yaml

CLI overrides (same syntax as inference.py)
    python eval_only.py \\
        --config configs/trl2d/inference_trl2d.yaml \\
        --override n_ensemble=8 num_sampling_steps=20
"""

import argparse
import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# CLI  (identical to inference.py)
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WaveletFlow eval-only (no inference)")
    p.add_argument("--config",   required=True, help="Path to YAML experiment config")
    p.add_argument(
        "--override",
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help="Override any config key, e.g. --override n_ensemble=8",
    )
    return p.parse_args()


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for kv in overrides:
        k, v = kv.split("=", 1)
        for cast in (int, float, lambda x: {"true": True, "false": False}[x.lower()]):
            try:
                v = cast(v)
                break
            except (ValueError, KeyError):
                pass
        keys = k.split(".")
        d = cfg
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = v
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = apply_overrides(cfg, args.override or [])

    project_root = cfg["project_root"]
    sys.path.insert(0, project_root)

    from wavelet_flow_matching.utils.well_data import get_well_dataset, get_well_inputs
    from wavelet_flow_matching.utils.eval import build_stats_df, build_psd_df

    well_base_path    = cfg["well_base_path"]
    well_dataset_name = cfg["well_dataset_name"]
    ckpt_name         = cfg["ckpt_name"]
    N_STEPS_INPUT     = cfg["n_steps_input"]
    NUM_SAMPLING      = cfg["num_sampling_steps"]
    N_ENSEMBLE        = cfg["n_ensemble"]   # M: ensemble members (independent realizations) per initial condition

    ckpt_base_name = re.split(r"_\d{8}_", ckpt_name)[0]
    if ckpt_base_name == ckpt_name:
        ckpt_base_name = Path(ckpt_name).stem

    stem     = f"{ckpt_base_name}_nsampling{NUM_SAMPLING}_nens{N_ENSEMBLE}"
    save_dir = Path(project_root) / "experiments" / well_dataset_name / "results"
    pkl_path = save_dir / f"{stem}_traj.pkl"

    print(f"\n{'═'*64}")
    print(f"  Eval-Only Pipeline  (no inference)")
    print(f"{'═'*64}")
    print(f"  Config    : {args.config}")
    print(f"  Dataset   : {well_dataset_name}")
    print(f"  Pickle    : {pkl_path}")
    print(f"{'─'*64}\n")

    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Inference pickle not found: {pkl_path}\n"
            f"Run inference.py first to generate it."
        )

    # ── Load pickle ───────────────────────────────────────────────────────────
    print("  Loading inference pickle …")
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)

    preds_arr    = data["preds"]        # (N_TRAJ, N_ENS, T, H, W, C)  numpy
    field_names  = data["field_names"]
    traj_indices = data["traj_indices"]

    print(f"  preds shape   : {preds_arr.shape}")
    print(f"  field names   : {field_names}")
    print(f"  traj indices  : {traj_indices}\n")

    N_TRAJ, N_ENS, T, H, W, C = preds_arr.shape
    assert N_ENS == N_ENSEMBLE, (
        f"Pickle has {N_ENS} ensemble members but config says {N_ENSEMBLE}."
    )

    # ── Rebuild ground-truth tensor ───────────────────────────────────────────
    # We need the full trajectory including the context window (Ti frames)
    # so that build_stats_df can align truth[Ti + step] with pred[step].
    print("  Rebuilding ground-truth tensor from dataset …")
    dataset = get_well_dataset(
        path      = well_base_path,
        physics   = well_dataset_name,
        split     = cfg.get("split", "test"),
        steps     = cfg.get("steps", -1),
        transform = cfg.get("transform", []),
        augment   = cfg.get("augment",   []),
    )

    seeds_list = []
    for idx in traj_indices:
        item      = dataset[idx]
        x_full, _ = get_well_inputs(item, device=torch.device("cpu"))
        # keep exactly Ti + T frames so truth and preds align
        seeds_list.append(x_full.numpy()[:N_STEPS_INPUT + T])

    truth_traj  = torch.from_numpy(np.stack(seeds_list, axis=0))  # (N_TRAJ, Ti+T, H, W, C)
    preds_torch = torch.from_numpy(preds_arr)                     # (N_TRAJ, N_ENS, T, H, W, C)
    predictions = {ckpt_base_name: preds_torch}

    print(f"  truth shape   : {tuple(truth_traj.shape)}\n")

    # ── Stats CSV ─────────────────────────────────────────────────────────────
    print("  Building stats DataFrame …")
    stats_df = build_stats_df(
        truth       = truth_traj,
        predictions = predictions,
        field_names = field_names,
        Ti          = N_STEPS_INPUT,
        lead_times  = None,
        stride      = cfg.get("stride", 1),
    )
    stats_csv = save_dir / f"{stem}_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"  [1/2] Stats CSV → {stats_csv}  ({len(stats_df):,} rows)")

    # ── PSD CSV ───────────────────────────────────────────────────────────────
    print("  Building PSD DataFrame …")
    psd_df = build_psd_df(
        truth         = truth_traj,
        predictions   = predictions,
        field_names   = field_names,
        Ti            = N_STEPS_INPUT,
        lead_times    = None,
        stride        = cfg.get("stride", 1),
        min_bin_count = cfg.get("min_bin_count", 3),
    )
    psd_csv = save_dir / f"{stem}_psd.csv"
    psd_df.to_csv(psd_csv, index=False)
    print(f"  [2/2] PSD CSV   → {psd_csv}  ({len(psd_df):,} rows)")

    print(f"\n{'═'*64}")
    print(f"  Done.  Outputs in: {save_dir}")
    print(f"{'═'*64}\n")


if __name__ == "__main__":
    main()
