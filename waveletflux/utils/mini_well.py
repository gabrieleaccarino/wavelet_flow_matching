# Cell 3b — Patch create_mini_well to fix the attribute name mismatch
import copy, os, re, shutil
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from the_well.data.datasets import WellDataset
from pathlib import Path

# Cell 2 — Patched function with mini_dataset_name override
def create_mini_well(
    dataset: WellDataset,
    well_base_path: str,
    mini_dataset_name: str,
    spatial_downsample_factor: int = 1,
    time_downsample_factor: int = 2,
    max_trajectories: int = 10,
    split: str = "train",
    time_fraction: float = 1.0,
):
    from the_well.data.miniwell import process_file

    # Mirrors canonical Well structure: well_base_path/{mini_dataset_name}/data/{split}/
    mini_root    = os.path.join(well_base_path, mini_dataset_name)
    split_output = os.path.join(mini_root, 'data', split)
    os.makedirs(split_output, exist_ok=True)

    # Copy stats.yaml to mini_root — same place WellDataset expects it
    stats_dest = os.path.join(mini_root, 'stats.yaml')
    if not os.path.exists(stats_dest):
        shutil.copy2(dataset.normalization_path, stats_dest)
        print(f"  Copied stats.yaml → {stats_dest}")

    total_trajectories = 0
    split_files = [f for f in dataset.files_paths if split in f]

    for file_path in tqdm(split_files, desc=f"{split}"):
        if total_trajectories >= max_trajectories:
            break

        with h5py.File(file_path, "r") as src_file:
            num_traj      = src_file.attrs["n_trajectories"]
            remaining     = max_trajectories - total_trajectories
            traj_to_write = min(num_traj, remaining)

            out_file_path = os.path.join(split_output, Path(file_path).name)

            with h5py.File(out_file_path, "w") as dst_file:
                process_file(
                    src_file,
                    dst_file,
                    spatial_downsample_factor,
                    time_downsample_factor,
                    time_fraction,
                    traj_to_write,
                )

            total_trajectories += traj_to_write
            print(f"    {Path(file_path).name}: {traj_to_write} traj written")

    print(f"  ✓ {split}: {total_trajectories} trajectories → {split_output}")