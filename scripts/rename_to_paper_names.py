#!/usr/bin/env python3
"""Rename experiment outputs to the model names used in the paper.

Inference jobs write files named after their training run, e.g.
``WaveletFlow_3scales_haar_v4.0_nsampling50_nens8_seed42_rayleigh_benard_metrics.csv``.
This script renames them to one scheme, keyed by the paper's model names:

    <MODEL>_<pde>_seed<S>_metrics.csv          per-seed metrics (3 seeds)
    <MODEL>_<pde>_ic<eta>_seed<S>_metrics.csv  IC-perturbation runs
    <MODEL>_<pde>_stats.csv                    single run (submission)
    <MODEL>_<pde>_perf.yaml                    inference profiling
    <MODEL>_<pde>_traj.pkl                     one-sample rollout (qualitative figures)

MODEL is one of WFM_1scale_<w>, WFM_3scale_<w> (w in haar, db2, db4, db6),
FM_pixel, FM_fourier (32 modes), FM_fourier16, WDNO (db2), WDNO_db6, LatentFM,
WNO, FNO, TFNO.

Files that are not used by the paper (older versions, backups, '-'-prefixed
disabled runs, UNet baselines, duplicates) are moved to
``_to_delete/experiments_obsolete/<pde>/<folder>/``.

Every move is written to a manifest (``artifacts/manifests/rename_*.csv``);
``--revert <manifest>`` undoes it. Run without ``--apply`` for a dry run.

    python scripts/rename_to_paper_names.py            # dry run
    python scripts/rename_to_paper_names.py --apply
    python scripts/rename_to_paper_names.py --revert artifacts/manifests/rename_<stamp>.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
PDES = ["turbulent_radiative_layer_2D", "rayleigh_benard", "active_matter"]
FOLDERS = ["metrics", "performance", "results"]

# run stem (pde, seed, sampling and suffix removed) -> paper model name
MODEL_RULES = [
    (r"WaveletFlow_1scale_(haar|db\d)(?:_ablation)?_v4\.0", r"WFM_1scale_\1"),
    (r"WaveletFlow_3scales_(haar|db\d)(?:_ablation)?_v4\.0", r"WFM_3scale_\1"),
    (r"FM_data_space_v4\.0", "FM_pixel"),
    (r"FourierFlow_32modes_v4\.0", "FM_fourier"),
    (r"FourierFlow_16modes_v4\.0", "FM_fourier16"),
    (r"WDNO(?:_1scale)?(?:_db2)?", "WDNO"),
    (r"WDNO_db6", "WDNO_db6"),
    (r"LatentFM_dcae_f32c64_v4\.0", "LatentFM"),
    (r"WNO_3scales_db2", "WNO"),
    (r"FNO", "FNO"),
    (r"TFNO", "TFNO"),
]
PAPER_MODELS = {"FM_pixel", "FM_fourier", "FM_fourier16", "WDNO", "WDNO_db6", "LatentFM", "WNO", "FNO", "TFNO"}
OBSOLETE = [
    r"^-",                       # disabled runs
    r"^BAK_", r"_bak\.", r"_traj_\.pkl$",
    r"_v4\.1_", r"_v1\.0_",
    r"^UNet",                    # not in the paper
    r"^WNO_1scale_",             # paper's WNO uses 3 scales
    r"^(T?FNO)_rollout_",        # older FNO/TFNO rollouts (superseded by <m>_<pde>_rollout)
]
KINDS = [("_metrics.csv", "metrics"), ("_stats.csv", "stats"), ("_perf.yaml", "perf"), ("_traj.pkl", "traj")]


def paper_model(stem: str) -> str | None:
    for rx, rep in MODEL_RULES:
        if re.fullmatch(rx, stem):
            return re.sub(rx, rep, stem) if "\\" in rep else rep
    return None


def target(name: str, pde: str) -> tuple[str, str]:
    """('rename'|'obsolete'|'keep'|'unknown', new name)."""
    if name == f"ground_truth_{pde}_traj.pkl":
        return "keep", name
    if any(re.search(rx, name) for rx in OBSOLETE):
        return "obsolete", name
    kind = next((k for suf, k in KINDS if name.endswith(suf)), None)
    if kind is None:
        return "unknown", name
    s = name[: -len(next(suf for suf, k in KINDS if k == kind))]
    seed = re.search(r"_seed(\d+)", s)
    eta = re.search(r"_ic(0p\d+)", s)
    for rx in (r"_seed\d+", r"_ic0p\d+", r"_nsampling\d+", r"_nens\d+", r"_rollout", "_" + pde):
        s = re.sub(rx, "", s)
    model = paper_model(s)
    if model is None:
        return "unknown", name
    parts = [model, pde]
    if eta:
        parts.append(f"ic{eta.group(1)}")
    if seed and kind == "metrics":
        parts.append(f"seed{seed.group(1)}")
    suffix = {"metrics": "metrics.csv", "stats": "stats.csv", "perf": "perf.yaml", "traj": "traj.pkl"}[kind]
    return "rename", "_".join(parts) + "_" + suffix


def plan():
    moves, unknown = [], []
    for pde in PDES:
        for folder in FOLDERS:
            d = EXP / pde / folder
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if p.name.startswith(".") or p.name == "__pycache__":
                    continue
                if p.is_dir():          # e.g. results/OLD
                    moves.append((p, ROOT / "_to_delete" / "experiments_obsolete" / pde / folder / p.name, "obsolete"))
                    continue
                action, new = target(p.name, pde)
                if action == "rename" and new != p.name:
                    moves.append((p, d / new, "rename"))
                elif action == "obsolete":
                    moves.append((p, ROOT / "_to_delete" / "experiments_obsolete" / pde / folder / p.name, "obsolete"))
                elif action == "unknown":
                    unknown.append(p)
    dst = [m[1] for m in moves]
    clash = {str(x) for x in dst if dst.count(x) > 1}
    clash |= {str(n) for p, n, a in moves if n.exists() and n not in [m[0] for m in moves]}
    return moves, unknown, clash


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--revert", type=Path)
    a = ap.parse_args()

    if a.revert:
        rows = list(csv.DictReader(a.revert.open()))
        for r in reversed(rows):
            src, dst = ROOT / r["new"], ROOT / r["old"]
            if src.exists() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.rename(dst)
        print(f"reverted {len(rows)} moves from {a.revert}")
        return

    moves, unknown, clash = plan()
    for p, n, act in moves:
        print(f"{act:8s} {p.relative_to(ROOT)}  ->  {n.relative_to(ROOT)}")
    for p in unknown:
        print(f"UNKNOWN  {p.relative_to(ROOT)}  (left in place)")
    if clash:
        sys.exit(f"name clashes, nothing done: {sorted(clash)}")
    print(f"\n{sum(a == 'rename' for *_, a in moves)} renames, "
          f"{sum(a == 'obsolete' for *_, a in moves)} to _to_delete, {len(unknown)} unknown")
    if not a.apply:
        print("dry run; pass --apply to move the files")
        return
    man = ROOT / "artifacts" / "manifests" / f"rename_{dt.datetime.now():%Y%m%d_%H%M%S}.csv"
    man.parent.mkdir(parents=True, exist_ok=True)
    with man.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["old", "new", "action"])
        for p, n, act in moves:
            n.parent.mkdir(parents=True, exist_ok=True)
            p.rename(n)
            w.writerow([p.relative_to(ROOT), n.relative_to(ROOT), act])
    print(f"manifest: {man.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
