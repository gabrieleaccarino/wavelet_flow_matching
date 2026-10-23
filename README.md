# waveletflux — Wavelet Flow Matching for Multi-Scale Physics Emulation

Code for *Wavelet Flow Matching for Multi-Scale Physics Emulation* (NeurIPS 2026).

## Layout

```
waveletflux/            package
  models/, modules/, wavelets/, utils/   model, training and inference code (torch)
  train_*.py, inference_*.py, eval_only.py   entry points used by scripts/*.sh (PBS)
  analysis/             numpy/pandas only — rebuilds every table of the paper
    catalog.py            model registry, PDE metadata, metric-file loading
    tables.py             table builders, across-seed mean ± std, LaTeX export
    physics.py            kinetic-energy diagnostic (rebuttal)
    ic_sensitivity.py     initial-condition perturbation study (rebuttal)
    profiling.py          inference speed / memory (Tables 2, S19, S20)
    paper_tables.py       one function per paper table: build, write tex/csv, return mean/std/n
    checks.py             rebuilds the submitted paper's tables (single run) cell by cell
configs/<pde>/          v4.0 training configs + inference YAMLs (`*_seed.yaml` = 3-seed runs)
scripts/                PBS job scripts + analysis entry points
  reproduce_tables.py     headless equivalent of notebooks/01_tables.ipynb
  check_paper_reproduction.py   CLI for analysis/checks.py
notebooks/
  01_tables.ipynb             builds + writes every table -> artifacts/tables/
  02_figures.ipynb            Figs 3a-c, S2-S4          -> artifacts/figures/
  03_qualitative_rollouts.ipynb  Figs 2a-b, S5          -> artifacts/figures/
  04_physics_consistency.ipynb   physics figures (rebuttal)
  05_cross_scale_coupling.ipynb  cross-scale sensitivity (needs GPU + checkpoint)
  06_profiling_tables.ipynb      Tables 2, S19, S20 (interactive)
  07_architecture_figure.ipynb   Fig 1 panels
  examples/, data_prep/
artifacts/              everything that goes into the paper (tracked in git)
  tables/tex, tables/csv, figures/, rebuttal/, checks/, manifests/
experiments/<pde>/      (not in git) ckpts/, metrics/, performance/
```

## Reproducing the tables

Run `notebooks/01_tables.ipynb` top to bottom: one cell per paper table, each
writing `artifacts/tables/tex/<name>.tex` and `csv/<name>.csv`; the last
section checks the builders against the submitted paper. Headless equivalent
(same functions, byte-identical output):

```bash
python scripts/reproduce_tables.py          # ~10 s, needs numpy, pandas, pyyaml
python scripts/check_paper_reproduction.py  # sanity check vs. the submitted paper
```
Set `WAVELETFLUX_ROOT=/path/to/checkout` if `experiments/` lives elsewhere.

**Metric files.** `experiments/<pde>/metrics/<model>_..._seed{42,84,168}_<pde>_metrics.csv`
(one per inference seed; all WFM variants and generative baselines) and
`<model>_..._<pde>_stats.csv` (single run; deterministic baselines FNO/TFNO/WNO).

**Aggregation.** Within a seed: mean over trajectories, fields and lead times;
across seeds: mean ± sample std. Spectral coherence "averaged over the rollout"
(Table 1, S9, S13) is horizon-balanced (mean of the three horizon means); the
physical-parameter tables (Table 3, S14–S18) average over all lead times.
Horizon bins are inclusive (e.g. RB: 1–32, 33–64, 65–196).

## Producing the metric files (cluster)

`scripts/train_<pde>.sh` → checkpoints; `scripts/inference_<pde>.sh` with
`configs/<pde>/inference_<pde>_seed.yaml` → `*_metrics.csv` and `*_perf.yaml`;
`scripts/inference_ic_pert.pbs` / `submit_ic_grid.sh` → IC-perturbation runs.
