# Rollout animations

The project page shows these GIFs (exact names):

| File | Used in |
|---|---|
| `rayleigh_benard_rollout.gif` | teaser at the top + Rayleigh–Bénard tab |
| `turbulent_radiative_layer_2D_rollout.gif` | Turbulent Radiative Layer 2D tab |
| `active_matter_rollout.gif` | Active Matter tab |

Generate them with [`notebooks/07_rollout_animations.ipynb`](../../../notebooks/07_rollout_animations.ipynb), which
writes them to `artifacts/gifs/`, then copy them here. Until a file is present the page shows the static qualitative figure with a
"coming soon" label.

Keep each GIF well under GitHub's 100 MB limit (ideally < 25 MB so the page loads quickly): raise `stride`,
set `MAX_FRAMES` or lower `panel_width` in the notebook if one comes out too large.
