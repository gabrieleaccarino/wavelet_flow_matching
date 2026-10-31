# Rollout animations

Drop the WFM rollout animations here with these exact names and the page picks them up automatically:

| File | Used in |
|---|---|
| `rayleigh_benard_rollout.mp4` | hero video + Rayleigh–Bénard tab |
| `turbulent_radiative_layer_2D_rollout.mp4` | Turbulent Radiative Layer 2D tab |
| `active_matter_rollout.mp4` | Active Matter tab |

Until a file is present, the page shows the qualitative figure as a poster with a "coming soon" label.

## Make them web-friendly

Browsers need H.264 + `yuv420p`; `+faststart` lets playback start before the whole file downloads.
GitHub rejects files over 100 MB, and pages load best below ~10 MB per video.

```bash
ffmpeg -i input.mp4 -c:v libx264 -preset veryfast -crf 26 -pix_fmt yuv420p \
       -vf "scale='min(1600,iw)':-2" -movflags +faststart -an rayleigh_benard_rollout.mp4
```

Raise `-crf` (e.g. 28–30) to shrink the file further.
