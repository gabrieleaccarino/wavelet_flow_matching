"""Pre-trained checkpoints on the Hugging Face Hub.

All models evaluated in the paper are hosted at
https://huggingface.co/gabrieleaccarino/wavelet-flow-matching, one folder per
``<pde>/<model>/``. Model names match the paper and the metric files in
``experiments/<pde>/metrics/``.

Example
-------
>>> from wavelet_flow_matching.hub import list_models, load_pretrained
>>> list_models("rayleigh_benard")
['WFM_1scale_haar', ..., 'LatentFM']
>>> model = load_pretrained("rayleigh_benard", "WFM_1scale_db4", map_location="cuda")

``download_checkpoint`` only downloads and returns the local file path, e.g. to
use it as ``ckpt_path`` in an inference YAML config.
"""
from __future__ import annotations

REPO_ID = "gabrieleaccarino/wavelet-flow-matching"

PDES = ("turbulent_radiative_layer_2D", "rayleigh_benard", "active_matter")

_WFM = [f"WFM_{s}_{w}" for s in ("1scale", "3scale") for w in ("haar", "db2", "db4", "db6")]
_FLOW_BASELINES = ["FM_pixel", "FM_fourier", "FM_fourier16"]
_NEURAL_OPS = ["FNO", "TFNO"]

MODELS = {
    pde: _WFM + _FLOW_BASELINES + _NEURAL_OPS + (["LatentFM"] if pde == "rayleigh_benard" else [])
    for pde in PDES
}

# Autoencoder used by LatentFM (Rozet et al., "Lost in Latent Space"; see the model card).
AUTOENCODER_FILE = "rayleigh_benard/LatentFM/autoencoder_dcae_f32c64.pth"


def list_models(pde: str | None = None):
    """Model names available for one PDE, or the full {pde: [models]} mapping."""
    if pde is None:
        return {k: list(v) for k, v in MODELS.items()}
    _check(pde)
    return list(MODELS[pde])


def checkpoint_file(pde: str, model: str) -> str:
    """Path of a checkpoint inside the Hub repository."""
    _check(pde, model)
    ext = ".pt" if model in _NEURAL_OPS else ".ckpt"
    return f"{pde}/{model}/{model}_{pde}{ext}"


def download_checkpoint(
    pde: str,
    model: str,
    repo_id: str = REPO_ID,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> str:
    """Download one checkpoint (cached after the first call) and return its local path."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id, checkpoint_file(pde, model), revision=revision, cache_dir=cache_dir)


def load_pretrained(
    pde: str,
    model: str,
    map_location: str = "cpu",
    repo_id: str = REPO_ID,
    revision: str | None = None,
    cache_dir: str | None = None,
    ae_state_path: str | None = None,
):
    """Download a checkpoint and return the model in eval mode.

    Returns a ``WaveletFlowModule`` (WFM_*, FM_pixel), ``FourierFlowModule``
    (FM_fourier*), ``LatentFlowModule`` (LatentFM) or ``NeuralOperatorModule``
    (FNO, TFNO). For LatentFM the autoencoder weights are downloaded too, unless
    ``ae_state_path`` points to a local copy.
    """
    path = download_checkpoint(pde, model, repo_id, revision, cache_dir)

    if model in _NEURAL_OPS:
        from wavelet_flow_matching.modules.neural_ops import NeuralOperatorModule
        return NeuralOperatorModule.from_pretrained(path, device=map_location)

    if model.startswith("FM_fourier"):
        from wavelet_flow_matching.modules.fourier_fm import FourierFlowModule as cls
    elif model == "LatentFM":
        from wavelet_flow_matching.modules.latent_fm import LatentFlowModule
        if ae_state_path is None:
            from huggingface_hub import hf_hub_download
            ae_state_path = hf_hub_download(repo_id, AUTOENCODER_FILE, revision=revision, cache_dir=cache_dir)
        net = LatentFlowModule.from_checkpoint(path, map_location=map_location, ae_state_path=ae_state_path)
        return net.to(map_location).eval()
    else:
        from wavelet_flow_matching.modules.wavelet_fm import WaveletFlowModule as cls

    return cls.from_checkpoint(path, map_location=map_location).to(map_location).eval()


def _check(pde: str, model: str | None = None) -> None:
    if pde not in MODELS:
        raise ValueError(f"Unknown pde {pde!r}. Choose from {list(MODELS)}.")
    if model is not None and model not in MODELS[pde]:
        raise ValueError(f"Unknown model {model!r} for {pde}. Choose from {MODELS[pde]}.")
