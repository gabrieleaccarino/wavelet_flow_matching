from the_well.data.datasets import TrajectoryData, TrajectoryMetadata
from the_well.data.augmentation import (
    Augmentation,
    Compose,
    RandomAxisFlip,
    RandomAxisPermute,
    RandomAxisRoll,
)
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Deterministic transformations  (applied to all splits)
# ─────────────────────────────────────────────────────────────────────────────

class LogScalars(Augmentation):
    """
    Allows log-transforming constant values for some PDEs (e.g. Rayleigh and Prandtl numbers)

    Args:
        data (TrajectoryData): The input trajectory data.
        metadata (TrajectoryMetadata): The metadata associated with the trajectory data.

    Returns:
        TrajectoryData: constant scalars with log-transform applied to relevant keys.
    """
    _KEYS = {"rayleigh", "prandtl", "rho0", "t0"}
    
    def __call__(self, data: TrajectoryData, metadata: TrajectoryMetadata) -> TrajectoryData:
        scalars = data["constant_scalars"]

        for key, value in scalars.items():
            if key.lower() in self._KEYS:
                assert (value > 0).all(), (
                    f"LogScalars: '{key}' has non-positive values {value}. "
                    "Cannot take log."
                )
                scalars[key] = value.log()

        data["constant_scalars"] = scalars

        return data
    

class NormalizeActiveMatterScalars(Augmentation):
    """
    Normalizes Active Matter simulation constants before they enter
    the scalar conditioning pathway of the model.

    - zeta  (activity strength, always > 0):
        log-transform → {0, 1.61, 2.40, 2.83}
        Rationale: zeta spans nearly an order of magnitude {1,5,11,17};
        log makes the spacing more uniform and matches how physical
        energy injection scales.

    - alpha (flow type, always < 0):
        affine → [-1, +1] via (alpha + 3) / 2
        Rationale: alpha ∈ {-1,-2,-3,-4,-5}, center = -3, half-range = 2.
        Cannot log-transform (negative). Centering avoids the bias
        that would arise from passing values like -4,-5 raw into a
        linear layer alongside the O(1) time embedding.
    """

    def __call__(
        self, data: TrajectoryData, metadata: TrajectoryMetadata
    ) -> TrajectoryData:
        scalars = data["constant_scalars"]

        if "L" in scalars:
            # drop L since it's redundant with the spatial grid size
            del scalars["L"]  

        if "zeta" in scalars:
            v = scalars["zeta"]
            assert (v > 0).all(), (
                f"NormalizeActiveMatterScalars: 'zeta' has non-positive "
                f"values {v}. Cannot take log."
            )
            scalars["zeta"] = v.log()

        if "alpha" in scalars:
            # alpha ∈ {-1,-2,-3,-4,-5}  →  {+1, 0, -0.5, -1} ... wait
            # mean = -3, half-range = 2  →  maps to {+1, +0.5, 0, -0.5, -1}
            scalars["alpha"] = (scalars["alpha"] + 3.0) / 2.0

        #keys_to_use = ["zeta", "alpha"]  # drop L
        #label = torch.stack([scalars[k] for k in keys_to_use], dim=-1)
        data["constant_scalars"] = scalars
        return data
    

def compose_transformation(*names: str) -> Augmentation:
    """
    Build a deterministic transformation pipeline applied to ALL dataset splits.
    Only supports transformations that are safe to apply unconditionally.
    """
    registry = {
        "log_scalars": LogScalars,
        "normalize_active_matter_scalars": NormalizeActiveMatterScalars,
    }
    ops = []
    for name in names:
        if name not in registry:
            raise NotImplementedError(
                f"Unknown transformation '{name}'. "
                f"Valid options: {list(registry)}. "
                "Note: stochastic augmentations belong in compose_augmentation()."
            )
        ops.append(registry[name]())
    return Compose(*ops) if ops else None


# ─────────────────────────────────────────────────────────────────────────────
# Stochastic augmentations  (applied to TRAINING split only)
# ─────────────────────────────────────────────────────────────────────────────

def compose_augmentation(*names: str) -> Augmentation:
    """
    Build a stochastic augmentation pipeline applied to the training split only.
    Only supports augmentations that should NOT be applied at val/test time.
    """
    registry = {
        "random_axis_flip":    RandomAxisFlip,
        "random_axis_permute": RandomAxisPermute,
        "random_axis_roll":    RandomAxisRoll,
    }
    ops = []
    for name in names:
        if name not in registry:
            raise NotImplementedError(
                f"Unknown augmentation '{name}'. "
                f"Valid options: {list(registry)}. "
                "Note: deterministic transforms belong in compose_transformation()."
            )
        ops.append(registry[name]())
    return Compose(*ops) if ops else None
