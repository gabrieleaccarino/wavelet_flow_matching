from the_well.data.datasets import TrajectoryData, TrajectoryMetadata
from the_well.data.augmentation import (
    Augmentation,
    Compose,
    RandomAxisFlip,
    RandomAxisPermute,
    RandomAxisRoll,
)

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
    

def compose_transformation(*names: str) -> Augmentation:
    """
    Build a deterministic transformation pipeline applied to ALL dataset splits.
    Only supports transformations that are safe to apply unconditionally.
    """
    registry = {
        "log_scalars": LogScalars,
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

#def compose_augmentation(*names: str) -> Augmentation:
#    augmentations = []
#
#    for name in names:
#        if name == "random_axis_flip":
#            augmentations.append(RandomAxisFlip())
#        elif name == "random_axis_permute":
#            augmentations.append(RandomAxisPermute())
#        elif name == "random_axis_roll":
#            augmentations.append(RandomAxisRoll())
#        elif name == "log_scalars":
#            augmentations.append(LogScalars())
#        else:
#            raise NotImplementedError()
#
#    return Compose(*augmentations)
