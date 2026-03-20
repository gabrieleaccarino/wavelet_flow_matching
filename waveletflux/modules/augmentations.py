from the_well.data.datasets import TrajectoryData, TrajectoryMetadata
from the_well.data.augmentation import (
    Augmentation,
    Compose,
    RandomAxisFlip,
    RandomAxisPermute,
    RandomAxisRoll,
)

class LogScalars(Augmentation):
    """
    Allows log-transforming constant values for some PDEs (e.g. Rayleigh and Prandtl numbers)

    Args:
        data (TrajectoryData): The input trajectory data.
        metadata (TrajectoryMetadata): The metadata associated with the trajectory data.

    Returns:
        TrajectoryData: constant scalars with log-transform applied to relevant keys.
    """
    
    def __call__(self, data: TrajectoryData, metadata: TrajectoryMetadata) -> TrajectoryData:
        scalars = data["constant_scalars"]

        for key, value in scalars.items():
            if key.lower() in ("rayleigh", "prandtl", "rho0", "T0"):
                scalars[key] = value.log()

        data["constant_scalars"] = scalars

        return data
    

def compose_augmentation(*names: str) -> Augmentation:
    augmentations = []

    for name in names:
        if name == "random_axis_flip":
            augmentations.append(RandomAxisFlip())
        elif name == "random_axis_permute":
            augmentations.append(RandomAxisPermute())
        elif name == "random_axis_roll":
            augmentations.append(RandomAxisRoll())
        elif name == "log_scalars":
            augmentations.append(LogScalars())
        else:
            raise NotImplementedError()

    return Compose(*augmentations)
