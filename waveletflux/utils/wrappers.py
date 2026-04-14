import torch.nn as nn
import torch

class TFNOMPSSafeWrapper(nn.Module):
    """
    Wraps a TFNO (or FNO) model to work around MPS kernel crashes.

    The crash originates in neuralop's SpectralConv layers, which use
    torch.fft and complex tensor contractions that are not fully supported
    on the MPS backend. This wrapper moves the tensor to CPU for the
    forward pass and returns the result to the original device.

    Usage:
        fno = TFNO(...)                         # build normally
        fno.load_state_dict(...)                # load checkpoint normally
        fno = TFNOMPSSafeWrapper(fno).to(device)   # wrap before inference
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.cpu()   # keep weights on CPU permanently

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        origin_device = x.device
        y = self.model(x.cpu())    # run entirely on CPU
        return y.to(origin_device) # return to MPS / caller's device


def make_mps_safe(model: nn.Module) -> nn.Module:
    """
    Convenience helper: wraps the model only when running on MPS.
    On CUDA or CPU it returns the model unchanged.
    """
    if torch.backends.mps.is_available() and next(model.parameters()).device.type == "mps":
        print("MPS detected — wrapping model with TFNOMPSSafeWrapper (FFT ops run on CPU).")
        return TFNOMPSSafeWrapper(model)
    return model