"""DWT round-trip tests (moved out of waveletflux/wavelets/DWT2D.py). Needs torch + pytorch_wavelets.
Run on the cluster env: python -m pytest tests/test_dwt2d.py"""
import pytest
torch = pytest.importorskip("torch")
from waveletflux.wavelets.DWT2D import WaveletDecomposer2D


def test_wavelet_reconstruction():
    """Test perfect reconstruction property."""
    print("Testing Wavelet Reconstruction...")
    print("-" * 70)
    
    decomposer = WaveletDecomposer2D(wavelet='haar', n_scales=3, mode='symmetric')
    
    # Create random input (use even dimensions for clean /2 division)
    B, C, T, H, W = 2, 4, 8, 64, 64
    x = torch.randn(B, C, T, H, W)
    
    print(f"Input shape: {x.shape}")
    
    # Forward decomposition
    coeffs = decomposer.forward_transform(x)
    
    print(f"\nDecomposed into {len(coeffs)} scales:")
    for level, coef in coeffs.items():
        print(f"  Level {level}: {coef.shape}")
        expected_h = H // (2 ** (level + 1))
        expected_w = W // (2 ** (level + 1))
        actual_h, actual_w = coef.shape[-2:]
        
        # Verify shape is correct (H/2^(level+1), W/2^(level+1))
        print(f"    Expected spatial: ({expected_h}, {expected_w}), Actual: ({actual_h}, {actual_w})", end="")
        if actual_h == expected_h and actual_w == expected_w:
            print(" ✓")
        else:
            print(" ✗ MISMATCH!")
            return False
    
    # Inverse reconstruction
    print("\nReconstructing...")
    x_rec = decomposer.inverse_transform(coeffs)
    
    print(f"Reconstructed shape: {x_rec.shape}")
    
    # Check shapes match
    if x.shape != x_rec.shape:
        print(f"✗ Shape mismatch: {x.shape} vs {x_rec.shape}")
        return False
    
    # Check reconstruction error
    error = torch.abs(x - x_rec).max().item()
    mean_error = torch.abs(x - x_rec).mean().item()
    
    print(f"\nReconstruction error:")
    print(f"  Max error: {error:.2e}")
    print(f"  Mean error: {mean_error:.2e}")
    
    # For Haar wavelet with pywt, perfect reconstruction should have very small error
    if error < 1e-5:
        print("\n✓ Perfect reconstruction verified!")
        return True
    else:
        print(f"\n✗ Reconstruction error too large: {error:.2e}")
        return False


def test_different_input_sizes():
    """Test with various input sizes."""
    print("\n" + "=" * 70)
    print("Testing Different Input Sizes")
    print("=" * 70)
    
    decomposer = WaveletDecomposer2D(wavelet='haar', n_scales=2, mode='symmetric')
    
    test_sizes = [
        (2, 4, 8, 32, 32),
        (2, 4, 8, 64, 64),
        (2, 4, 8, 128, 128),
        (1, 1, 4, 64, 64),
    ]
    
    all_passed = True
    
    for size in test_sizes:
        B, C, T, H, W = size
        print(f"\nTesting size: (B={B}, C={C}, T={T}, H={H}, W={W})")
        
        x = torch.randn(B, C, T, H, W)
        coeffs = decomposer.forward_transform(x)
        x_rec = decomposer.inverse_transform(coeffs)
        
        error = torch.abs(x - x_rec).max().item()
        
        if x.shape == x_rec.shape and error < 1e-5:
            print(f"  ✓ Passed (error: {error:.2e})")
        else:
            print(f"  ✗ Failed (error: {error:.2e}, shape match: {x.shape == x_rec.shape})")
            all_passed = False
    
    return all_passed