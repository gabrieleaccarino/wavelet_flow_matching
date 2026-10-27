import torch.nn.functional as F 
from typing import Dict, Tuple
import torch.nn as nn
import numpy as np
import torch
import ptwt
import pywt

        


"""
Wavelet Utilities for Natural Multi-Scale Processing

This module handles wavelet decomposition/reconstruction using PyWavelets directly.
This ensures:
- Correct shape handling (exact /2 downsampling)
- Proper padding mode support
- Perfect reconstruction

Natural information flow:
- Forward: Fine → Coarse (progressive abstraction)
- Inverse: Coarse → Fine (progressive refinement)
"""



class WaveletDecomposer2D(nn.Module):
    """
    2D Wavelet Decomposition using PyWavelets backend.
    
    Forward decomposition proceeds from fine to coarse:
    - Level 0: H/2 × W/2 resolution
    - Level 1: H/4 × W/4 resolution
    - Level n: H/(2^n) × W/(2^n) resolution (coarsest)
    
    Each level produces 4 coefficient matrices:
    - LL (approximation): Low-freq both directions
    - LH: Low-freq horizontal, High-freq vertical
    - HL: High-freq horizontal, Low-freq vertical  
    - HH (diagonal): High-freq both directions
    
    Note: We decompose only the spatial dimensions (H, W), the last two.
    """
    
    def __init__(
        self, 
        wavelet: str = 'haar',
        n_scales: int = 3,
        mode: str = 'symmetric'
    ):
        """
        Args:
            wavelet: Wavelet family ('haar', 'db2', 'db4', 'sym4', etc.)
            n_scales: Number of decomposition levels
            mode: Padding mode ('zero', 'symmetric', 'reflect', 'periodic')
        """
        super().__init__()
        
        self.wavelet = wavelet
        self.n_scales = n_scales
        self.mode = mode
        
        # Validate wavelet
        if wavelet not in pywt.wavelist():
            raise ValueError(f"Unknown wavelet: {wavelet}. Available: {pywt.wavelist()}")
        
        # Validate mode
        valid_modes = ['zero', 'symmetric', 'reflect', 'periodic', 'constant', 'smooth', 'periodization']
        if mode not in valid_modes:
            raise ValueError(f"Unknown mode: {mode}. Available: {valid_modes}")
    
    def _dwt2d_single(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-level 2D DWT using PyWavelets on (H, W) dimensions.
        
        Args:
            x: Input tensor (B*C*T, H, W) as numpy array
            
        Returns:
            (LL, LH, HL, HH): Approximation and three detail coefficients
            Each has shape (B*C*T, H/2, W/2)
        """
        # Convert to numpy for pywt
        device = x.device
        dtype = x.dtype
        x_np = x.cpu().numpy()
        
        # Initialize output arrays
        batch_size = x_np.shape[0]
        
        # Process each item in batch
        coeffs_list = []
        for i in range(batch_size):
            # Apply 2D DWT using pywt
            coeffs = pywt.dwt2(x_np[i], self.wavelet, mode=self.mode)
            # coeffs = (LL, (LH, HL, HH))
            coeffs_list.append(coeffs)
        
        # Stack results
        LL = np.stack([c[0] for c in coeffs_list], axis=0)  # (B, H/2, W/2)
        LH = np.stack([c[1][0] for c in coeffs_list], axis=0)
        HL = np.stack([c[1][1] for c in coeffs_list], axis=0)
        HH = np.stack([c[1][2] for c in coeffs_list], axis=0)
        
        # Convert back to torch
        LL = torch.from_numpy(LL).to(device=device, dtype=dtype)
        LH = torch.from_numpy(LH).to(device=device, dtype=dtype)
        HL = torch.from_numpy(HL).to(device=device, dtype=dtype)
        HH = torch.from_numpy(HH).to(device=device, dtype=dtype)
        
        return LL, LH, HL, HH
    
    def forward_transform(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Multi-level wavelet decomposition (fine → coarse).
        
        Args:
            x: Input tensor (B, C, T, H, W)
            
        Returns:
            Dictionary mapping scale index to coefficients:
            {
                0: (B, C, T, 4, H/2, W/2),       # Finest details from level 0
                1: (B, C, T, 4, H/4, W/4),       # Details from level 1
                ...
                n-1: (B, C, T, 4, H/2^n, W/2^n), # Details from level n-1
                n: (B, C, T, 4, H/2^n, W/2^n)    # Coarsest (all 4 coeffs including LL)
            }
            
            Note: Index increases from fine (0) to coarse (n)
            Each scale has 4 coefficients in order: [LL, LH, HL, HH]
            For intermediate levels (0 to n-1): LL is a placeholder (will be replaced by next level)
            For final level (n): LL contains the actual coarsest approximation
        """
        if len(x.shape) == 5:
            B, C, T, H, W = x.shape
        elif len(x.shape) == 4:
            B, C, H, W = x.shape
            T = 1
            x = x.unsqueeze(2)  # Add dummy time dimension for uniform processing
        else:
            raise ValueError(f"Input tensor must be 4D or 5D. Got shape: {x.shape}")
        
        coeffs = {}
        
        # Process each (B, C, T) independently on (H, W)
        # Reshape to (B*C*T, H, W) for batch processing
        current = x.permute(0, 2, 1, 3, 4).reshape(B * T * C, H, W)  # (B*T*C, H, W)
        
        # Decompose from fine to coarse
        for level in range(self.n_scales):
            # Get current shape
            _, H_cur, W_cur = current.shape
            
            # Apply 2D DWT
            LL, LH, HL, HH = self._dwt2d_single(current)  # Each: (B*T*C, H/2, W/2)
            
            # Get new dimensions
            _, H_new, W_new = LL.shape
            
            # Reshape back to (B, C, T, H_new, W_new)
            LL = LL.reshape(B, T, C, H_new, W_new).permute(0, 2, 1, 3, 4)
            LH = LH.reshape(B, T, C, H_new, W_new).permute(0, 2, 1, 3, 4)
            HL = HL.reshape(B, T, C, H_new, W_new).permute(0, 2, 1, 3, 4)
            HH = HH.reshape(B, T, C, H_new, W_new).permute(0, 2, 1, 3, 4)
            
            # Stack: (B, C, T, 4, H_new, W_new) in order [LL, LH, HL, HH]
            level_coeffs = torch.stack([LL, LH, HL, HH], dim=3)
            
            # Store coefficients for this level
            coeffs[level] = level_coeffs
            
            # For next iteration, use LL (approximation)
            current = LL.permute(0, 2, 1, 3, 4).reshape(B * T * C, H_new, W_new)
        
        return coeffs
    
    def _idwt2d_single(
        self, 
        LL: torch.Tensor, 
        LH: torch.Tensor, 
        HL: torch.Tensor, 
        HH: torch.Tensor
    ) -> torch.Tensor:
        """
        Single-level inverse 2D DWT using PyWavelets.
        
        Args:
            LL, LH, HL, HH: Wavelet coefficients (B*C*T, H, W)
            
        Returns:
            Reconstructed signal (B*C*T, 2*H, 2*W)
        """
        # Convert to numpy
        device = LL.device
        dtype = LL.dtype
        
        LL_np = LL.cpu().numpy()
        LH_np = LH.cpu().numpy()
        HL_np = HL.cpu().numpy()
        HH_np = HH.cpu().numpy()
        
        batch_size = LL_np.shape[0]
        
        # Reconstruct each item in batch
        reconstructed = []
        for i in range(batch_size):
            # pywt.idwt2 expects (LL, (LH, HL, HH))
            coeffs = (LL_np[i], (LH_np[i], HL_np[i], HH_np[i]))
            rec = pywt.idwt2(coeffs, self.wavelet, mode=self.mode)
            reconstructed.append(rec)
        
        # Stack and convert back to torch
        x_rec = np.stack(reconstructed, axis=0)
        x_rec = torch.from_numpy(x_rec).to(device=device, dtype=dtype)
        
        return x_rec
    
    def inverse_transform(self, coeffs: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Multi-level wavelet reconstruction (coarse → fine).
        
        Args:
            coeffs: Dictionary from forward_transform
                Each value: (B, C, T, 4, H_scale, W_scale)
                where 4 coefficients are in order: [LL, LH, HL, HH]
            
        Returns:
            Reconstructed signal (B, C, T, H, W)
            
        Process:
            1. Start from coarsest level (highest index)
            2. Extract all 4 coefficients (LL, LH, HL, HH)
            3. Reconstruct to get approximation for next level
            4. Iteratively combine with details from finer scales
            5. Reconstruct progressively until original resolution
        """
        # Start from coarsest level (highest index)
        max_level = max(coeffs.keys())
        
        # Extract coarsest coefficients (B, C, T, 4, H_coarse, W_coarse)
        coarsest = coeffs[max_level]
        B, C, T, _, H_coarse, W_coarse = coarsest.shape
        
        # Extract LL, LH, HL, HH from coarsest level
        LL = coarsest[:, :, :, 0]  # (B, C, T, H_coarse, W_coarse)
        LH = coarsest[:, :, :, 1]
        HL = coarsest[:, :, :, 2]
        HH = coarsest[:, :, :, 3]
        
        # Reshape to (B*T*C, H_coarse, W_coarse) for batch IDWT
        LL_flat = LL.permute(0, 2, 1, 3, 4).reshape(B * T * C, H_coarse, W_coarse)
        LH_flat = LH.permute(0, 2, 1, 3, 4).reshape(B * T * C, H_coarse, W_coarse)
        HL_flat = HL.permute(0, 2, 1, 3, 4).reshape(B * T * C, H_coarse, W_coarse)
        HH_flat = HH.permute(0, 2, 1, 3, 4).reshape(B * T * C, H_coarse, W_coarse)
        
        # Reconstruct first level from coarsest
        current = self._idwt2d_single(LL_flat, LH_flat, HL_flat, HH_flat)  # (B*T*C, H_coarse*2, W_coarse*2)
        
        # Iteratively refine with finer details (coarse → fine)
        for level in range(max_level - 1, -1, -1):
            details = coeffs[level]  # (B, C, T, 4, H_level, W_level)
            
            # Extract detail coefficients (use LH, HL, HH; LL is the current approximation)
            # Note: coeffs[level][..., 0] is a placeholder or previous LL, we use 'current' instead
            LH_level = details[:, :, :, 1]  # (B, C, T, H_level, W_level)
            HL_level = details[:, :, :, 2]
            HH_level = details[:, :, :, 3]
            
            # Current should match the size of these detail coefficients
            _, H_curr, W_curr = current.shape
            _, _, _, H_level, W_level = LH_level.shape
            
            # Reshape current from (B*T*C, H, W) to (B, C, T, H, W)
            LL_curr = current.reshape(B, T, C, H_curr, W_curr).permute(0, 2, 1, 3, 4)
            
            # Verify shapes match
            #if LL_curr.shape[-2:] != (H_level, W_level):
            #    raise RuntimeError(
            #        f"Shape mismatch at level {level}: "
            #        f"current LL shape {LL_curr.shape[-2:]} != "
            #        f"detail shape {(H_level, W_level)}"
            #   )
            _, _, _, H_curr_new, W_curr_new = LL_curr.shape
            if (H_curr_new, W_curr_new) != (H_level, W_level):
                LL_curr = LL_curr[..., :H_level, :W_level]          # crop if too large
                pad_h = max(0, H_level - LL_curr.shape[-2])
                pad_w = max(0, W_level - LL_curr.shape[-1])
                if pad_h > 0 or pad_w > 0:                          # pad if too small
                    LL_curr = F.pad(LL_curr, (0, pad_w, 0, pad_h), mode='replicate')
            
            # Reshape all to (B*T*C, H_level, W_level) for batch IDWT
            LL_flat = LL_curr.permute(0, 2, 1, 3, 4).reshape(B * T * C, H_level, W_level)
            LH_flat = LH_level.permute(0, 2, 1, 3, 4).reshape(B * T * C, H_level, W_level)
            HL_flat = HL_level.permute(0, 2, 1, 3, 4).reshape(B * T * C, H_level, W_level)
            HH_flat = HH_level.permute(0, 2, 1, 3, 4).reshape(B * T * C, H_level, W_level)
            
            # Reconstruct next finer level
            current = self._idwt2d_single(LL_flat, LH_flat, HL_flat, HH_flat)
        
        # Reshape final result to (B, C, T, H, W)
        _, H_final, W_final = current.shape
        result = current.reshape(B, T, C, H_final, W_final).permute(0, 2, 1, 3, 4)
        
        return result




