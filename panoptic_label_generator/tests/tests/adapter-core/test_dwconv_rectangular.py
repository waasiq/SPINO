#!/usr/bin/env python3
"""
Test DWConv with rectangular inputs
"""

import torch
from models.vit_adapter.adapter_modules import DWConv

def test_dwconv_rectangular():
    print("Testing DWConv with rectangular inputs...")
    
    # Test with 504x1008 input dimensions
    height, width = 504, 1008
    patch_size = 14
    
    # Calculate patch dimensions
    H_patches = height // patch_size  # 36
    W_patches = width // patch_size   # 72
    N_patches = H_patches * W_patches  # 2592
    
    print(f"Input image: {height}x{width}")
    print(f"Patch dimensions: {H_patches}x{W_patches} = {N_patches} patches")
    
    # Test DWConv
    dwconv = DWConv(dim=192)
    
    # Create test input
    B = 1
    C = 192
    x = torch.randn(B, N_patches, C)
    
    # Test with adapter resolution (should be around height//16, width//16)
    H_adapter = height // 16  # ~31
    W_adapter = width // 16   # ~63
    
    print(f"Adapter resolution: {H_adapter}x{W_adapter}")
    print(f"Input tensor shape: {x.shape}")
    
    try:
        output = dwconv(x, H_adapter, W_adapter)
        print(f"✅ DWConv Success! Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"❌ DWConv Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dwconv_rectangular()
    print(f"Test {'PASSED' if success else 'FAILED'}")
