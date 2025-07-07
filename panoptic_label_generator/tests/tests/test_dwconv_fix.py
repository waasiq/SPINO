#!/usr/bin/env python3
"""
Test script to verify DWConv fix with rectangular inputs
"""

import torch
from models.vit_adapter.adapter_modules import DWConv

def test_dwconv_rectangular():
    print("Testing DWConv with rectangular input...")
    
    # Test configuration matching the error case
    # Image: [1, 3, 504, 1008] -> patches: (36, 72)
    # This means H=30, W=60 for the adapter (504//16 = 31.5 ≈ 30, 1008//16 = 63 ≈ 60)
    
    B = 1
    C = 192  # from the error
    H = 30   # from the error context
    W = 60   # approximate from 1008//16
    
    # Calculate expected input size for DWConv
    # The input comes from the extractor, so it should have the pattern:
    # 4*H*W + H*W + (H//2)*(W//2) tokens
    tokens_level1 = 4 * H * W   # 7200
    tokens_level2 = H * W       # 1800  
    tokens_level3 = (H // 2) * (W // 2)  # 450
    N = tokens_level1 + tokens_level2 + tokens_level3  # 9450
    
    print(f"Test parameters:")
    print(f"  B={B}, C={C}, H={H}, W={W}")
    print(f"  Expected N={N} (tokens_level1={tokens_level1}, tokens_level2={tokens_level2}, tokens_level3={tokens_level3})")
    
    # Create test input
    x = torch.randn(B, N, C)
    
    # Create DWConv module
    dwconv = DWConv(dim=C)
    
    try:
        with torch.no_grad():
            output = dwconv(x, H, W)
        print(f"✅ DWConv Success! Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"❌ DWConv Error: {e}")
        return False

def test_dwconv_square():
    print("\nTesting DWConv with square input (518x518)...")
    
    B = 1
    C = 192
    H = 32   # 518//16 ≈ 32
    W = 32
    
    tokens_level1 = 4 * H * W   
    tokens_level2 = H * W       
    tokens_level3 = (H // 2) * (W // 2)  
    N = tokens_level1 + tokens_level2 + tokens_level3  
    
    print(f"Test parameters:")
    print(f"  B={B}, C={C}, H={H}, W={W}")
    print(f"  Expected N={N}")
    
    x = torch.randn(B, N, C)
    dwconv = DWConv(dim=C)
    
    try:
        with torch.no_grad():
            output = dwconv(x, H, W)
        print(f"✅ DWConv Success! Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"❌ DWConv Error: {e}")
        return False

if __name__ == "__main__":
    success1 = test_dwconv_rectangular()
    success2 = test_dwconv_square()
    
    print(f"\n=== Test Summary ===")
    print(f"Rectangular input: {'✓ PASS' if success1 else '✗ FAIL'}")
    print(f"Square input: {'✓ PASS' if success2 else '✗ FAIL'}")
    print(f"Overall: {'PASSED' if success1 and success2 else 'FAILED'}")
