#!/usr/bin/env python3
"""
Test script to debug DWConv issues
"""

import torch
from models.vit_adapter.adapter_modules import DWConv

def test_dwconv():
    print("Testing DWConv with different input sizes...")
    
    # Test case 1: Square input (518x518)
    print("\n=== Test 1: Square input (518x518) ===")
    H, W = 30, 60  # This would be 518//14 ≈ 37, but let's use adapter dimensions
    B, C = 1, 192
    N = 4*H*W + H*W + (H//2)*(W//2)  # Expected total
    print(f"Expected N for H={H}, W={W}: {N}")
    
    x = torch.randn(B, N, C)
    dwconv = DWConv(C)
    
    try:
        output = dwconv(x, H, W)
        print(f"✅ Success! Input: {x.shape}, Output: {output.shape}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test case 2: Rectangular input (504x1008) 
    print("\n=== Test 2: Rectangular input (504x1008) ===")
    H, W = 30, 60  # Adapter dimensions (not patch dimensions)
    B, C = 1, 192
    N = 4*H*W + H*W + (H//2)*(W//2)  # Expected total
    print(f"Expected N for H={H}, W={W}: {N}")
    
    x = torch.randn(B, N, C)
    dwconv = DWConv(C)
    
    try:
        output = dwconv(x, H, W)
        print(f"✅ Success! Input: {x.shape}, Output: {output.shape}")
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test case 3: What we actually get from the error
    print("\n=== Test 3: Actual error case ===")
    N = 10466  # From the error
    H, W = 30, 60  # Try different H, W values
    B, C = 1, 192
    
    expected = 4*H*W + H*W + (H//2)*(W//2)
    print(f"N={N}, Expected for H={H}, W={W}: {expected}")
    
    x = torch.randn(B, N, C)
    dwconv = DWConv(C)
    
    try:
        output = dwconv(x, H, W)
        print(f"✅ Success! Input: {x.shape}, Output: {output.shape}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        
        # Try to find working H, W
        print("Trying to find working H, W values...")
        for h_test in [25, 30, 35, 40]:
            for w_test in [50, 55, 60, 65, 70]:
                expected_test = 4*h_test*w_test + h_test*w_test + (h_test//2)*(w_test//2)
                if abs(expected_test - N) < 100:  # Close enough
                    print(f"  Trying H={h_test}, W={w_test}, expected={expected_test}")
                    try:
                        output_test = dwconv(x, h_test, w_test)
                        print(f"  ✅ Found working values: H={h_test}, W={w_test}")
                        return
                    except:
                        continue

if __name__ == "__main__":
    test_dwconv()
