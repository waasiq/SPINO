#!/usr/bin/env python3
"""
Test the factorization fix for ViT adapter with rectangular inputs
"""

import torch
import math

def _factorize_tokens(n_tokens, target_h, target_w):
    """Find the best factorization of n_tokens into height x width"""
    target_ratio = target_w / target_h if target_h > 0 else 1.0
    best_h, best_w = 1, n_tokens
    min_diff = float('inf')
    
    for h in range(1, int(math.sqrt(n_tokens)) + 1):
        if n_tokens % h == 0:
            w = n_tokens // h
            ratio = w / h
            diff = abs(ratio - target_ratio)
            if diff < min_diff:
                min_diff = diff
                best_h, best_w = h, w
    
    return best_h, best_w

def test_factorization():
    print("Testing factorization fix...")
    
    # Test case from the error: 504x1008 input
    H_adapter = 31  # 504/16 rounded
    W_adapter = 63  # 1008/16 rounded
    
    # Test c2 tokens (from the error: 6096384 / 768 = 7938 tokens)
    n_c2 = 7938
    target_h2 = H_adapter * 2  # 62
    target_w2 = W_adapter * 2  # 126
    
    h2, w2 = _factorize_tokens(n_c2, target_h2, target_w2)
    print(f"c2: {n_c2} tokens -> {h2}x{w2} = {h2*w2} tokens")
    print(f"   Target: {target_h2}x{target_w2} = {target_h2*target_w2}")
    print(f"   Match: {h2*w2 == n_c2}")
    
    # Test some other cases
    test_cases = [
        (4225, 31, 63),  # c2 from SPM
        (1089, 15, 31),  # c3 from SPM  
        (289, 7, 15),    # c4 from SPM
    ]
    
    for n_tokens, target_h, target_w in test_cases:
        h, w = _factorize_tokens(n_tokens, target_h, target_w)
        print(f"{n_tokens} tokens -> {h}x{w} = {h*w} tokens (target: {target_h}x{target_w})")
        print(f"   Match: {h*w == n_tokens}")
    
    print("Test completed!")

if __name__ == "__main__":
    test_factorization()
