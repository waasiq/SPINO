#!/usr/bin/env python3
"""Debug the shape error in ViTAdapter."""

import torch
import torch.nn.functional as F
import sys
import os
sys.path.append('/work/dlclarge2/masoodw-spino100/SPINO/panoptic_label_generator')

from models.dino_vit_adapter import ViTAdapter

def debug_shape_error():
    """Debug the shape error step by step."""
    print("Debugging ViTAdapter shape error...")
    
    # Create model with correct constructor
    vit_kwargs = {
        'patch_size': 14,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4,
        'qkv_bias': True,
    }
    
    model = ViTAdapter(
        pretrain_size=518,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        init_values=0.0,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        add_vit_feature=True,
        use_extra_extractor=True,
        with_cp=False,
        vit_arch_name="vit_base",
        vit_kwargs=vit_kwargs,
        vit_pretrained=False
    )
    
    model.eval()
    
    # Test with rectangular image
    print("Testing with rectangular image (504x1008)...")
    x = torch.randn(1, 3, 504, 1008)
    
    # Let's add some debug prints to understand the shapes
    print(f"Input shape: {x.shape}")
    print(f"Input H: {x.shape[2]}, W: {x.shape[3]}")
    print(f"H_adapter: {x.shape[2] // 16}, W_adapter: {x.shape[3] // 16}")
    
    # Calculate expected ViT patch dimensions
    patch_size = 14
    H_vit = x.shape[2] // patch_size
    W_vit = x.shape[3] // patch_size
    print(f"H_vit: {H_vit}, W_vit: {W_vit}")
    print(f"Expected ViT patches: {H_vit * W_vit}")
    
    # Test the individual components
    try:
        # Try to run just the initial parts
        from models.vit_adapter.adapter_modules import deform_inputs
        
        deform_inputs1, deform_inputs2 = deform_inputs(x)
        print(f"deform_inputs1 shapes: {[d.shape for d in deform_inputs1]}")
        print(f"deform_inputs2 shapes: {[d.shape for d in deform_inputs2]}")
        
        # Test SPM
        c1, c2, c3, c4 = model.spm(x)
        print(f"SPM outputs - c1: {c1.shape}, c2: {c2.shape}, c3: {c3.shape}, c4: {c4.shape}")
        
        # Calculate token counts
        bs = x.shape[0]
        print(f"Token counts - c2: {c2.shape[1]}, c3: {c3.shape[1]}, c4: {c4.shape[1]}")
        
        # Now test the factorization
        H_adapter = x.shape[2] // 16
        W_adapter = x.shape[3] // 16
        print(f"H_adapter: {H_adapter}, W_adapter: {W_adapter}")
        
        # Test factorization for each level
        print("\nTesting factorization:")
        
        # c2 (highest resolution)
        n_c2 = c2.shape[1]
        target_h2, target_w2 = H_adapter * 2, W_adapter * 2
        h2_actual, w2_actual = model._factorize_tokens(n_c2, target_h2, target_w2)
        print(f"c2: {n_c2} tokens, target {target_h2}x{target_w2}, actual {h2_actual}x{w2_actual}")
        print(f"c2 check: {h2_actual * w2_actual} == {n_c2}? {h2_actual * w2_actual == n_c2}")
        
        # c3 (base resolution)
        n_c3 = c3.shape[1]
        target_h3, target_w3 = H_adapter, W_adapter
        h3_actual, w3_actual = model._factorize_tokens(n_c3, target_h3, target_w3)
        print(f"c3: {n_c3} tokens, target {target_h3}x{target_w3}, actual {h3_actual}x{w3_actual}")
        print(f"c3 check: {h3_actual * w3_actual} == {n_c3}? {h3_actual * w3_actual == n_c3}")
        
        # c4 (lowest resolution)
        n_c4 = c4.shape[1]
        target_h4, target_w4 = H_adapter // 2, W_adapter // 2
        h4_actual, w4_actual = model._factorize_tokens(n_c4, target_h4, target_w4)
        print(f"c4: {n_c4} tokens, target {target_h4}x{target_w4}, actual {h4_actual}x{w4_actual}")
        print(f"c4 check: {h4_actual * w4_actual} == {n_c4}? {h4_actual * w4_actual == n_c4}")
        
    except Exception as e:
        print(f"Error in debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_shape_error()
