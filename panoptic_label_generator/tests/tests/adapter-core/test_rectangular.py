#!/usr/bin/env python3
"""
Test script to verify ViT adapter handles rectangular inputs
"""

import torch
from models.dino_vit_adapter import ViTAdapter

def test_vit_adapter_rectangular():
    print("Testing ViT adapter with rectangular input (504x1008)...")
    
    # Test configuration for rectangular input
    vit_kwargs = {
        'patch_size': 16,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4,
        'qkv_bias': True,
        'img_size': 518,  # Note: this is different from actual input size
    }
    
    # Create ViT adapter
    adapter = ViTAdapter(
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
        vit_pretrained=False  # Set to False for testing
    )
    
    # Test with rectangular input (504x1008)
    batch_size = 1
    channels = 3
    height = 504
    width = 1008
    
    x = torch.randn(batch_size, channels, height, width)
    
    print(f"Input shape: {x.shape}")
    print(f"Patches: {height//14} x {width//14} = {(height//14) * (width//14)}")
    
    # Test SPM only first
    try:
        with torch.no_grad():
            c1, c2, c3, c4 = adapter.spm(x)
        print(f"SPM Success!")
        print(f"  c1: {c1.shape}")
        print(f"  c2: {c2.shape} (tokens: {c2.shape[1]})")
        print(f"  c3: {c3.shape} (tokens: {c3.shape[1]})")
        print(f"  c4: {c4.shape} (tokens: {c4.shape[1]})")
        
        # Test deform inputs
        from models.vit_adapter.adapter_modules import deform_inputs
        deform_inputs1, deform_inputs2 = deform_inputs(x)
        print(f"Deform inputs calculated successfully")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vit_adapter_rectangular()
    print(f"Test: {'PASSED' if success else 'FAILED'}")
