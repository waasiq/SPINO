#!/usr/bin/env python3
"""
Debug script to understand the feature tensor shapes
"""

import torch
from models.dino_vit_adapter import ViTAdapter

def debug_feature_shapes():
    print("Debug feature shapes...")
    
    # Test configuration - simplified to match test_vit_adapter.py
    vit_kwargs = {
        'patch_size': 14,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4,
        'qkv_bias': True,
        'img_size': 518,
    }
    
    # Create ViT adapter but with minimal interaction to avoid the error
    adapter = ViTAdapter(
        pretrain_size=518,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        init_values=0.0,
        interaction_indexes=[[0, 2]],  # Just one interaction to debug
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
    
    # Test input
    x = torch.randn(1, 3, 518, 518)
    
    # Forward through SPM only
    with torch.no_grad():
        c1, c2, c3, c4 = adapter.spm(x)
        print(f"SPM outputs:")
        print(f"  c1 shape: {c1.shape}")
        print(f"  c2 shape: {c2.shape}")  
        print(f"  c3 shape: {c3.shape}")
        print(f"  c4 shape: {c4.shape}")
        
        # After level embed
        c2, c3, c4 = adapter._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)
        print(f"  concatenated c shape: {c.shape}")

if __name__ == "__main__":
    debug_feature_shapes()
