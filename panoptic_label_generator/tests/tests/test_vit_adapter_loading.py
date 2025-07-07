#!/usr/bin/env python3
"""
Test script to verify ViT adapter loading with proper DINOv2 configuration
"""

import torch
from models.dino_vit_adapter import ViTAdapter

def test_vit_adapter_loading():
    print("Testing ViT adapter loading with DINOv2 ViT-B configuration...")
    
    # Test ViT-B configuration (same as in boundary_cityscapes.yaml)
    vit_kwargs = {
        'patch_size': 14,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4,
        'qkv_bias': True,
        'img_size': 518,
    }
    
    try:
        # Create ViT adapter with ViT-B configuration
        adapter = ViTAdapter(
            pretrain_size=518,
            conv_inplane=64,
            n_points=4,
            deform_num_heads=6,
            init_values=1.0,
            interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
            with_cffn=True,
            cffn_ratio=0.25,
            deform_ratio=1.0,
            add_vit_feature=True,
            use_extra_extractor=True,
            with_cp=False,
            vit_arch_name="vit_base",
            vit_kwargs=vit_kwargs,
            vit_pretrained=True  # This should work now
        )
        
        print(f"✅ ViT adapter created successfully!")
        print(f"   - Embed dim: {adapter.embed_dim}")
        print(f"   - Patch size: {adapter.patch_size}")
        print(f"   - Pos embed shape: {adapter.pos_embed.shape}")
        print(f"   - Number of blocks: {len(adapter.blocks)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating ViT adapter: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vit_adapter_loading()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
