#!/usr/bin/env python3

"""
Debug script to verify parameter freezing and training setup for ViT-Adapter
"""

import torch
from models.dino_vit_adapter import DinoVisionTransformerAdapter

def debug_adapter_params():
    """Debug the parameter freezing in ViT-Adapter"""
    
    # Create a ViT-Adapter model
    model = DinoVisionTransformerAdapter(
        dinov2_vit_model='vits14',
        use_vit_adapter=True
  
    )
    
    print("=== Parameter Analysis ===")
    
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    adapter_params = 0
    backbone_params = 0
    
    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        if param.requires_grad:
            trainable_params += param.numel()
            print(f"  ✓ {name}: {param.numel()} params")
        else:
            frozen_params += param.numel()
            
        # Count adapter vs backbone parameters
        if any(x in name for x in ['spm', 'interactions', 'level_embed', 'up', 'norm']):
            adapter_params += param.numel()
        else:
            backbone_params += param.numel()
    
    print(f"\nFrozen parameters: {frozen_params}")
    print(f"Total DINOv2 blocks: {len(model.blocks)}")
    
    # Check which blocks are trainable
    trainable_blocks = []
    for name, param in model.named_parameters():
        if 'blocks.' in name and param.requires_grad:
            try:
                block_idx = int(name.split('.')[1])
                if block_idx not in trainable_blocks:
                    trainable_blocks.append(block_idx)
            except:
                pass
    
    print(f"Trainable blocks: {sorted(trainable_blocks)}")
    
    print(f"\n=== Summary ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    print(f"Adapter parameters: {adapter_params:,}")
    print(f"Backbone parameters: {backbone_params:,}")
    
    # Test forward pass
    print(f"\n=== Forward Pass Test ===")
    x = torch.randn(1, 3, 504, 1008)
    
    try:
        with torch.no_grad():
            output = model(x)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shapes: {[o.shape for o in output]}")
        
        # Check gradients
        x.requires_grad_(True)
        output = model(x)
        loss = sum(o.mean() for o in output)
        loss.backward()
        
        grad_params = sum(1 for p in model.parameters() if p.grad is not None)
        print(f"  Parameters with gradients: {grad_params}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        
    return model

if __name__ == "__main__":
    debug_adapter_params()
