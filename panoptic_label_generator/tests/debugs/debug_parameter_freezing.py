#!/usr/bin/env python3
"""
Debug script to check parameter freezing and model setup for ViT-Adapter
"""
import torch
import sys
import os
sys.path.append('/work/dlclarge2/masoodw-spino100/SPINO/panoptic_label_generator')

from models.dino_vit_adapter import ViTAdapter

def analyze_model_parameters(model, name="Model"):
    """Analyze which parameters are frozen vs trainable"""
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    print(f"\n=== {name} Parameter Analysis ===")
    
    # Group parameters by component
    component_stats = {}
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        # Determine component
        if 'blocks.' in name:
            component = 'DINOv2_blocks'
        elif 'patch_embed' in name:
            component = 'patch_embed'
        elif 'pos_embed' in name:
            component = 'pos_embed'
        elif 'spm' in name:
            component = 'spatial_prior_module'
        elif 'interactions' in name:
            component = 'interactions'
        elif 'level_embed' in name:
            component = 'level_embed'
        elif 'up' in name:
            component = 'upsample'
        elif 'norm' in name:
            component = 'norm_layers'
        else:
            component = 'other'
        
        if component not in component_stats:
            component_stats[component] = {'total': 0, 'trainable': 0, 'frozen': 0}
        
        component_stats[component]['total'] += param.numel()
        
        if param.requires_grad:
            trainable_params += param.numel()
            component_stats[component]['trainable'] += param.numel()
            print(f"  TRAINABLE: {name} - {param.numel():,} params")
        else:
            frozen_params += param.numel()
            component_stats[component]['frozen'] += param.numel()
            print(f"  FROZEN: {name} - {param.numel():,} params")
    
    print(f"\n=== Summary ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")
    
    print(f"\n=== Component Breakdown ===")
    for component, stats in component_stats.items():
        total_comp = stats['total']
        trainable_comp = stats['trainable']
        frozen_comp = stats['frozen']
        print(f"{component}:")
        print(f"  Total: {total_comp:,} params")
        print(f"  Trainable: {trainable_comp:,} ({100*trainable_comp/total_comp:.1f}%)")
        print(f"  Frozen: {frozen_comp:,} ({100*frozen_comp/total_comp:.1f}%)")
    
    return total_params, trainable_params, frozen_params

def test_forward_pass(model):
    """Test forward pass to make sure everything works"""
    print(f"\n=== Testing Forward Pass ===")
    
    # Create dummy input
    batch_size = 1
    input_tensor = torch.randn(batch_size, 3, 504, 1008)
    
    try:
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            print(f"Forward pass successful!")
            print(f"Input shape: {input_tensor.shape}")
            print(f"Output shapes: {[f'f{i+1}: {out.shape}' for i, out in enumerate(outputs)]}")
            
            # Check if outputs are reasonable
            for i, out in enumerate(outputs):
                print(f"f{i+1} stats: min={out.min():.3f}, max={out.max():.3f}, mean={out.mean():.3f}")
        
        return True
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return False

def main():
    print("=== ViT-Adapter Debug Analysis ===")
    
    # Create model
    model = ViTAdapter(
        img_size=(504, 1008),
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=torch.nn.LayerNorm,
        drop_path_rate=0.0,
        interaction_indexes=[[3, 5, 7], [6, 8, 10], [9, 11, 11]],
        add_vit_feature=True,
        pretrained_dinov2_path="/work/dlclarge2/masoodw-spino100/SPINO/checkpoints/dinov2_vits14_pretrain.pth"
    )
    
    # Analyze parameters
    total_params, trainable_params, frozen_params = analyze_model_parameters(model, "ViT-Adapter")
    
    # Test forward pass
    success = test_forward_pass(model)
    
    if success:
        print(f"\n=== Model is working correctly! ===")
        print(f"Key insights:")
        print(f"- {trainable_params:,} trainable parameters ({100*trainable_params/total_params:.1f}%)")
        print(f"- {frozen_params:,} frozen parameters ({100*frozen_params/total_params:.1f}%)")
        
        if trainable_params < 1000000:  # Less than 1M trainable params
            print("WARNING: Very few trainable parameters - might need to unfreeze more layers")
        elif trainable_params > 50000000:  # More than 50M trainable params
            print("WARNING: Many trainable parameters - might overfit with only 10 samples")
        else:
            print("✓ Reasonable number of trainable parameters")
    else:
        print("❌ Model has issues - check the implementation")

if __name__ == "__main__":
    main()
