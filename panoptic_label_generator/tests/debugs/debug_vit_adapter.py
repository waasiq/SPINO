#!/usr/bin/env python3
"""
Debug script to understand the shape mismatches in ViT adapter
"""

import torch
from models.vit_adapter.adapter_modules import deform_inputs

def debug_deform_inputs():
    print("Debugging deform_inputs function...")
    
    # Test input
    x = torch.randn(1, 3, 518, 518)
    print(f"Input shape: {x.shape}")
    
    deform_inputs1, deform_inputs2 = deform_inputs(x)
    
    print("\nDeform inputs 1:")
    print(f"Reference points shape: {deform_inputs1[0].shape}")
    print(f"Spatial shapes: {deform_inputs1[1]}")
    print(f"Level start index: {deform_inputs1[2]}")
    
    print("\nDeform inputs 2:")
    print(f"Reference points shape: {deform_inputs2[0].shape}")
    print(f"Spatial shapes: {deform_inputs2[1]}")
    print(f"Level start index: {deform_inputs2[2]}")
    
    # Calculate expected spatial dimensions
    h, w = 518, 518
    print(f"\nCalculated spatial dimensions:")
    print(f"h // 8 = {h // 8}, w // 8 = {w // 8}")
    print(f"h // 16 = {h // 16}, w // 16 = {w // 16}")
    print(f"h // 32 = {h // 32}, w // 32 = {w // 32}")
    print(f"h // 14 = {h // 14}, w // 14 = {w // 14}")
    
    # Calculate total spatial points
    spatial_shapes_1 = deform_inputs1[1]
    total_points_1 = (spatial_shapes_1[:, 0] * spatial_shapes_1[:, 1]).sum()
    print(f"\nTotal spatial points for deform_inputs1: {total_points_1}")
    
    spatial_shapes_2 = deform_inputs2[1]
    total_points_2 = (spatial_shapes_2[:, 0] * spatial_shapes_2[:, 1]).sum()
    print(f"Total spatial points for deform_inputs2: {total_points_2}")

if __name__ == "__main__":
    debug_deform_inputs()
