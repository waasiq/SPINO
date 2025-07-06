#!/usr/bin/env python3
"""
Debug script to understand the tensor shape issues
"""

import torch
from models.vit_adapter.adapter_modules import deform_inputs, get_reference_points

def debug_deform_inputs():
    print("Debug deform_inputs...")
    
    # Test with same dimensions as in the error
    batch_size = 1
    channels = 3
    height = 518
    width = 518
    
    x = torch.randn(batch_size, channels, height, width)
    
    print(f"Input shape: {x.shape}")
    
    # Get deform inputs
    deform_inputs1, deform_inputs2 = deform_inputs(x)
    
    print(f"deform_inputs1:")
    print(f"  reference_points shape: {deform_inputs1[0].shape}")
    print(f"  spatial_shapes: {deform_inputs1[1]}")
    print(f"  level_start_index: {deform_inputs1[2]}")
    
    print(f"deform_inputs2:")
    print(f"  reference_points shape: {deform_inputs2[0].shape}")
    print(f"  spatial_shapes: {deform_inputs2[1]}")
    print(f"  level_start_index: {deform_inputs2[2]}")
    
    # Check assertion that's failing
    bs, c, h, w = x.shape
    
    # For deform_inputs1
    spatial_shapes1 = deform_inputs1[1]
    total_spatial_1 = (spatial_shapes1[:, 0] * spatial_shapes1[:, 1]).sum().item()
    print(f"deform_inputs1 total spatial: {total_spatial_1}")
    
    # For deform_inputs2
    spatial_shapes2 = deform_inputs2[1]
    total_spatial_2 = (spatial_shapes2[:, 0] * spatial_shapes2[:, 1]).sum().item()
    print(f"deform_inputs2 total spatial: {total_spatial_2}")
    
    # Let's see what the reference points expect
    ref_points_1 = deform_inputs1[0]
    ref_points_2 = deform_inputs2[0]
    print(f"Reference points 1 shape: {ref_points_1.shape}")
    print(f"Reference points 2 shape: {ref_points_2.shape}")

if __name__ == "__main__":
    debug_deform_inputs()
