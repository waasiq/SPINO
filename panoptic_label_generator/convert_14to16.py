import argparse
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('filename', nargs='?', type=str, default='dinov2_models/dinov2_vitb14_pretrain.pth')
parser.add_argument('--debug', action='store_true', help='Print expected vs actual parameter names')

args = parser.parse_args()

model = torch.load(args.filename, map_location=torch.device('cpu'))

print("Original model parameters:")
for k, v in model.items():
    print(f"{k}: {v.shape}")
print("\n" + "="*50 + "\n")

# resize patch embedding from 14x14 to 16x16
patch_embed = model['patch_embed.proj.weight']
patch_embed = F.interpolate(patch_embed, size=(16, 16), mode='bilinear', align_corners=False)
model['patch_embed.proj.weight'] = patch_embed

# Resize position embedding for 512x1024 input with 16x16 patches
# 512/16 = 32, 1024/16 = 64, so we need 32x64 + 1 = 2049 positions
if 'pos_embed' in model:
    pos_embed = model['pos_embed']  # [1, 1370, 768] from DinoV2
    if args.debug:
        print(f"Original pos_embed shape: {pos_embed.shape}")
    
    # Extract class token (first position)
    cls_pos = pos_embed[:, 0:1, :]  # [1, 1, 768]
    
    # Extract spatial positions and reshape for interpolation
    spatial_pos = pos_embed[:, 1:, :]  # [1, 1369, 768] 
    # 1369 = 37*37 (DinoV2 uses 37x37 grid for 14x14 patches on 518x518 images)
    spatial_pos = spatial_pos.reshape(1, 37, 37, 768).permute(0, 3, 1, 2)  # [1, 768, 37, 37]
    
    # Interpolate to 32x64 grid for 512x1024 input with 16x16 patches
    # Height: 512/16 = 32, Width: 1024/16 = 64
    spatial_pos = F.interpolate(spatial_pos, size=(32, 64), mode='bilinear', align_corners=False)
    spatial_pos = spatial_pos.permute(0, 2, 3, 1).reshape(1, 2048, 768)  # [1, 2048, 768]
    
    # Concatenate class token and spatial positions
    new_pos_embed = torch.cat([cls_pos, spatial_pos], dim=1)  # [1, 2049, 768]
    model['pos_embed'] = new_pos_embed
    
    if args.debug:
        print(f"New pos_embed shape: {new_pos_embed.shape}")

# Filter out LayerScale parameters and modify block structure
new_model = {}
for k, v in model.items():
    # Skip LayerScale parameters (ls1.gamma, ls2.gamma) - target model doesn't have them
    if '.ls1.gamma' in k or '.ls2.gamma' in k:
        if args.debug:
            print(f"Skipping LayerScale parameter: {k}")
        continue
    
    # Keep mask_token as target model expects it
    # (Don't skip it like we do with LayerScale parameters)
    
    new_k = k  # Keep original parameter name
    
    # Add block structure modification - target model expects blocks.0.X.parameter
    if new_k.startswith('blocks.'):
        parts = new_k.split('.')
        if len(parts) >= 3:  # blocks.X.parameter
            block_idx = parts[1]  # e.g., '1' from 'blocks.1.norm1'
            remaining_path = '.'.join(parts[2:])  # e.g., 'norm1.weight'
            new_k = f'blocks.0.{block_idx}.{remaining_path}'
    
    if args.debug:
        if new_k != k:
            print(f"Renaming: {k} -> {new_k}")
    
    new_model[new_k] = v


# DON'T add level_embed or any adapter parameters!
# Only convert what exists in the original DinoV2 model
# Let the ViT Adapter model initialize its own adapter parameters properly

print("Converted model parameters:")
for k, v in new_model.items():
    print(f"{k}: {v.shape}")

print(f"\n{'='*50}")
print("CONVERTED MODEL SUMMARY:")
print(f"{'='*50}")
print(f"Total parameters in converted model: {len(new_model)}")
print(f"✅ ViT backbone parameters: Successfully converted from DinoV2")
print(f"❌ Adapter parameters: NOT included (will use model's own initialization)")

torch.save(new_model, args.filename.replace('.pth', '_14to16.pth'))
print(f"\nModel saved to: {args.filename.replace('.pth', '_14to16.pth')}")

print(f"\n{'='*50}")
print("LOADING INSTRUCTIONS:")
print(f"{'='*50}")
print("1. Load with strict=False:")
print("   model.load_state_dict(torch.load('dinov2_models/dinov2_vitb14_pretrain_14to16.pth'), strict=False)")
print("2. Only ViT backbone will be loaded from DinoV2 pretrained weights")
print("3. Adapter parameters will use their original initialization:")
print("   - Linear layers: Truncated normal (std=0.02) for weights, zero for bias")
print("   - Conv2d/ConvTranspose2d: He normal initialization")
print("   - BatchNorm/LayerNorm: weight=1.0, bias=0.0")
print("   - level_embed: Normal distribution")
print("   - MSDeformAttn: Uses its own _reset_parameters() method")
print("4. This preserves the designed initialization scheme from ViT-Adapter authors")
print("5. Fine-tune the entire model if needed")