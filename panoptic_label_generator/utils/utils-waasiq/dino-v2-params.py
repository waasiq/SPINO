import torch

# Load the original DinoV2 model
model = torch.load('dinov2_models/dinov2_vitb14_pretrain.pth', map_location='cpu')

print("=== ALL PARAMETERS IN ORIGINAL MODEL ===")
for k, v in model.items():
    print(f"{k}: {v.shape}")

print("\n=== LAYERSCALE-RELATED PARAMETERS ===")
for k, v in model.items():
    if 'gamma' in k or 'ls' in k:
        print(f"{k}: {v.shape}")

print("\n=== BLOCK STRUCTURE ANALYSIS ===")
block_params = {}
for k, v in model.items():
    if k.startswith('blocks.'):
        parts = k.split('.')
        if len(parts) >= 3:
            block_idx = parts[1]
            param_name = '.'.join(parts[2:])
            if block_idx not in block_params:
                block_params[block_idx] = []
            block_params[block_idx].append(param_name)

for block_idx in sorted(block_params.keys(), key=int):
    print(f"\nBlock {block_idx} parameters:")
    for param in sorted(block_params[block_idx]):
        print(f"  {param}")