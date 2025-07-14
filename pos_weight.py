import torch
import numpy as np
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.neighbors import radius_neighbors_graph
import sys
import os

# Add the panoptic_label_generator to the Python path
sys.path.append('/work/dlclarge2/masoodw-spino100/dl-lab/SPINO/panoptic_label_generator')

from datasets.cityscapes import CityscapesDataModule
from utils.transforms import ToTensor, ImageNormalize, MaskPostProcess
from yacs.config import CfgNode as CN

# Setup dataset configuration
cfg_dataset = CN()
cfg_dataset.name = "cityscapes"
cfg_dataset.path = "/work/dlclarge2/masoodw-spino100/dl-lab/cityscapes"
cfg_dataset.feed_img_size = [1024, 2048]
cfg_dataset.offsets = [0]
cfg_dataset.remove_classes = []

# Create minimal transforms for pos_weight calculation
minimal_transforms = [
    ToTensor(),
    MaskPostProcess(),
    ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

# Create dataset
cityscapes_dm = CityscapesDataModule(
    cfg_dataset=cfg_dataset,
    num_classes=19,
    batch_size=1,
    num_workers=1,
    transform_train=minimal_transforms,
    transform_test=minimal_transforms,
    label_mode="cityscapes_19",
    train_sample_indices=None,  # Use all training samples
    test_sample_indices=None
)

cityscapes_dm.setup('fit')
train_dataloader = cityscapes_dm.train_dataloader()

total_boundary_pixels = 0
total_non_boundary_pixels = 0

# Configuration from boundary_cityscapes.yaml
patch_size = 14  # DINOv2 patch size
upsample_factor = 3.5  # From your config
neighbor_radius = 1.5  # From your config  
num_boundary_neighbors = 1  # From your config

print("Starting pos_weight calculation...")
print(f"Using dataset at: {cfg_dataset.path}")

# Set model to eval mode if it's within a LightningModule, or just ensure no gradients are computed
with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Calculating pos_weight")):
        # Get data from batch
        rgb = batch['rgb']
        ins = batch['instance'].long()  # (B, H_orig, W_orig)

        # --- Replicate the ins_boundary generation logic from training_step ---
        batch_size = rgb.shape[0]
        rgb_h, rgb_w = rgb.shape[2:]  # Get original image dimensions
        
        patches_h, patches_w = rgb_h // patch_size, rgb_w // patch_size
        network_output_size = (int(patches_h * upsample_factor), int(patches_w * upsample_factor))

        # Resize instance mask to network output size
        ins_resized = TF.resize(ins, size=[network_output_size[0], network_output_size[1]],
                                interpolation=TF.InterpolationMode.NEAREST)  # (B, H_resized, W_resized)
        ins_flattened = ins_resized.view(ins_resized.shape[0], -1)  # (B, H_resized * W_resized)

        # Generate connected indices for the current H,W
        h_resized, w_resized = network_output_size
        coordinates = np.indices((h_resized, w_resized)).reshape(2, -1).T
        graph = radius_neighbors_graph(coordinates, radius=neighbor_radius,
                                       mode='connectivity', include_self=False)
        connected_indices_np = np.argwhere(graph == 1)  # (I, 2)
        connected_indices = torch.from_numpy(connected_indices_np).long()

        # --- Boundary GT generation for direct mode ---
        # Check if connected pixels have different instance IDs (boundary)
        is_boundary_pair = (ins_flattened[:, connected_indices[:, 0]] !=
                            ins_flattened[:, connected_indices[:, 1]]).cpu().numpy().astype(int)  # (B, I)

        # Tile connected_indices for batch
        connected_indices_tiled = np.tile(connected_indices_np, (ins_resized.shape[0], 1, 1))  # (B, I, 2)
        indices = connected_indices_tiled[:, :, 0]  # (B, I)

        # Aggregate boundary counts per pixel
        pixel_boundary_counts = np.add.reduceat(is_boundary_pair,
                                                 np.unique(indices, return_index=True, axis=1)[1],
                                                 axis=1)  # (B, H_resized * W_resized)

        # Apply the num_boundary_neighbors threshold
        current_ins_boundary_map = (pixel_boundary_counts >= num_boundary_neighbors).astype(int)  # (B, H*W)

        # Count pixels
        total_boundary_pixels += np.sum(current_ins_boundary_map == 1)
        total_non_boundary_pixels += np.sum(current_ins_boundary_map == 0)

# Calculate pos_weight
if total_boundary_pixels == 0:
    print("Warning: No boundary pixels found in the dataset. pos_weight will be set to 1.0 (or a very high number).")
    pos_weight_value = 1.0 # Or a very large number if you expect boundaries but none found
else:
    pos_weight_value = total_non_boundary_pixels / total_boundary_pixels

print(f"Total Boundary Pixels: {total_boundary_pixels}")
print(f"Total Non-Boundary Pixels: {total_non_boundary_pixels}")
print(f"Calculated pos_weight: {pos_weight_value}")

# Write results to file
output_file = "pos_weight_results.txt"
with open(output_file, 'w') as f:
    f.write(f"Pos_weight calculation results for Cityscapes boundary detection\n")
    f.write(f"Dataset path: {cfg_dataset.path}\n")
    f.write(f"Configuration:\n")
    f.write(f"  - patch_size: {patch_size}\n")
    f.write(f"  - upsample_factor: {upsample_factor}\n")
    f.write(f"  - neighbor_radius: {neighbor_radius}\n")
    f.write(f"  - num_boundary_neighbors: {num_boundary_neighbors}\n")
    f.write(f"\nResults:\n")
    f.write(f"  - Total Boundary Pixels: {total_boundary_pixels}\n")
    f.write(f"  - Total Non-Boundary Pixels: {total_non_boundary_pixels}\n")
    f.write(f"  - Calculated pos_weight: {pos_weight_value}\n")
    f.write(f"\nRecommended BCEWithLogitsLoss configuration:\n")
    f.write(f"criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor({pos_weight_value}))\n")

print(f"\nResults written to: {output_file}")
print(f"Use this pos_weight value in your loss function: {pos_weight_value}")