import os
import torch

# Base URL for the model
_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"

# Folder to store the downloaded model
output_folder = "dinov2_models"
os.makedirs(output_folder, exist_ok=True)

# Path to save the model
model_path = os.path.join(output_folder, "dinov2_vitb14_pretrain.pth")

# Download the model using torch.hub
if not os.path.exists(model_path):
    print("Downloading model...")
    torch.hub.download_url_to_file(_DINOV2_BASE_URL, model_path)
    print(f"Model saved to {model_path}")
else:
    print(f"Model already exists at {model_path}")