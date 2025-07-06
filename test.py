import torch

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
    print("Number of CUDA devices:", torch.cuda.device_count())
else:
    print("CUDA is not available.")
