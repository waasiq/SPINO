import torch
from segment_anything import sam_model_registry

def get_sam_model():
    model_type = "vit_b"
    model_path = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    
    # Instantiate the SAM model architecture
    sam = sam_model_registry[model_type](checkpoint=None)
    
    # Load the state dictionary from the URL
    state_dict = torch.hub.load_state_dict_from_url(model_path)
    sam.load_state_dict(state_dict)
    
    return sam.image_encoder