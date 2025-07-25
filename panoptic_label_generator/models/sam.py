import torch
from segment_anything import sam_model_registry

# Load the model
def get_sam_model():
    model_type = "vit_b"  # Or "vit_l", "vit_h" depending on your model path
    model_path = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    
    # Instantiate the SAM model architecture
    sam = sam_model_registry[model_type](checkpoint=None)
    
    # Load the state dictionary from the URL
    state_dict = torch.hub.load_state_dict_from_url(model_path)
    
    # Load the state dictionary into the instantiated model
    # It's good practice to ensure all keys match, but for freezing the encoder
    # we specifically care about the image_encoder part.
    sam.load_state_dict(state_dict)
    
    # Return the image encoder module
    return sam.image_encoder


get_sam_model()  # Call to ensure the model is loaded and parameters are printed
# Example usage:
# your_instance = YourClass(model_name='sam')


# get_sam_model()

# model_path = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
# checkpoint = torch.hub.load_state_dict_from_url(model_path)

# # Extract and print only image_encoder layers
# image_encoder_layers = {k: v for k, v in checkpoint.items() if k.startswith('image_encoder')}

# # Print image_encoder layer names
# for key in image_encoder_layers.keys():
#     print(key)
