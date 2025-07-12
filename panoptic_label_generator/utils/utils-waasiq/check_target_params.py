
from models import ViTAdapter
from models.vit.vision_transformer import DinoVisionTransformer  # Replace with your actual imports

# Create the target model WITHOUT loading the pretrained weights
target_model = ViTAdapter(
    conv_inplane=64, 
    n_points=4,
    deform_num_heads=6, 
    init_values=0.,
    interaction_indexes=[[0, 2, 4], [5, 7, 9], [10, 11, 11]],  # Standard interaction pattern
    with_cffn=True,
    cffn_ratio=0.25, 
    deform_ratio=1.0, 
    add_vit_feature=True,
    use_extra_extractor=True, 
    with_cp=False,
    vit_arch_name="vit_base",  # Use vit_base to match your converted model
    vit_pretrained=True  # Set to False to avoid loading weights
)

print("TARGET MODEL EXPECTS THESE PARAMETERS:")
print("="*60)
for name, param in target_model.named_parameters():
    print(f"{name}: {param.shape}")

print("\n" + "="*60)
print("TARGET MODEL STATE DICT KEYS:")
print("="*60)
for key in target_model.state_dict().keys():
    print(key)

# If you want to see only the 'blocks' related parameters:
print("\n" + "="*60)
print("ONLY 'blocks' PARAMETERS:")
print("="*60)
for name, param in target_model.named_parameters():
    if 'blocks' in name:
        print(f"{name}: {param.shape}")