"""
Example usage of ViTAdapter in panoptic_label_generator

This shows how to integrate the ViTAdapter into your fine-tuning classes.
"""

from models.dino_vit_adapter import ViTAdapter

# Example configuration for ViTAdapter
def create_vit_adapter(arch_name="vit_small", patch_size=14, use_checkpoint=True):
    """Create a ViTAdapter model with checkpointing enabled"""
    
    # Default interaction indexes for different depths
    interaction_indexes = [[0, 4], [4, 8], [8, 12]] if arch_name == "vit_base" else [[0, 2], [2, 4], [4, 6]]
    
    vit_kwargs = {
        'patch_size': patch_size,
        'embed_dim': 768 if arch_name == "vit_base" else 384,
        'depth': 12 if arch_name == "vit_base" else 6,
        'num_heads': 12 if arch_name == "vit_base" else 6,
        'mlp_ratio': 4,
        'block_chunks': 0,
    }
    
    model = ViTAdapter(
        pretrain_size=518,  # Common DINOv2 pretrain size
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        init_values=0.,
        interaction_indexes=interaction_indexes,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        add_vit_feature=True,
        use_extra_extractor=True,
        with_cp=False,  # This is for internal adapter checkpointing
        use_checkpoint=use_checkpoint,  # This is for our gradient checkpointing
        vit_arch_name=arch_name,
        vit_kwargs=vit_kwargs,
        vit_pretrained=True
    )
    
    return model

# Example integration in fine-tuning classes:
"""
In your SemanticFineTuner, BoundaryFineTuner, etc., you can now use:

from models.dino_vit_adapter import ViTAdapter

class SemanticFineTuner(pl.LightningModule):
    def __init__(self, dinov2_vit_model="vits14", use_checkpoint=False, **kwargs):
        super().__init__()
        
        # Use ViTAdapter instead of basic DinoVisionTransformer
        self.encoder = ViTAdapter(
            vit_arch_name="vit_small" if "vits" in dinov2_vit_model else "vit_base",
            vit_kwargs={'patch_size': 14},
            use_checkpoint=use_checkpoint,
            # ... other parameters
        )
        
        # The rest of your model initialization...
"""
