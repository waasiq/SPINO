# EVA02 model implementation for panoptic segmentation
# Adapted from DINOv2 structure

import torch
import torch.nn as nn
import timm
from typing import Optional, List

def _make_eva02_model(
    model_name: str = "eva02_base_patch14_224",
    pretrained: bool = True,
    img_size: int = 224,
    **kwargs
):
    """
    Create EVA02 model using timm
    Available models:
    - eva02_tiny_patch14_196
    - eva02_small_patch14_196  
    - eva02_base_patch14_224
    - eva02_large_patch14_224
    """
    model = timm.create_model(
        model_name, 
        pretrained=pretrained,
        **kwargs
    )
    
    # Add custom attributes to match DINOv2 interface
    model.patch_size = 14  # EVA02 models use 14x14 patches
    
    # Set num_features based on model variant
    if 'tiny' in model_name:
        model.num_features = 192
    elif 'small' in model_name:
        model.num_features = 384
    elif 'base' in model_name:
        model.num_features = 768
    elif 'large' in model_name:
        model.num_features = 1024
    else:
        # Fallback: get from model's embed_dim if available
        model.num_features = getattr(model, 'embed_dim', 768)
    
    return model

def eva02_tiny_patch14_196(*, pretrained: bool = True, **kwargs):
    """EVA02 Tiny model with patch size 14x14, input size 196x196"""
    return _make_eva02_model(
        model_name="eva02_tiny_patch14_196", 
        pretrained=pretrained, 
        img_size=196,
        **kwargs
    )

def eva02_small_patch14_196(*, pretrained: bool = True, **kwargs):
    """EVA02 Small model with patch size 14x14, input size 196x196"""
    return _make_eva02_model(
        model_name="eva02_small_patch14_196", 
        pretrained=pretrained, 
        img_size=196,
        **kwargs
    )

def eva02_base_patch14_224(*, pretrained: bool = True, **kwargs):
    """EVA02 Base model with patch size 14x14, input size 224x224"""
    return _make_eva02_model(
        model_name="eva02_base_patch14_224", 
        pretrained=pretrained, 
        img_size=224,
        **kwargs
    )

def eva02_large_patch14_224(*, pretrained: bool = True, **kwargs):
    """EVA02 Large model with patch size 14x14, input size 224x224"""
    return _make_eva02_model(
        model_name="eva02_large_patch14_224", 
        pretrained=pretrained, 
        img_size=224,
        **kwargs
    )

def eva02_large_patch14_336(*, pretrained: bool = True, **kwargs):
    """EVA02 Large model with patch size 14x14, input size 336x336"""
    return _make_eva02_model(
        model_name="eva02_large_patch14_336", 
        pretrained=pretrained, 
        img_size=336,
        **kwargs
    )

def eva02_large_patch14_448(*, pretrained: bool = True, **kwargs):
    """EVA02 Large model with patch size 14x14, input size 448x448"""
    return _make_eva02_model(
        model_name="eva02_large_patch14_448", 
        pretrained=pretrained, 
        img_size=448,
        **kwargs
    )

# Wrapper class to match DINOv2 interface
class EVA02Wrapper(nn.Module):
    def __init__(self, eva02_model):
        super().__init__()
        self.model = eva02_model
        self.num_features = eva02_model.num_features
        self.patch_size = eva02_model.patch_size
        
    def forward_features(self, x, return_attention_features=False, return_blocks=None):
        """
        Forward pass that returns features similar to DINOv2 interface
        """
        # EVA02 doesn't support attention features extraction like DINOv2
        if return_attention_features:
            raise NotImplementedError("EVA02 doesn't support attention feature extraction")
        
        # Get features from EVA02 by accessing intermediate layers
        # We need to manually extract features from the transformer blocks
        B, C, H, W = x.shape
        
        # Get patch embeddings similar to how DINOv2 works
        x_patch = self.model.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Add cls token
        cls_token = self.model.cls_token.expand(B, -1, -1)
        x_patch = torch.cat([cls_token, x_patch], dim=1)  # [B, num_patches+1, embed_dim]
        
        # Add positional embeddings
        if hasattr(self.model, 'pos_embed') and self.model.pos_embed is not None:
            x_patch = x_patch + self.model.pos_embed
        
        # Apply transformer blocks
        for block in self.model.blocks:
            x_patch = block(x_patch)
        
        # Apply final layer norm
        if hasattr(self.model, 'norm') and self.model.norm is not None:
            x_patch = self.model.norm(x_patch)
        
        return {'x': x_patch}
    
    def forward(self, x):
        return self.model(x)