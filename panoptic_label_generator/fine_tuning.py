from typing import List, Optional
from typing import Any, Dict 
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from models.dino_v2 import (
    dinov2_vitb14,
    dinov2_vitg14,
    dinov2_vitl14,
    dinov2_vits14,
)

from models.dino_vit_adapter import ViTAdapter

from torch import nn


class FineTuner(pl.LightningModule):
    def __init__(self, dinov2_vit_model: str, blocks: Optional[List[int]] = None,
                 upsample_factor: Optional[float] = None,
                 use_vit_adapter: bool = False, 
                 vit_adapter_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.dinov2_vit_model = dinov2_vit_model
        self.blocks = blocks
        self.upsample_factor = upsample_factor
        self.use_vit_adapter = use_vit_adapter
        self.vit_adapter_kwargs = vit_adapter_kwargs if vit_adapter_kwargs is not None else {}

        print(f"Using DINOv2 model: {self.dinov2_vit_model}, "
              f"blocks: {self.blocks}, "
              f"upsample_factor: {self.upsample_factor}, "
              f"use_vit_adapter: {self.use_vit_adapter}, "
              f"vit_adapter_kwargs: {self.vit_adapter_kwargs}")

        if self.use_vit_adapter:
            # Map DINOv2 model names to ViT configurations
            dinov2_configs = {
                'vits14': {
                    'vit_arch_name': 'vit_small',
                    'vit_kwargs': {
                        'patch_size': 14,
                        'embed_dim': 384,
                        'depth': 12,
                        'num_heads': 6,
                        'mlp_ratio': 4,
                        'qkv_bias': True,
                        'img_size': 518,
                    }
                },
                'vitb14': {
                    'vit_arch_name': 'vit_base',
                    'vit_kwargs': {
                        'patch_size': 14,
                        'embed_dim': 768,
                        'depth': 12,
                        'num_heads': 12,
                        'mlp_ratio': 4,
                        'qkv_bias': True,
                        'img_size': 518,
                    }
                },
                'vitl14': {
                    'vit_arch_name': 'vit_large',
                    'vit_kwargs': {
                        'patch_size': 14,
                        'embed_dim': 1024,
                        'depth': 24,
                        'num_heads': 16,
                        'mlp_ratio': 4,
                        'qkv_bias': True,
                        'img_size': 518,
                    }
                },
                'vitg14': {
                    'vit_arch_name': 'vit_giant2',
                    'vit_kwargs': {
                        'patch_size': 14,
                        'embed_dim': 1536,
                        'depth': 40,
                        'num_heads': 24,
                        'mlp_ratio': 4,
                        'qkv_bias': True,
                        'img_size': 518,
                        'ffn_layer': 'swiglufused',
                    }
                }
            }
            
            # Get configuration for the specified model
            if self.dinov2_vit_model not in dinov2_configs:
                raise ValueError(f'Unknown DINOv2 model: {self.dinov2_vit_model}')
            
            model_config = dinov2_configs[self.dinov2_vit_model]
            
            # Set up default ViT adapter configuration
            default_adapter_kwargs = {
                'pretrain_size': 518,
                'init_values': 1.0,
                'interaction_indexes': [[0, 2], [3, 5], [6, 8], [9, 11]],  # Default interaction indexes
                'vit_pretrained': True,
                **model_config  # Use the model-specific configuration
            }

            # Update with user-provided kwargs
            default_adapter_kwargs.update(self.vit_adapter_kwargs)
            
            # Create the ViT adapter
            self.encoder = ViTAdapter(**default_adapter_kwargs)
            self.feat_dim = self.encoder.embed_dim
            self.patch_size = self.encoder.patch_size

            print(f"Using ViT-Adapter with model: {self.dinov2_vit_model}, "
                  f"architecture: {default_adapter_kwargs['vit_arch_name']}, "
                  f"featdim: {self.feat_dim}, "
                  f"patch_size: {self.patch_size}")

            # Freeze the ViT-Adapter encoder
            for param in self.encoder.parameters():
                param.requires_grad = False
            
        else: # Default DINOv2 model
            if dinov2_vit_model == 'vits14':
                self.encoder = dinov2_vits14(pretrained=True)
            elif dinov2_vit_model == 'vitb14':
                self.encoder = dinov2_vitb14(pretrained=True)
            elif dinov2_vit_model == 'vitl14':
                self.encoder = dinov2_vitl14(pretrained=True)
            elif dinov2_vit_model == 'vitg14':
                self.encoder = dinov2_vitg14(pretrained=True)
            else:
                raise ValueError(f'Unknown model {dinov2_vit_model}')

            self.feat_dim = self.encoder.num_features
            self.patch_size = self.encoder.patch_size
            self.encoder.mask_token = None  # can't use ddp_find_unused_parameters_false otherwise

            for param in self.encoder.parameters():  # freeze backbone
                param.requires_grad = False

        if blocks is None:
            self.num_blocks = 1
        else:
            self.num_blocks = len(blocks)

    def forward_encoder(self, img: torch.Tensor, feature_key: str = 'x'):
        img_h, img_w = img.shape[2:]
        patches_h, patches_w = img_h // self.patch_size, img_w // self.patch_size

        # For debug purposes
        # print("Forwarding encoder with image size:", img.shape,
        #       "patch size:", self.patch_size,
        #       "patches (H, W):", (patches_h, patches_w))

        if self.use_vit_adapter:
            features = self.encoder(img) 

            if self.upsample_factor is not None:
                network_output_size = (int(patches_h * self.upsample_factor), int(patches_w * self.upsample_factor))
                encoder_output_feature = F.interpolate(features[0], size=network_output_size, mode='bilinear', align_corners=False)
            else:
                encoder_output_feature = features[1] # Default to 1/8 resolution
            
            return encoder_output_feature 
        else: 
            return_attention_features = any([(feature_key in x) for x in ['q', 'k', 'v', 'attn']])
            with torch.no_grad():
                block_outputs = self.encoder.forward_features(
                    img,
                    return_attention_features=return_attention_features,
                    return_blocks=self.blocks)
                if self.blocks is None:
                    block_outputs = [block_outputs]
                outs = []
                for x in block_outputs:
                    x = x[feature_key]
                    if feature_key == 'attn':
                        return x  # (B, num_heads, Patches+1, Patches+1)
                    if feature_key in ['q', 'k', 'v']:
                        # (B, Patches+1, num_heads, feat_dim // num_heads)
                        x = x.permute((0, 2, 1, 3)).contiguous()
                        x = x.reshape((x.shape[0], -1, self.feat_dim))  # (B, Patches+1, feat_dim)
                    outs.append(x)
                x = torch.cat(outs, dim=2)  # (B, Patches+1, feat_dim * self.num_blocks)

                x = x[:, 1:, :]  # (B, Patches, feat_dim)
                x = x.permute((0, 2, 1)).contiguous()  # (B, feat_dim, H*W)
                x = x.reshape((x.shape[0], self.feat_dim * self.num_blocks, patches_h,
                            patches_w))  # (B, feat_dim, H, W)
                if self.upsample_factor is not None:
                    x = nn.functional.interpolate(x, scale_factor=self.upsample_factor, mode='bilinear',
                                                align_corners=False)  # (B, feat_dim, H, W)
            return x