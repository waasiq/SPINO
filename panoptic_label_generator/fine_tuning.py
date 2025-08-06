from typing import List, Optional

import pytorch_lightning as pl
import torch
import loralib as lora

from models.dino_v2 import (
    dinov2_vitb14,
    dinov2_vitg14,
    dinov2_vitl14,
    dinov2_vits14,
)

from models.eva02 import (
    eva02_tiny_patch14_196,
    eva02_small_patch14_196,
    eva02_base_patch14_224,
    eva02_large_patch14_224,
    eva02_large_patch14_336,
    eva02_large_patch14_448,
    EVA02Wrapper
)

from models.sam import get_sam_model 
from torch import nn
import torch.nn.functional as F
from models.vit_adapter.vit_adapter import ViTAdapter
from models.vit_comer.vit_comer import ViTCoMer

class FineTuner(pl.LightningModule):
    def __init__(self, model: str, blocks: Optional[List[int]] = None,
                 upsample_factor: Optional[float] = None, use_vitadapter: bool = False, 
                 use_vitcomer: bool = False, use_lora: bool = False):
        super().__init__()
        self.model = model
        self.blocks = blocks
        self.upsample_factor = upsample_factor
        self.use_vitadapter = use_vitadapter
        self.use_vitcomer = use_vitcomer
        self.use_lora = use_lora

        if use_vitadapter and use_vitcomer:
            raise ValueError("Cannot use both ViTAdapter and ViTCoMer at the same time. Choose one.")

        dinov2_models = {
            'vits14': (dinov2_vits14, 'ViT-S14'),
            'vitb14': (dinov2_vitb14, 'ViT-B14'),
            'vitl14': (dinov2_vitl14, 'ViT-L14'),
            'vitg14': (dinov2_vitg14, 'ViT-G14')
        }

        eva02_models = {
            'tiny_patch14_196': (eva02_tiny_patch14_196, 'EVA02-Tiny-14'),
            'small_patch14_196': (eva02_small_patch14_196, 'EVA02-Small-14'),
            'base_patch14_224': (eva02_base_patch14_224, 'EVA02-Base-14'),
            'large_patch14_224': (eva02_large_patch14_224, 'EVA02-Large-14'),
            'large_patch14_336': (eva02_large_patch14_336, 'EVA02-Large-14-336'),
            'large_patch14_448': (eva02_large_patch14_448, 'EVA02-Large-14-448')
        }

        if self.use_vitadapter: 
            self.encoder = ViTAdapter()
            self.encoder_type = 'custom'
            print('[ENCODER] Using encoder: ViTAdapter')
        elif self.use_vitcomer:
            self.encoder = ViTCoMer()
            self.encoder_type = 'custom'
            print('[ENCODER] Using encoder: ViTCoMeR')
        elif model in dinov2_models:
            model_fn, name = dinov2_models[model]
            self.encoder = model_fn(pretrained=True)
            self.encoder_type = 'dinov2'
            print(f'[ENCODER] Using encoder: {name}')
        elif model in eva02_models:
            model_fn, name = eva02_models[model]
            base_model = model_fn(pretrained=True)
            self.encoder = EVA02Wrapper(base_model)
            self.encoder_type = 'eva02'
            print(f'[ENCODER] Using encoder: {name}')
        elif model == 'sam':
            self.encoder = get_sam_model()
            self.encoder_type = 'sam'
            print('[ENCODER] Using encoder: SAM')
        else:
            raise ValueError(f'Unknown model')

        # Freeze the encoder if not using adapter, comer, or lora
        if not self.use_vitadapter and not self.use_vitcomer and not self.use_lora:
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif self.use_lora:
            self.lora_layers = nn.ModuleDict()
            apply_lora(self.encoder, self.lora_layers)
            #sets requires_grad to False for all parameters without the string "lora_" in their names
            lora.mark_only_lora_as_trainable(self.encoder)
            assert any('lora' in name.lower() for name, _ in self.named_parameters()), 'LoRA layers not found!'
            print('LoRA enabled')

        if self.encoder_type == 'sam':
            self.feat_dim = self.encoder.neck[0].out_channels 
            self.patch_size = 16
        else:
            self.feat_dim = self.encoder.num_features 
            self.patch_size = self.encoder.patch_size
        self.encoder.mask_token = None  # can't use ddp_find_unused_parameters_false otherwise

        if blocks is None:
            self.num_blocks = 1
        else:
            self.num_blocks = len(blocks)

    def forward_encoder(self, img: torch.Tensor, feature_key: str = 'x'):
        img_h, img_w = img.shape[2:]
        patches_h, patches_w = img_h // self.patch_size, img_w // self.patch_size

        return_attention_features = any([(feature_key in x) for x in ['q', 'k', 'v', 'attn']])

        # Logic for ViTAdapter and ViTCoMer
        if self.encoder_type == 'custom':
            f1, f2, f3, f4 = self.encoder.forward(img)
            _, _, h_f1, w_f1 = f1.shape

            # Upsample f2, f3, f4 to the same resolution as f1
            # Assuming f1, f2, f3, f4 have the same 'dim' (channels)
            f2_upsampled = F.interpolate(f2, size=(h_f1, w_f1), mode='bilinear', align_corners=False)
            f3_upsampled = F.interpolate(f3, size=(h_f1, w_f1), mode='bilinear', align_corners=False)
            f4_upsampled = F.interpolate(f4, size=(h_f1, w_f1), mode='bilinear', align_corners=False)

            x = torch.cat([f1, f2_upsampled, f3_upsampled, f4_upsampled], dim=1)
            return x
        elif self.encoder_type == 'sam':
            features = self.encoder(img)  # Shape: [B, embed_dim, 64, 64]
            
            # Convert spatial features to token format for consistency with other ViTs
            B, C, H, W = features.shape
            # Flatten spatial dimensions: [B, C, H*W] -> [B, H*W, C]
            features_tokens = features.flatten(2).transpose(1, 2)  # [B, H*W, C]
            # Create block output with both 'token' and 'x' keys for compatibility
            block_outputs = [{'token': features_tokens, 'x': features_tokens}] 

            if self.blocks is None:
                block_outputs = [block_outputs[-1]]  # Take the last layer
                
            outs = []
            for block_output in block_outputs:
                if isinstance(block_output, dict):
                    x = block_output[feature_key]  # Extract the requested feature type
                else:
                    x = block_output  # If it's direct tensor output
                    
                if feature_key == 'attn':
                    if return_attention_features:
                        return x
                    else:
                        raise NotImplementedError("SAM attention extraction not implemented")
                        
                if feature_key in ['q', 'k', 'v']:
                    raise NotImplementedError("SAM doesn't support q/k/v feature extraction in this implementation")
                    
                # x should be in format [B, H*W, C] (without class token) or [B, H*W+1, C] (with class token)
                outs.append(x)
        elif self.encoder_type == 'dinov2':
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
                    return x
                if feature_key in ['q', 'k', 'v']:
                    # (B, Patches+1, num_heads, feat_dim // num_heads)
                    x = x.permute((0, 2, 1, 3)).contiguous()
                    x = x.reshape((x.shape[0], -1, self.feat_dim))  # (B, Patches+1, feat_dim)
                outs.append(x)
        elif self.encoder_type == 'eva02':
            if return_attention_features:
                raise NotImplementedError("EVA02 doesn't support attention feature extraction")
            
            # Get features from EVA02
            block_outputs = self.encoder.forward_features(img)
            if self.blocks is None:
                block_outputs = [block_outputs]
            else:
                # For EVA02, we only get one feature output, so replicate for blocks
                block_outputs = [block_outputs] * len(self.blocks)
            
            outs = []
            for x in block_outputs:
                x = x[feature_key]  # Should be token features [B, H*W+1, C]
                # EVA02 doesn't have attention weights, so skip attention-related features
                if feature_key == 'attn':
                    raise NotImplementedError("EVA02 doesn't support attention weight extraction")
                if feature_key in ['q', 'k', 'v']:
                    raise NotImplementedError("EVA02 doesn't support q/k/v feature extraction")
                outs.append(x)
        
        # Handle the final processing differently for SAM vs other encoders
        if self.encoder_type == 'sam':
            x = torch.cat(outs, dim=2)  # (B, Patches, feat_dim * self.num_blocks)
            # SAM doesn't have class token, so don't remove anything
            # x = x[:, 1:, :]  # Skip this line for SAM
            x = x.permute((0, 2, 1)).contiguous()  # (B, feat_dim, H*W)
            x = x.reshape((x.shape[0], self.feat_dim * self.num_blocks, patches_h,
                            patches_w))  # (B, feat_dim, H, W)
            if self.upsample_factor is not None:
                x = nn.functional.interpolate(x, scale_factor=self.upsample_factor, mode='bilinear',
                                                align_corners=False)  # (B, feat_dim, H, W)
        else:
            # For DINOv2, EVA02, and other ViTs with class tokens
            x = torch.cat(outs, dim=2)  # (B, Patches+1, feat_dim * self.num_blocks)
            x = x[:, 1:, :]  # (B, Patches, feat_dim) - Remove class token
            x = x.permute((0, 2, 1)).contiguous()  # (B, feat_dim, H*W)
            x = x.reshape((x.shape[0], self.feat_dim * self.num_blocks, patches_h,
                                patches_w))  # (B, feat_dim, H, W)
            if self.upsample_factor is not None:
                x = nn.functional.interpolate(x, scale_factor=self.upsample_factor, mode='bilinear',
                                                align_corners=False)  # (B, feat_dim, H, W)

        return x

def apply_lora(model, lora_store: nn.ModuleDict, rank=4, alpha=16, skip_blocks=[0, 1, 2]):
    for name, module in model.named_modules():
        if any(f'blocks.{i}.' in name for i in skip_blocks):
            continue  # skip LoRA for early blocks
        if name.endswith('attn.qkv') and isinstance(module, nn.Linear):
            # name = "blocks.3.attn.qkv"
            parent_name = '.'.join(name.split('.')[:-1]) # parent_name = "blocks.3.attn"
            parent = dict(model.named_modules())[parent_name]

            lora_module = lora.Linear(
                in_features=module.in_features,
                out_features=module.out_features,
                r=rank,
                lora_alpha=alpha,
                fan_in_fan_out=False,
                bias=module.bias is not None
            )

            lora_module.weight.data = module.weight.data.clone()
            if module.bias is not None:
                lora_module.bias.data = module.bias.data.clone()

            unique_name = name.replace('.', '_')
            lora_store[unique_name] = lora_module
            setattr(parent, name.split('.')[-1], lora_store[unique_name])