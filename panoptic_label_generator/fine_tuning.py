from typing import List, Optional

import pytorch_lightning as pl
import torch
from models.dino_v2 import (
    dinov2_vitb14,
    dinov2_vitg14,
    dinov2_vitl14,
    dinov2_vits14,
)
from torch import nn
import torch.nn.functional as F
from models.vit_adapter.vit_adapter import ViTAdapter
from models.vit_comer.vit_comer import ViTCoMer

class FineTuner(pl.LightningModule):
    def __init__(self, dinov2_vit_model: str, blocks: Optional[List[int]] = None,
                 upsample_factor: Optional[float] = None, use_vitadapter: bool = False, use_vitcomer: bool = False):
        super().__init__()
        self.dinov2_vit_model = dinov2_vit_model
        self.blocks = blocks
        self.upsample_factor = upsample_factor
        self.use_vitadapter = use_vitadapter
        self.use_vitcomer = use_vitcomer

        if use_vitadapter and use_vitcomer:
            raise ValueError("Cannot use both ViTAdapter and ViTCoMer at the same time. Choose one.")

        if self.use_vitadapter: 
            self.encoder = ViTAdapter()
            print(f'[ENCODER] Using encoder: ViTAdapter')
        elif self.use_vitcomer:
            self.encoder = ViTCoMer()
            print(f'[ENCODER] Using encoder: ViTCoMeR')
        elif dinov2_vit_model == 'vits14':
            self.encoder = dinov2_vits14(pretrained=True)
            print(f'[ENCODER] Using encoder: ViT-S14')
        elif dinov2_vit_model == 'vitb14':
            self.encoder = dinov2_vitb14(pretrained=True)
            print(f'[ENCODER] Using encoder: ViT-B14')
        elif dinov2_vit_model == 'vitl14':
            self.encoder = dinov2_vitl14(pretrained=True)
            print(f'[ENCODER] Using encoder: ViT-L14')
        elif dinov2_vit_model == 'vitg14':
            self.encoder = dinov2_vitg14(pretrained=True)
            print(f'[ENCODER] Using encoder: ViT-G14')
        else:
            raise ValueError(f'Unknown model {dinov2_vit_model}')


        # Freeze the encoder if not using adapter or comer
        if self.use_vitadapter == False and self.use_vitcomer == False:
            for param in self.encoder.parameters():
                param.requires_grad = False

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
        if self.use_vitadapter or self.use_vitcomer:
            f1, f2, f3, f4 = self.encoder.forward(img)
            _, _, h_f1, w_f1 = f1.shape

            # Upsample f2, f3, f4 to the same resolution as f1
            # Assuming f1, f2, f3, f4 have the same 'dim' (channels)
            f2_upsampled = F.interpolate(f2, size=(h_f1, w_f1), mode='bilinear', align_corners=False)
            f3_upsampled = F.interpolate(f3, size=(h_f1, w_f1), mode='bilinear', align_corners=False)
            f4_upsampled = F.interpolate(f4, size=(h_f1, w_f1), mode='bilinear', align_corners=False)

            x = torch.cat([f1, f2_upsampled, f3_upsampled, f4_upsampled], dim=1)
            return x
    
        # Default behavior for other models
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
