from typing import List, Optional

import pytorch_lightning as pl
import torch
from torch import nn

from models.dino_v2 import (
    dinov2_vitb14,
    dinov2_vitg14,
    dinov2_vitl14,
    dinov2_vits14,
)
from models.vit_comer.vit_comer import ViTCoMer  # assuming full model is here


class FineTuner(pl.LightningModule):
    def __init__(self, dinov2_vit_model: str, blocks: Optional[List[int]] = None,
                 upsample_factor: Optional[float] = None, train_output_size: Optional[List[int]] = None):
        super().__init__()
        self.dinov2_vit_model = dinov2_vit_model
        self.blocks = blocks
        self.upsample_factor = upsample_factor
        self.train_output_size = train_output_size

        if dinov2_vit_model == 'vits14':
            self.encoder = dinov2_vits14(pretrained=True)
        elif dinov2_vit_model == 'vitb14':
            self.encoder = dinov2_vitb14(pretrained=True)
        elif dinov2_vit_model == 'vitl14':
            self.encoder = dinov2_vitl14(pretrained=True)
        elif dinov2_vit_model == 'vitg14':
            self.encoder = dinov2_vitg14(pretrained=True)
        elif dinov2_vit_model == 'comer':
            img_h, img_w = self.train_output_size or (504, 1008)
            img_size = (img_h, img_w)  # e.g., (504, 1008)
            pretrain_size = min(img_h, img_w)  # assuming pretrained on square input
            self.encoder = ViTCoMer(
                img_size=img_size,
                patch_size=16,
                embed_dim=384,
                depth=12,
                num_heads=6,
                pretrained=True,
                interaction_indexes=[[3, 5], [6, 8], [9, 11], [9, 11]],
                pretrain_size=pretrain_size
            )
        else:
            raise ValueError(f'Unknown model {dinov2_vit_model}')
 
        self.feat_dim = getattr(self.encoder, "num_features", 384)
        self.patch_size = getattr(self.encoder, "patch_size", 16)
        if hasattr(self.encoder, 'mask_token'):
            self.encoder.mask_token = None  # optional for DDP issues

        # Freeze non-LoRA params if using LoRA
        for name, param in self.encoder.named_parameters():
            if 'lora_' not in name:
                param.requires_grad = False

        trainable_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        print(f"[LoRA/CoMer] Trainable encoder parameters: {trainable_params}")

        if blocks is None:
            self.num_blocks = 1
        else:
            self.num_blocks = len(blocks)

    def forward_encoder(self, img: torch.Tensor, feature_key: str = 'x'):
        img_h, img_w = img.shape[2:]
        patches_h, patches_w = img_h // self.patch_size, img_w // self.patch_size

        if self.dinov2_vit_model == "comer":
            with torch.no_grad():
                feats = self.encoder(img)  # returns list of 4 levels
                x = feats[0]  # pick top feature (or fuse if needed)
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
                        return x
                    if feature_key in ['q', 'k', 'v']:
                        x = x.permute((0, 2, 1, 3)).contiguous()
                        x = x.reshape((x.shape[0], -1, self.feat_dim))
                    outs.append(x)
                x = torch.cat(outs, dim=2)
                x = x[:, 1:, :].permute((0, 2, 1)).contiguous()
                x = x.reshape((x.shape[0], self.feat_dim * self.num_blocks, patches_h, patches_w))

        if self.upsample_factor is not None:
            x = nn.functional.interpolate(x, scale_factor=self.upsample_factor, mode='bilinear',
                                          align_corners=False)
        return x
