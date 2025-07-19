from typing import List, Optional

import pytorch_lightning as pl
import torch
from models.eva02 import (
    eva02_tiny_patch14_196,
    eva02_small_patch14_196,
    eva02_base_patch14_224,
    eva02_large_patch14_224,
    eva02_large_patch14_336,
    eva02_large_patch14_448,
    EVA02Wrapper
)
from torch import nn
import torch.nn.functional as F
from models.vit_adapter.vit_adapter import ViTAdapter
from models.vit_comer.vit_comer import ViTCoMer

class EVA02FineTuner(pl.LightningModule):
    def __init__(self, eva02_model: str, blocks: Optional[List[int]] = None,
                 upsample_factor: Optional[float] = None, use_vitadapter: bool = False, 
                 use_vitcomer: bool = False):
        super().__init__()
        self.eva02_model = eva02_model
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
        elif eva02_model == 'tiny_patch14_196':
            base_model = eva02_tiny_patch14_196(pretrained=True)
            self.encoder = EVA02Wrapper(base_model)
            print(f'[ENCODER] Using encoder: EVA02-Tiny-14')
        elif eva02_model == 'small_patch14_196':
            base_model = eva02_small_patch14_196(pretrained=True)
            self.encoder = EVA02Wrapper(base_model)
            print(f'[ENCODER] Using encoder: EVA02-Small-14')
        elif eva02_model == 'base_patch14_224':
            base_model = eva02_base_patch14_224(pretrained=True)
            self.encoder = EVA02Wrapper(base_model)
            print(f'[ENCODER] Using encoder: EVA02-Base-14')
        elif eva02_model == 'large_patch14_224':
            base_model = eva02_large_patch14_224(pretrained=True)
            self.encoder = EVA02Wrapper(base_model)
            print(f'[ENCODER] Using encoder: EVA02-Large-14')
        elif eva02_model == 'large_patch14_336':
            base_model = eva02_large_patch14_336(pretrained=True)
            self.encoder = EVA02Wrapper(base_model)
            print(f'[ENCODER] Using encoder: EVA02-Large-14-336')
        elif eva02_model == 'large_patch14_448':
            base_model = eva02_large_patch14_448(pretrained=True)
            self.encoder = EVA02Wrapper(base_model)
            print(f'[ENCODER] Using encoder: EVA02-Large-14-448')
        else:
            raise ValueError(f'Unknown EVA02 model {eva02_model}')

        # Freeze the encoder if not using adapter or comer
        if self.use_vitadapter == False and self.use_vitcomer == False:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.feat_dim = self.encoder.num_features
        self.patch_size = self.encoder.patch_size

        if blocks is None:
            self.num_blocks = 1
        else:
            self.num_blocks = len(blocks)

    def forward_encoder(self, img: torch.Tensor, feature_key: str = 'x'):
        img_h, img_w = img.shape[2:]
        patches_h, patches_w = img_h // self.patch_size, img_w // self.patch_size

        return_attention_features = any([(feature_key in x) for x in ['q', 'k', 'v', 'attn']])

        # Logic for ViTAdapter and ViTCoMer (unchanged)
        if self.use_vitadapter or self.use_vitcomer:
            f1, f2, f3, f4 = self.encoder.forward(img)
            _, _, h_f1, w_f1 = f1.shape

            # Upsample f2, f3, f4 to the same resolution as f1
            f2_upsampled = F.interpolate(f2, size=(h_f1, w_f1), mode='bilinear', align_corners=False)
            f3_upsampled = F.interpolate(f3, size=(h_f1, w_f1), mode='bilinear', align_corners=False)
            f4_upsampled = F.interpolate(f4, size=(h_f1, w_f1), mode='bilinear', align_corners=False)

            x = torch.cat([f1, f2_upsampled, f3_upsampled, f4_upsampled], dim=1)
            return x
    
        # EVA02 specific behavior
        with torch.no_grad():
            # EVA02 doesn't support attention features extraction like DINOv2
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
            
            x = torch.cat(outs, dim=2)  # (B, Patches+1, feat_dim * self.num_blocks)

            x = x[:, 1:, :]  # Remove cls token: (B, Patches, feat_dim)
            x = x.permute((0, 2, 1)).contiguous()  # (B, feat_dim, H*W)
            x = x.reshape((x.shape[0], self.feat_dim * self.num_blocks, patches_h,
                           patches_w))  # (B, feat_dim, H, W)
            
            if self.upsample_factor is not None:
                x = nn.functional.interpolate(x, scale_factor=self.upsample_factor, mode='bilinear',
                                              align_corners=False)  # (B, feat_dim, H, W)
        return x

    def configure_optimizers(self):
        """Configure optimizers - only train the head, freeze EVA02 backbone"""
        # Get only non-frozen parameters (head parameters)
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=1e-3,  # Higher learning rate for head
            weight_decay=1e-4
        )
        
        return optimizer