import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_, trunc_normal_
from typing import List, Optional 
import pytorch_lightning as pl

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dino_v2 import dinov2_vitb14, dinov2_vits14
from external.ms_deformable_attention.modules.deform_attn import MSDeformAttn
from typing import List, Optional

import pytorch_lightning as pl
import torch
import math 

from adapter.adapter_modules import SpatialPriorModule, InteractionBlockWithCls, deform_inputs
from torch import nn

from adapter.patch_embed import PatchEmbed

#! Check the blocks thing
class DinoV2(pl.LightningModule):
    def __init__(self, dinov2_vit_model: str, blocks: Optional[List[int]] = None,
                 upsample_factor: Optional[float] = None):
        super().__init__()
        self.dinov2_vit_model = dinov2_vit_model
        self.blocks = blocks
        self.upsample_factor = upsample_factor

        if dinov2_vit_model == 'vits14':
            self.encoder = dinov2_vits14(pretrained=True)
        elif dinov2_vit_model == 'vitb14':
            self.encoder = dinov2_vitb14(pretrained=True)

        self.feat_dim = self.encoder.num_features
        self.patch_size = self.encoder.patch_size
        self.encoder.mask_token = None  # can't use ddp_find_unused_parameters_false otherwise

        for param in self.encoder.parameters():  # freeze backbone
            param.requires_grad = False

        if blocks is None:
            self.num_blocks = 1
        else:
            self.num_blocks = None

#! Params for the vit-small: embed_dim = 384
#! Params for the vit-base: embed_dim = 768
class ViTAdapter(DinoV2):
    def __init__(self, pretrain_size=224, conv_inplane=64, n_points=4, blocks=1, embed_dim=384, dinov2_vit_model='vits14',
                 deform_num_heads=6, init_values=0., interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]], with_cffn=True,
                 cffn_ratio=0.25, deform_ratio=1.0, add_vit_feature=True,
                 use_extra_extractor=True, with_cp=False, num_tokens=1,
                 vit_model=None, vit_kwargs=dict()):

        super().__init__(dinov2_vit_model, blocks)

        if dinov2_vit_model == 'vits14':
            self.embed_dim = 384
        elif dinov2_vit_model == 'vitb14':
            self.embed_dim = 768

        self.mask_token = None
        self.num_blocks = blocks
        self.blocks = self.encoder.blocks  
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.num_tokens = num_tokens

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        self.level_embed = nn.Parameter(torch.zeros(3, self.embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=self.embed_dim, with_cp=False)
        self.interactions = nn.Sequential(*[
            InteractionBlockWithCls(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, # drop_path=self.drop_path_rate, #! Commented out default is adapter_module but check later on 
                             with_cffn=with_cffn, # norm_layer=self.norm_layer, #! Commented out default is in adapter_module but check later on 
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])

        self.up = nn.ConvTranspose2d(self.embed_dim, self.embed_dim, 2, 2)
        self.norm1 = nn.BatchNorm2d(self.embed_dim)
        self.norm2 = nn.BatchNorm2d(self.embed_dim)
        self.norm3 = nn.BatchNorm2d(self.embed_dim)
        self.norm4 = nn.BatchNorm2d(self.embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

        

    def init_weights(self) -> None:
        pass

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # Uses pre-train size
    def _get_pos_embed(self, pos_embed, H, W):
        # pos_embed shape: [1, num_patches, embed_dim]
        # We need to reshape it to match the current H, W
        B, N, C = pos_embed.shape
        # Calculate the square root to get the original spatial dimensions
        # For a square pretrain size, we have sqrt(N) x sqrt(N) patches
        pretrain_H = int(math.sqrt(N))
        pretrain_W = int(math.sqrt(N))
        
        # If not square, use the pretrain_size
        if pretrain_H * pretrain_W != N:
            pretrain_H = self.pretrain_size[0] // 14
            pretrain_W = self.pretrain_size[1] // 14
            
        pos_embed = pos_embed.reshape(1, pretrain_H, pretrain_W, C).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, C, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        _, _, h, w = x.shape  # Input image shape: torch.Size([1, 3, 504, 1008])
        patchEmbed = PatchEmbed(
            img_size=(h, w),  # Only height and width, not the full shape
            patch_size=self.patch_size, 
            in_chans=3, 
            embed_dim=self.embed_dim
        ).to(x.device)
        x = patchEmbed.forward(x)

        W_vit = w // self.patch_size
        H_vit = h // self.patch_size
        W_adapter = w // 14
        H_adapter = h // 14
        bs, n, dim = x.shape

        num_patches = patchEmbed.num_patches
        # Get position embeddings from the DINOv2 encoder and interpolate them for current input size
        pos_embed_full = self.encoder.interpolate_pos_encoding(x, w, h) #! Added by claude fixes the error on x = x + pos_embed but check
        # Split into class token and patch position embeddings
        cls_pos_embed = pos_embed_full[:, 0:1, :]
        pos_embed = pos_embed_full[:, 1:, :]

        x = x + pos_embed
        cls = self.cls_token.expand(x.shape[0], -1, -1) + cls_pos_embed

        #! Interaction gives error 
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c, cls = layer(x, c, cls, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H_adapter, W_adapter)
            outs.append(x.transpose(1, 2).view(bs, dim, H_vit, W_vit).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H_adapter * 2, W_adapter * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H_adapter, W_adapter).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H_adapter // 2, W_adapter // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)

        
        # Concatinating fusion layers - Waasiq + Sarthak's thought process
        target_size = f2.shape[-2:]
    
        f1_resized = F.interpolate(f1, size=target_size, mode='bilinear', align_corners=False)
        f3_resized = F.interpolate(f3, size=target_size, mode='bilinear', align_corners=False)
        f4_resized = F.interpolate(f4, size=target_size, mode='bilinear', align_corners=False)
        
        # Concatenate or add features
        fused = torch.cat([f1_resized, f2, f3_resized, f4_resized], dim=1)

        return fused 


class Identity(torch.nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

    def forward(self, x):
        return x
