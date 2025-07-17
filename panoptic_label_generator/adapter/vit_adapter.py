import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_, trunc_normal_
from typing import List, Optional 
import pytorch_lightning as pl
import math 

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from external.ms_deformable_attention.modules.deform_attn import MSDeformAttn
from typing import List, Optional, Callable, Sequence, Tuple, Union

import logging
from functools import partial

from adapter.adapter_modules import SpatialPriorModule, InteractionBlockWithCls, deform_inputs
from adapter.dino_transformer import DinoVisionTransformer


#! Params for the vit-small: embed_dim = 384
#! Params for the vit-base: embed_dim = 768

#! Disabled with_cffn for now, in original its True
class ViTAdapter(DinoVisionTransformer):
    def __init__(self, pretrain_size=518, conv_inplane=64, n_points=4, blocks=0, embed_dim=768,
                 deform_num_heads=6, init_values=0., interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]], with_cffn=False,
                 cffn_ratio=0.25, deform_ratio=1.0, add_vit_feature=False,
                 use_extra_extractor=True, with_cp=False, num_tokens=1):

        super().__init__()
        url = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        self.load_state_dict(state_dict, strict=True)

        # Freeze vit 
        for param in self.parameters():
            param.requires_grad = False

        self.mask_token = None
        self.num_blocks = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        self.interactions = nn.Sequential(*[
            InteractionBlockWithCls(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate, 
                             with_cffn=with_cffn, norm_layer=self.norm_layer, 
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1
                                               else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])

        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.norm3 = nn.BatchNorm2d(embed_dim)
        self.norm4 = nn.BatchNorm2d(embed_dim)

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

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 14, self.pretrain_size[1] // 14, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        #! AI thinks this is correct - Debugged with print outputs. 
        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Input image shape: torch.Size([1, 3, 448, 896])
        _, _, h, w = x.shape  

        # Patch embedding - this is correct I (Waasiq) verified it.
        #x = self.patch_embed.forward(x).to(x.device) # Output: torch.Size([1, 2592, 384])
        x = self.patch_embed(x)

        W_vit = w // self.patch_size # Width of the ViT patches: 896 // 14 = 64
        H_vit = h // self.patch_size # Height of the ViT patches: 448 // 14 = 32
        W_adapter = w // 16 # Width of the Adapter patches: 896 // 16 = 56
        H_adapter = h // 16 # Height of the Adapter patches: 448 // 16 = 28
        bs, n, dim = x.shape

        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H_vit, W_vit)
        x = x + pos_embed

        cls = self.cls_token.expand(x.shape[0], -1, -1) + self.pos_embed[:, 0]

        #* This is where the magic happens
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
            x1, x2, x3, x4 = outs # All x_i are (bs, dim, 36, 72) from H_vit, W_vit
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)

        # Return the features
        return [f1, f2, f3, f4]
