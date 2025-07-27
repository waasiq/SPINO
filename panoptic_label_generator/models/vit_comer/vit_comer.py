import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from external.ms_deformable_attention.modules.deform_attn import MSDeformAttn
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_

from models.dino_transformer import DinoVisionTransformer

from .comer_modules import CNN, CTIBlock, deform_inputs

_logger = logging.getLogger(__name__)

#! Params for the vit-small: embed_dim = 384
#! Params for the vit-base: embed_dim = 768

class ViTCoMer(DinoVisionTransformer):
    def __init__(self, pretrain_size=518, embed_dim=768, num_heads=6, conv_inplane=64, n_points=4, deform_num_heads=6,
                 init_values=0., interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]], with_cffn=False, cffn_ratio=0.25,
                 deform_ratio=1.0, add_vit_feature=False, use_extra_CTI=False, pretrained=None, with_cp=False,
                 use_CTI_toV=True, use_CTI_toC=True, cnn_feature_interaction=False, dim_ratio=6.0,
                 *args, **kwargs):
        
        super().__init__( *args, **kwargs)

        #! Change this to dynamic loading later on
        url = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        self.load_state_dict(state_dict, strict=True)
        
        # Freezing the parameters
        for param in self.parameters():
            param.requires_grad = False

        # self.num_classes = 80
        self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.use_CTI_toC = use_CTI_toC
        self.use_CTI_toV = use_CTI_toV
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim
        self.patch_size = kwargs.get("patch_size", 14)
        self.num_features = embed_dim
        
        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = CNN(inplanes=conv_inplane, embed_dim=embed_dim)
        self.interactions = nn.Sequential(*[
            CTIBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                            init_values=init_values, drop_path=self.drop_path_rate,
                            norm_layer=self.norm_layer, with_cffn=with_cffn,
                            cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                            use_CTI_toV=use_CTI_toV if isinstance(use_CTI_toV, bool) else use_CTI_toV[i],
                            use_CTI_toC=use_CTI_toC if isinstance(use_CTI_toC, bool) else use_CTI_toC[i],
                            dim_ratio=dim_ratio,
                            cnn_feature_interaction=cnn_feature_interaction if isinstance(cnn_feature_interaction, bool) else cnn_feature_interaction[i],
                            extra_CTI=((True if i == len(interaction_indexes) - 1 else False) and use_extra_CTI))
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

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

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 14, self.pretrain_size[1] // 14, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
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
        # deform_inputs are not used consistently; they can be generated inside the blocks
        # if needed, but the logic should be self-contained in the CTI blocks.

        # 1. === SPM forward and Shape Calculation ===
        _, _, h_img, w_img = x.shape  # Original image size: 336, 784
        c1, c2, c3, c4 = self.spm(x)
        c1_orig = c1

        c_shapes = [c2.shape[2:], c3.shape[2:], c4.shape[2:]]
        c_lens = [c2.shape[1], c3.shape[1], c4.shape[1]]


        # Store the exact spatial shapes and sequence lengths of the original CNN features
        bs, _, c2_h, c2_w = c2.shape
        _, _, c3_h, c3_w = c3.shape
        _, _, c4_h, c4_w = c4.shape
        
        # These are now the ground truth for reshaping later
        # c2_h=42, c2_w=98
        # c3_h=21, c3_w=49
        # c4_h=10, c4_w=24
        
        # Flatten features and store their original lengths for slicing
        c1 = c1.view(bs, self.embed_dim, -1).transpose(1, 2)
        c2 = c2.view(bs, self.embed_dim, -1).transpose(1, 2)
        c3 = c3.view(bs, self.embed_dim, -1).transpose(1, 2)
        c4 = c4.view(bs, self.embed_dim, -1).transpose(1, 2)

        c2_len, c3_len, c4_len = c2.shape[1], c3.shape[1], c4.shape[1]

        # Add level embeddings
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # 2. === Patch Embedding forward ===
        x = self.patch_embed(x)
        
        W_vit = w_img // self.patch_size   # 56
        H_vit = h_img // self.patch_size   # 24
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H_vit, W_vit)
        x = x + pos_embed

        # 3. === Interaction ===
        # The CTI blocks should ideally handle their own inputs without needing
        # pre-calculated deform_inputs passed from the top level.
        # We pass the ground-truth shapes of the middle CNN layer (c3).
        H_adapter, W_adapter = c3_h, c3_w
        
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]   
            x, c = layer(
                    x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                    H=H_adapter,          # Pass H
                    W=W_adapter,          # Pass W
                    c_shapes=c_shapes,
                    c_lens=c_lens,
                    patch_size=self.patch_size
                )

        
        # The logic for creating 'outs' seems to be for a different architecture (e.g., SegFormer)
        # Ensure this is what you intend.
        x_spatial = x.transpose(1, 2).view(bs, self.embed_dim, H_vit, W_vit)
        x_resized_to_adapter = F.interpolate(x_spatial, size=(H_adapter, W_adapter), mode='bilinear', align_corners=False)
        outs.append(x_resized_to_adapter.contiguous())

        # 4. === Split & Reshape (ROBUST METHOD) ===
        # Use the stored original lengths for precise slicing.
        c2_out = c[:, :c2_len, :]
        c3_out = c[:, c2_len : c2_len + c3_len, :]
        c4_out = c[:, c2_len + c3_len : c2_len + c3_len + c4_len, :]

        # Use the stored original spatial dimensions for safe reshaping.
        # This completely removes the need for the fragile factorization logic.
        c2_reshaped = c2_out.transpose(1, 2).view(bs, self.embed_dim, c2_h, c2_w)
        c3_reshaped = c3_out.transpose(1, 2).view(bs, self.embed_dim, c3_h, c3_w)
        c4_reshaped = c4_out.transpose(1, 2).view(bs, self.embed_dim, c4_h, c4_w)

        c1_out = self.up(c2_reshaped) + c1_orig

        if self.add_vit_feature:
            # Assuming 4 interaction blocks, one for each feature scale
            x1, x2, x3, x4 = outs
            # This part requires careful thought on which ViT output corresponds to which CNN scale
            # Example resizing:
            x1 = F.interpolate(x1, size=(c1_out.shape[2], c1_out.shape[3]), mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, size=(c2_reshaped.shape[2], c2_reshaped.shape[3]), mode='bilinear', align_corners=False)
            x3 = F.interpolate(x3, size=(c3_reshaped.shape[2], c3_reshaped.shape[3]), mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, size=(c4_reshaped.shape[2], c4_reshaped.shape[3]), mode='bilinear', align_corners=False)
            c1_out, c2_reshaped, c3_reshaped, c4_reshaped = c1_out + x1, c2_reshaped + x2, c3_reshaped + x3, c4_reshaped + x4

        # Final Norm
        f1 = self.norm1(c1_out)
        f2 = self.norm2(c2_reshaped)
        f3 = self.norm3(c3_reshaped)
        f4 = self.norm4(c4_reshaped)
        return [f1, f2, f3, f4]