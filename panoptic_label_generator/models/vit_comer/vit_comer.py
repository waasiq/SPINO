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
                 init_values=0., interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]], with_cffn=True, cffn_ratio=0.25,
                 deform_ratio=1.0, add_vit_feature=False, use_extra_CTI=True, pretrained=None,with_cp=False,
                 use_CTI_toV=True, use_CTI_toC=True, cnn_feature_interaction=True, dim_ratio=6.0,
                 *args, **kwargs):
        kwargs['patch_size'] = 14  # DINOv2 uses patch_size=14
        
        super().__init__( *args, **kwargs)

        self.patch_size = 14  # Explicitly set for clarity
        self.grid_size = None  # Will be set dynamically

        #! Change this to dynamic loading later on
        url = 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth'
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        self.load_state_dict(state_dict, strict=True)

        # Calculate actual pretrain size from position embeddings
        num_pos_patches = self.pos_embed.shape[1] - 1  # Subtract 1 for cls token
        grid_size = int(math.sqrt(num_pos_patches))
        actual_pretrain_size = grid_size * 14  # DINOv2 uses patch_size=14
        
        print(f"Detected pretrain size: {actual_pretrain_size}x{actual_pretrain_size}")
        self.pretrain_size = (actual_pretrain_size, actual_pretrain_size)

        # Freeze vit 
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
        self.patch_size = kwargs.get("patch_size", 16)
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
        # Get the actual number of position embeddings from the tensor
        num_pos_embeds = pos_embed.shape[1]  # Should be 1369 for DINOv2
        orig_size = int(math.sqrt(num_pos_embeds))
        
        # Verify it's a perfect square
        if orig_size * orig_size != num_pos_embeds:
            raise ValueError(f"Position embedding has {num_pos_embeds} patches, not a perfect square")
        
        # Reshape from flat to 2D grid
        pos_embed = pos_embed.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
        
        # Interpolate to match the actual H, W dimensions (can be rectangular)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False)
        
        # Flatten back to sequence format
        pos_embed = pos_embed.reshape(1, -1, H * W).permute(0, 2, 1)
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
        # Debug information
        print(f"Input image shape: {x.shape}")  # Should be [B, 3, H, W]
        original_x = x
        
        # Before CNN operations
        print(f"Before deform_inputs: {original_x.shape}")
        deform_inputs1, deform_inputs2 = deform_inputs(original_x)
        
        # SPM forward
        print(f"Before SPM: {original_x.shape}")
        c1, c2, c3, c4 = self.spm(original_x)
        print(f"After SPM - c1: {c1.shape}, c2: {c2.shape}, c3: {c3.shape}, c4: {c4.shape}")
        
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)
        print(f"Concatenated CNN features shape: {c.shape}")

        _, _, h, w = original_x.shape  # Use original_x, not x

        # ViT pathway - patch embed the original image
        print(f"Before patch embedding: {original_x.shape}")
        vit_features = self.patch_embed(original_x)  # Now shape: [batch_size, 2048, 768]
        print(f"After patch embedding: {vit_features.shape}")
        
        # FIX: Use vit_features.shape instead of x.shape
        bs, n, dim = vit_features.shape
        print(f"Extracted dimensions - bs: {bs}, n: {n}, dim: {dim}")

        # Calculate actual grid dimensions from the patch embedding output
        W_vit = w // self.patch_size 
        H_vit = h // self.patch_size 
        print(f"Grid dimensions - H_vit: {H_vit}, W_vit: {W_vit}")
        
        # Store for consistent use across modules
        self.grid_size = (H_vit, W_vit)

        # Debug: verify the calculation
        expected_patches = H_vit * W_vit
        actual_patches = n
        print(f"Patch verification - expected: {expected_patches}, actual: {actual_patches}")
        assert expected_patches == actual_patches, f"Patch count mismatch: expected {expected_patches}, got {actual_patches}"

        W_adapter = w // 16 
        H_adapter = h // 16 
        print(f"Adapter dimensions - H_adapter: {H_adapter}, W_adapter: {W_adapter}")

        # FIX: Calculate pos_embed BEFORE using it
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H_vit, W_vit)
        print(f"Position embed shape: {pos_embed.shape}")

        # FIX: Add pos_embed to vit_features, not undefined variable
        vit_features = vit_features + pos_embed
        print(f"After adding position embedding: {vit_features.shape}")

        # REMOVE: This duplicate SPM call is unnecessary
        # c1, c2, c3, c4 = self.spm(x)

        # Interaction
        outs = list()
        print(f"Starting interaction loops...")
        for i, layer in enumerate(self.interactions):
            print(f"Interaction {i} - Input vit_features: {vit_features.shape}, c: {c.shape}")
            indexes = self.interaction_indexes[i]
            vit_features, c = layer(vit_features, c, self.blocks[indexes[0]:indexes[-1] + 1],
                                deform_inputs1, deform_inputs2, H_adapter, W_adapter)
            print(f"Interaction {i} - Output vit_features: {vit_features.shape}, c: {c.shape}")
            outs.append(vit_features.transpose(1, 2).view(bs, dim, H_adapter, W_adapter).contiguous())
        
        print(f"Completed interactions. Number of outputs: {len(outs)}")

        # Split & Reshape
        print(f"Before splitting - c: {c.shape}")
        c2_new = c[:, 0:c2.size(1), :]
        c3_new = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4_new = c[:, c2.size(1) + c3.size(1):, :]
        print(f"After splitting - c2: {c2_new.shape}, c3: {c3_new.shape}, c4: {c4_new.shape}")

        c2_new = c2_new.transpose(1, 2).view(bs, dim, H_adapter * 2, W_adapter * 2).contiguous()
        c3_new = c3_new.transpose(1, 2).view(bs, dim, H_adapter, W_adapter).contiguous()
        c4_new = c4_new.transpose(1, 2).view(bs, dim, H_adapter // 2, W_adapter // 2).contiguous()
        print(f"After reshaping - c2: {c2_new.shape}, c3: {c3_new.shape}, c4: {c4_new.shape}")
        
        c1 = self.up(c2_new) + c1
        print(f"After upsampling and adding - c1: {c1.shape}")

        # Update variables for consistency
        c2, c3, c4 = c2_new, c3_new, c4_new

        if self.add_vit_feature:
            print("Adding ViT features to CNN features...")
            x1, x2, x3, x4 = outs
            print(f"ViT outputs before interpolation - x1: {x1.shape}, x2: {x2.shape}, x3: {x3.shape}, x4: {x4.shape}")
            
            x1 = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x4, scale_factor=0.5, mode='bilinear', align_corners=False)
            print(f"ViT outputs after interpolation - x1: {x1.shape}, x2: {x2.shape}, x3: {x3.shape}, x4: {x4.shape}")
            
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4
            print(f"Final combined features - c1: {c1.shape}, c2: {c2.shape}, c3: {c3.shape}, c4: {c4.shape}")

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        print(f"After normalization - f1: {f1.shape}, f2: {f2.shape}, f3: {f3.shape}, f4: {f4.shape}")
        
        return [f1, f2, f3, f4]
