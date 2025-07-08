import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import pytorch_lightning as pl

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.dino_v2 import dinov2_vitb14, dinov2_vitg14, dinov2_vitl14, dinov2_vits14

class DeformableAttention(nn.Module):
    """Deformable Attention v2 with offset modulation"""
    def __init__(self, feat_dim: int, num_heads=8, offset_groups=4):
        super().__init__()
        self.num_heads = num_heads
        self.offset_groups = offset_groups
        self.head_dim = feat_dim // num_heads
        
        # Offset prediction
        self.offset_conv = nn.Conv2d(feat_dim, 2 * offset_groups, 3, padding=1)
        self.modulator_conv = nn.Conv2d(feat_dim, offset_groups, 3, padding=1)
        
        # Attention projections
        self.q_proj = nn.Conv2d(feat_dim, feat_dim, 1)
        self.kv_proj = nn.Conv2d(feat_dim, 2 * feat_dim, 1)
        self.out_proj = nn.Conv2d(feat_dim, feat_dim, 1)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 1.)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Generate offsets and modulators
        offsets = self.offset_conv(x)  # [B, 2*G, H, W]
        modulators = 1. + torch.tanh(self.modulator_conv(x))  # [B, G, H, W]
        
        # Project queries, keys, values
        q = self.q_proj(x).view(B, self.num_heads, self.head_dim, H, W)
        kv = self.kv_proj(x).view(B, 2 * self.num_heads, self.head_dim, H, W)
        k, v = kv.chunk(2, dim=1)  # [B, nh, hd, H, W] each
        
        # Deformable sampling
        sampled_feats = []
        for h in range(self.num_heads):
            # Get offsets for this head
            g = h % self.offset_groups
            offset = offsets[:, 2*g:2*(g+1)]  # [B, 2, H, W]
            modulator = modulators[:, g]  # [B, H, W]
            
            # Sample features
            grid = self._get_sampling_grid(x, offset)
            sampled_v = F.grid_sample(
                v[:, h], grid, mode='bilinear', padding_mode='zeros', align_corners=False
            ) * modulator.unsqueeze(1)
            sampled_feats.append(sampled_v)
        
        # Combine and project
        out = torch.stack(sampled_feats, dim=1)  # [B, nh, hd, H, W]
        out = out.reshape(B, C, H, W)
        return self.out_proj(out)

    def _get_sampling_grid(self, x: torch.Tensor, offset: torch.Tensor):
        B, _, H, W = x.shape
        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W))
        grid = torch.stack((xx, yy), 0).float().to(x.device)  # [2, H, W]
        grid = grid.unsqueeze(0) + offset  # [B, 2, H, W]
        
        # Normalize to [-1, 1]
        grid[:, 0] = 2.0 * grid[:, 0] / (W - 1) - 1.0  # x coord
        grid[:, 1] = 2.0 * grid[:, 1] / (H - 1) - 1.0  # y coord
        return grid.permute(0, 2, 3, 1)  # [B, H, W, 2]

class ViTAdapter(nn.Module):
    def __init__(self, feat_dim: int, use_deformable=True, reduction_ratio=0.25):
        super().__init__()
        reduced_dim = int(feat_dim * reduction_ratio)
        
        # Bottleneck layers
        self.down = nn.Conv2d(feat_dim, reduced_dim, 1)
        self.act = nn.GELU()
        self.up = nn.Conv2d(reduced_dim, feat_dim, 1)
        
        # Deformable branch
        self.use_deformable = use_deformable
        if use_deformable:
            self.deform_attn = DeformableAttention(feat_dim)
        
        # Init
        nn.init.normal_(self.down.weight, std=1e-3)
        nn.init.normal_(self.up.weight, std=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.down(x)
        x = self.act(x)
        x = self.up(x)
        
        if self.use_deformable:
            x = x + self.deform_attn(identity)
        return identity + x

class FineTunerWithDeformableAdapter(pl.LightningModule):
    def __init__(self, dinov2_vit_model: str, blocks: Optional[List[int]] = None,
                 upsample_factor: Optional[float] = None, use_adapter=True,
                 adapter_reduction=0.25, use_deformable=True):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize DINOv2 backbone
        if dinov2_vit_model == 'vits14':
            self.encoder = dinov2_vits14(pretrained=True)
        elif dinov2_vit_model == 'vitb14':
            self.encoder = dinov2_vitb14(pretrained=True)
        elif dinov2_vit_model == 'vitl14':
            self.encoder = dinov2_vitl14(pretrained=True)
        elif dinov2_vit_model == 'vitg14':
            self.encoder = dinov2_vitg14(pretrained=True)
        
        self.feat_dim = self.encoder.num_features
        self.patch_size = self.encoder.patch_size
        for p in self.encoder.parameters():
            p.requires_grad = False
            
        # Adapter
        if use_adapter:
            adapter_feat_dim = self.feat_dim * (len(blocks) if blocks else 1)
            self.adapter = ViTAdapter(
                adapter_feat_dim,
                use_deformable=use_deformable,
                reduction_ratio=adapter_reduction
            )

    def forward_encoder(self, img: torch.Tensor, feature_key='x'):
        img_h, img_w = img.shape[2:]
        patches_h, patches_w = img_h // self.patch_size, img_w // self.patch_size

        return_attention_features = any([(feature_key in x) for x in ['q', 'k', 'v', 'attn']])
        with torch.no_grad():
            block_outputs = self.encoder.forward_features(
                img,
                return_attention_features=return_attention_features,
                return_blocks=self.hparams.blocks)
            if self.hparams.blocks is None:
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
            x = torch.cat(outs, dim=2)  # (B, Patches+1, feat_dim * num_blocks)

            x = x[:, 1:, :]  # (B, Patches, feat_dim * num_blocks)
            x = x.permute((0, 2, 1)).contiguous()  # (B, feat_dim * num_blocks, H*W)
            
            # Calculate the total feature dimension
            total_feat_dim = self.feat_dim * len(block_outputs)
            x = x.reshape((x.shape[0], total_feat_dim, patches_h, patches_w))  # (B, total_feat_dim, H, W)
            
            if self.hparams.upsample_factor is not None:
                x = F.interpolate(x, scale_factor=self.hparams.upsample_factor, mode='bilinear',
                                align_corners=False)  # (B, total_feat_dim, H, W)
        
        # Apply adapter
        if self.hparams.use_adapter:
            x = self.adapter(x)
        return x