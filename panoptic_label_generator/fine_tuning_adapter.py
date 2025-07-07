
from models.dino_vit_adapter import ViTAdapter

import torch
import torch.nn as nn
import pytorch_lightning as pl


class FineTunerAdapter(pl.LightningModule):
    def __init__(self, dinov2_vit_model: str, blocks=None, upsample_factor=None):
        super().__init__()
        self.encoder = ViTAdapter(
            vit_arch_name='vit_small',  # You can parametrize this too
            vit_kwargs=dict(patch_size=14),
            vit_pretrained=True
        )

        # Optional freezeho
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.feat_dim = self.encoder.embed_dim
        self.num_blocks = 1  # Adapter returns list of 4 features, we’ll select one
        self.upsample_factor = upsample_factor

    def forward_encoder(self, img: torch.Tensor):
        # ViTAdapter returns [f1, f2, f3, f4]; f1 has highest resolution
        features = self.encoder(img)
        x = features[0]  # Use highest resolution output
        if self.upsample_factor is not None:
            x = nn.functional.interpolate(x, scale_factor=self.upsample_factor, mode='bilinear',
                                          align_corners=False)
        return x
