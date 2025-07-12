import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from fine_tuning_adapter import FineTunerWithDeformableAdapter
from PIL import Image
from pytorch_lightning.cli import LightningCLI
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

# Ignore some torch warnings
warnings.filterwarnings('ignore', '.*The default behavior for interpolate/upsample with float*')
warnings.filterwarnings(
    'ignore', '.*Default upsampling behavior when mode=bicubic is changed to align_corners=False*')
warnings.filterwarnings('ignore', '.*Only one label was provided to `remove_small_objects`*')


class SemanticFineTunerWithDeformable(FineTunerWithDeformableAdapter):
    """Fine-tunes a small head on top of the DINOv2 model with deformable adapter for semantic segmentation.

    Parameters
    ----------
    dinov2_vit_model : str
        ViT model name of DINOv2. One of ['vits14', 'vitl14', 'vitg14', 'vitb14'].
    num_classes : int
        Number of classes for semantic segmentation.
    train_output_size : Tuple[int, int]
        Output size [H, W] after head.
    head : str
        Head to use for semantic segmentation. One of ['linear', 'knn', 'cnn', 'mlp'].
    ignore_index : int
        Index to ignore in the loss.
    top_k_percent_pixels : float
        Percentage of hardest pixels to keep for the loss.
    test_output_size : Tuple[int, int]
        Final output size [H, W] of the model during prediction/testing.
    test_plot : bool
        Whether to plot the predictions during testing.
    use_checkpoint : bool
        Whether to use gradient checkpointing to save memory.
    """

    def __init__(self, num_classes: int, train_output_size: Tuple[int, int],
                 head='mlp', ignore_index=255, top_k_percent_pixels=1.0,
                 test_output_size=None, test_plot=False,
                 use_checkpoint=False, **kwargs):
        super().__init__(**kwargs)
        
        self.num_classes = num_classes
        self.train_output_size = train_output_size
        self.ignore_index = ignore_index
        self.top_k_percent_pixels = top_k_percent_pixels
        self.test_output_size = test_output_size
        self.test_plot = test_plot
        self.use_checkpoint = use_checkpoint

        # Initialize head based on adapter feature dimension
        adapter_feat_dim = self.feat_dim * (len(self.hparams.blocks) if self.hparams.blocks else 1)
        
        if head == 'linear':
            self.head = nn.Conv2d(adapter_feat_dim, num_classes, kernel_size=1, stride=1, padding=0)
        elif head == 'knn':
            self.head = KNeighborsClassifier(n_neighbors=5)
            self.knn_X, self.knn_y = [], []
        elif head == 'cnn':
            self.head = nn.Sequential(
                nn.Conv2d(adapter_feat_dim, 600, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(600, 600, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(600, 400, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(400, 200, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(200, num_classes, kernel_size=1, stride=1, padding=0),
            )
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Conv2d(adapter_feat_dim, 600, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(600, 600, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(600, 400, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(400, 200, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(200, num_classes, kernel_size=1, stride=1, padding=0),
            )
        else:
            raise ValueError(f'Unknown head {head}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_encoder(x)  # (B, feat_dim, H, W)
        if isinstance(self.head, KNeighborsClassifier):
            if self.training:
                return x  # return only features during training
            feat_shape = x.shape
            x = x.permute(0, 2, 3, 1).reshape(-1, feat_shape[1])
            x = x.detach().cpu().numpy()
            x = self.head.predict_proba(x)  # (B * H * W, num_classes)
            x = torch.from_numpy(x).to(self.device)
            x = x.reshape(feat_shape[0], feat_shape[2], feat_shape[3], -1) \
                .permute(0, 3, 1, 2)  # (B, num_classes, H, W)
        else:
            x = self.head(x)  # (B, num_classes, H, W)
        x = nn.functional.interpolate(x, size=self.train_output_size, mode='bilinear',
                                      align_corners=False)
        return x

    def training_step(self, train_batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        rgb = train_batch['rgb']
        sem = train_batch['semantic'].long()

        if isinstance(self.head, KNeighborsClassifier):
            x = self(rgb)  # (B, feat_dim, H, W)
            feat_h, feat_w = x.shape[2:]
            x = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])  # (B * H * W, feat_dim)
            x = x.detach().cpu().numpy()
            self.knn_X.append(x)

            sem = TF.resize(sem, [feat_h, feat_w], interpolation=InterpolationMode.NEAREST)
            sem = sem.reshape(-1)
            sem = sem.detach().cpu().numpy()
            self.knn_y.append(sem)

            loss = torch.tensor([0.0], requires_grad=True).to(self.device)  # dummy loss
        else:
            sem = TF.resize(sem, self.train_output_size, interpolation=InterpolationMode.NEAREST)
            pred = self(rgb)
            loss = F.cross_entropy(pred, sem, ignore_index=self.ignore_index, reduction='none')

            if self.top_k_percent_pixels < 1.0:
                loss = loss.contiguous().view(-1)
                # Hard pixel mining
                top_k_pixels = int(self.top_k_percent_pixels * loss.numel())
                loss, _ = torch.topk(loss, top_k_pixels)
            loss = loss.mean()

        self.log('train_loss', loss)
        return loss

    def predict(self, rgb: torch.Tensor) -> torch.Tensor:
        pred = self(rgb)  # (B, num_classes, H, W)
        if self.test_output_size is not None:
            pred = nn.functional.interpolate(pred, size=self.test_output_size, mode='bilinear',
                                             align_corners=False)
        pred = torch.argmax(pred, dim=1)  # (B, H, W)
        return pred

    def on_train_epoch_end(self):
        if isinstance(self.head, KNeighborsClassifier):
            if self.knn_X:
                X = np.concatenate(self.knn_X, axis=0)
                y = np.concatenate(self.knn_y, axis=0)
                self.head.fit(X, y)
                self.knn_X, self.knn_y = [], []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return [optimizer], [scheduler]


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument('--data_params', type=dict)


if __name__ == '__main__':
    CLI(model_class=SemanticFineTunerWithDeformable, save_config_callback=None)
