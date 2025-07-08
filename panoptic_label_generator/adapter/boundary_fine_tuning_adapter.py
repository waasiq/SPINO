import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from sklearn.neighbors import KNeighborsClassifier, radius_neighbors_graph
from pytorch_lightning.cli import LightningCLI
from fine_tuning_adapter import FineTunerWithDeformableAdapter
from typing import Any, Dict

class BoundaryFineTunerWithDeformable(FineTunerWithDeformableAdapter):
    def __init__(self, mode='direct', head='mlp', neighbor_radius=1.5,
                 threshold_boundary=0.95, num_boundary_neighbors=1,
                 test_output_size=None, test_multi_scales=None, test_plot=False,
                 **kwargs):
        super().__init__(**kwargs)
        
        assert mode in ['affinity', 'direct']
        self.mode = mode
        self.neighbor_radius = neighbor_radius
        self.threshold_boundary = threshold_boundary
        self.num_boundary_neighbors = num_boundary_neighbors
        self.test_output_size = test_output_size
        self.test_multi_scales = test_multi_scales
        self.test_plot = test_plot
        
        self.connected_indices_cache = None
        
        # Initialize head
        if mode == 'affinity':
            self.head = nn.Sequential(
                nn.Linear(2*self.feat_dim, 600), nn.ReLU(),
                nn.Linear(600, 600), nn.ReLU(),
                nn.Linear(600, 400), nn.ReLU(),
                nn.Linear(400, 1)
            )
        elif head == 'linear':
            self.head = nn.Conv2d(self.feat_dim, 1, 1)
        elif head == 'knn':
            self.head = KNeighborsClassifier(n_neighbors=5)
            self.knn_X, self.knn_y = [], []
        elif head == 'cnn':
            self.head = nn.Sequential(
                nn.Conv2d(self.feat_dim, 600, 3, padding=1), nn.ReLU(),
                nn.Conv2d(600, 600, 3, padding=1), nn.ReLU(),
                nn.Conv2d(600, 400, 3, padding=1), nn.ReLU(),
                nn.Conv2d(400, 1, 3, padding=1)
            )
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Conv2d(self.feat_dim, 600, 1), nn.ReLU(),
                nn.Conv2d(600, 600, 1), nn.ReLU(),
                nn.Conv2d(600, 400, 1), nn.ReLU(),
                nn.Conv2d(400, 1, 1)
            )

    def connected_indices(self, h, w, batch_size=1):
        if (self.connected_indices_cache is not None and
                self.connected_indices_cache[0] == h and
                self.connected_indices_cache[1] == w and
                self.connected_indices_cache[2] == batch_size):
            return self.connected_indices_cache[3]

        y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        y_grid = y_grid.reshape(-1)
        x_grid = x_grid.reshape(-1)
        points = np.stack((y_grid, x_grid), axis=1).astype(np.float64)

        connectivity = radius_neighbors_graph(points, radius=self.neighbor_radius,
                                            mode='connectivity', include_self=False)
        connectivity = connectivity.tocoo()
        connected_indices = np.stack((connectivity.row, connectivity.col), axis=1)

        if batch_size > 1:
            connected_indices = np.tile(connected_indices, (batch_size, 1, 1))
            for b in range(1, batch_size):
                connected_indices[b] += b * h * w

        self.connected_indices_cache = (h, w, batch_size, connected_indices)
        return connected_indices

    def forward(self, img, segment_mask=None):
        x = self.forward_encoder(img)
        
        if self.mode == 'affinity':
            batch_size = x.shape[0]
            h, w = x.shape[2], x.shape[3]
            x = x.view((batch_size, self.feat_dim, h * w)).permute((0, 2, 1)).contiguous()
            
            if segment_mask is not None:
                segment_mask = segment_mask.view(-1)
                x = x.view(-1, self.feat_dim)
                x = x[segment_mask == 1, :]
                x = x.view(batch_size, -1, self.feat_dim)
                
            connected_indices = self.connected_indices(h, w, batch_size)
            x = x.view(-1, self.feat_dim)
            connected_indices = connected_indices.reshape(-1, 2)
            x1 = x[connected_indices[:, 0], :]
            x2 = x[connected_indices[:, 1], :]
            x = torch.cat((x1, x2), dim=1)
            x = x.view(batch_size, -1, 2 * self.feat_dim)
            x = self.head(x)
            x = torch.sigmoid(x)
        else:
            if isinstance(self.head, KNeighborsClassifier):
                if self.training:
                    return x
                feat_shape = x.shape
                x = x.permute(0, 2, 3, 1).reshape(-1, feat_shape[1])
                x = x.detach().cpu().numpy()
                x = self.head.predict_proba(x)
                x = np.expand_dims(x[:, 1], axis=1)
                x = torch.from_numpy(x).to(self.device)
                x = x.reshape(feat_shape[0], feat_shape[2], feat_shape[3], -1).permute(0, 3, 1, 2)
            else:
                x = torch.sigmoid(self.head(x))
        return x

    def training_step(self, train_batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        rgb = train_batch['rgb']
        ins = train_batch['instance'].long()

        device = rgb.device
        batch_size = rgb.shape[0]
        rgb_h, rgb_w = rgb.shape[2:]
        patches_h, patches_w = rgb_h // self.patch_size, rgb_w // self.patch_size
        upsample_factor = 1.0 if self.hparams.upsample_factor is None else self.hparams.upsample_factor
        network_output_size = (int(patches_h * upsample_factor), int(patches_w * upsample_factor))

        ins = TF.resize(ins, size=[network_output_size[0], network_output_size[1]],
                        interpolation=TF.InterpolationMode.NEAREST)
        ins_flattened = ins.view(ins.shape[0], -1)
        connected_indices = self.connected_indices(network_output_size[0], network_output_size[1])

        if self.mode == 'affinity':
            ins_boundary = (ins_flattened[:, connected_indices[:, 0]] ==
                            ins_flattened[:, connected_indices[:, 1]]).to(torch.float)
            pred = self(rgb)
            pred = pred.squeeze(2)
        elif self.mode == 'direct':
            ins_boundary = (ins_flattened[:, connected_indices[:, 0]] !=
                            ins_flattened[:, connected_indices[:, 1]]).cpu().numpy().astype(int)
            connected_indices = np.tile(connected_indices, (batch_size, 1, 1))
            indices = connected_indices[:, :, 0]
            ins_boundary = np.add.reduceat(ins_boundary,
                                           np.unique(indices, return_index=True, axis=1)[1],
                                           axis=1)
            ins_boundary = np.logical_not(ins_boundary >= self.num_boundary_neighbors)
            ins_boundary = torch.Tensor(ins_boundary.reshape(batch_size, network_output_size[0],
                                                             network_output_size[1])).to(device)

            if isinstance(self.head, KNeighborsClassifier):
                x = self(rgb)
                x = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
                x = x.detach().cpu().numpy()
                self.knn_X.append(x)
                ins_boundary = ins_boundary.reshape(-1)
                ins_boundary = ins_boundary.detach().cpu().numpy()
                self.knn_y.append(ins_boundary)
            else:
                pred = self(rgb)
                pred = pred.squeeze(1)

        if isinstance(self.head, KNeighborsClassifier):
            loss = torch.tensor([0.0], requires_grad=True).to(self.device)
        else:
            loss = F.binary_cross_entropy(pred, ins_boundary)

        self.log('train_loss', loss)
        return loss

    def predict(self, rgb: torch.Tensor) -> torch.Tensor:
        batch_size = rgb.shape[0]
        rgb_h, rgb_w = rgb.shape[2:]
        patches_h, patches_w = rgb_h // self.patch_size, rgb_w // self.patch_size
        upsample_factor = 1.0 if self.hparams.upsample_factor is None else self.hparams.upsample_factor
        network_output_size = (int(patches_h * upsample_factor), int(patches_w * upsample_factor))

        if self.mode == 'direct':
            if self.test_multi_scales is None:
                pred = self(rgb)
                pred = pred.squeeze(1)
            else:
                raise NotImplementedError("Multi-scale testing not implemented")
            
            pred = (pred > self.threshold_boundary).to(torch.float)

        pred = nn.functional.interpolate(pred.unsqueeze(1), size=self.test_output_size,
                                         mode='nearest').squeeze(1)
        return pred

class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument('--data_params', type=dict)

if __name__ == '__main__':
    CLI(model_class=BoundaryFineTunerWithDeformable, save_config_callback=None)