import logging
from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn
from external.ms_deformable_attention.modules.deform_attn import MSDeformAttn
from timm.models.layers import DropPath
import torch.utils.checkpoint as cp

import math

_logger = logging.getLogger(__name__)


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 16, w // 16)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]
    
    spatial_shapes = torch.as_tensor([(h // 16, w // 16)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8),
                                                   (h // 16, w // 16),
                                                   (h // 32, w // 32)], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]
    
    return deform_inputs1, deform_inputs2


def deform_inputs_only_one(x, h, w):
    # bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)], device=x.device)
    deform_inputs = [reference_points, spatial_shapes, level_start_index]
    
    return deform_inputs


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MRFP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = MultiDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        
        # Calculate actual sequence lengths based on spatial dimensions
        len_x1 = H * 2 * W * 2  # 8x downsampling
        len_x2 = H * W           # 16x downsampling  
        len_x3 = N - len_x1 - len_x2  # 32x downsampling (remainder)
        
        # Calculate actual spatial dimensions for x3
        H3 = int(math.sqrt(len_x3 * H / W)) if len_x3 > 0 else H // 2
        W3 = len_x3 // H3 if H3 > 0 else W // 2
        
        # Adjust H3, W3 to ensure H3 * W3 = len_x3
        if H3 * W3 != len_x3 and len_x3 > 0:
            # Find best factor pair for len_x3
            best_h3 = H // 2
            best_w3 = W // 2
            min_diff = float('inf')
            
            for h in range(1, int(math.sqrt(len_x3)) + 1):
                if len_x3 % h == 0:
                    w = len_x3 // h
                    # Prefer dimensions close to expected ratios
                    expected_h = H // 2
                    expected_w = W // 2
                    diff = abs(h - expected_h) + abs(w - expected_w)
                    if diff < min_diff:
                        min_diff = diff
                        best_h3, best_w3 = h, w
            H3, W3 = best_h3, best_w3
        
        x1 = x[:, 0:len_x1, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, len_x1:len_x1+len_x2, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, len_x1+len_x2:len_x1+len_x2+len_x3, :].transpose(1, 2).view(B, C, H3, W3).contiguous()
        
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x
    

class MultiDWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        dim1 = dim
        dim = dim // 2

        self.dwconv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv2 = nn.Conv2d(dim, dim, 5, 1, 2, bias=True, groups=dim)

        self.dwconv3 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv4 = nn.Conv2d(dim, dim, 5, 1, 2, bias=True, groups=dim)

        self.dwconv5 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv6 = nn.Conv2d(dim, dim, 5, 1, 2, bias=True, groups=dim)

        self.act1 = nn.GELU()
        self.bn1 = nn.BatchNorm2d(dim1)

        self.act2 = nn.GELU()
        self.bn2 = nn.BatchNorm2d(dim1)

        self.act3 = nn.GELU()
        self.bn3 = nn.BatchNorm2d(dim1)

    def forward(self, x, H, W):
        B, N, C = x.shape
        
        # Calculate actual sequence lengths based on spatial dimensions
        len_x1 = H * 2 * W * 2  # 8x downsampling
        len_x2 = H * W           # 16x downsampling  
        len_x3 = N - len_x1 - len_x2  # 32x downsampling (remainder)
        
        # Calculate actual spatial dimensions for x3
        H3 = int(math.sqrt(len_x3 * H / W)) if len_x3 > 0 else H // 2
        W3 = len_x3 // H3 if H3 > 0 else W // 2
        
        # Adjust H3, W3 to ensure H3 * W3 = len_x3
        if H3 * W3 != len_x3 and len_x3 > 0:
            # Find best factor pair for len_x3
            best_h3 = H // 2
            best_w3 = W // 2
            min_diff = float('inf')
            
            for h in range(1, int(math.sqrt(len_x3)) + 1):
                if len_x3 % h == 0:
                    w = len_x3 // h
                    # Prefer dimensions close to expected ratios
                    expected_h = H // 2
                    expected_w = W // 2
                    diff = abs(h - expected_h) + abs(w - expected_w)
                    if diff < min_diff:
                        min_diff = diff
                        best_h3, best_w3 = h, w
            H3, W3 = best_h3, best_w3
        
        x1 = x[:, 0:len_x1, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, len_x1:len_x1+len_x2, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, len_x1+len_x2:len_x1+len_x2+len_x3, :].transpose(1, 2).view(B, C, H3, W3).contiguous()
        
        x11, x12 = x1[:,:C//2,:,:], x1[:,C//2:,:,:]
        x11 = self.dwconv1(x11)  # BxCxHxW
        x12 = self.dwconv2(x12)
        x1 = torch.cat([x11, x12], dim=1)
        x1 = self.act1(self.bn1(x1)).flatten(2).transpose(1, 2)
        

        x21, x22 = x2[:,:C//2,:,:], x2[:,C//2:,:,:]
        x21 = self.dwconv3(x21)
        x22 = self.dwconv4(x22)
        x2 = torch.cat([x21, x22], dim=1)
        x2 = self.act2(self.bn2(x2)).flatten(2).transpose(1, 2)

        x31, x32 = x3[:,:C//2,:,:], x3[:,C//2:,:,:]
        x31 = self.dwconv5(x31)
        x32 = self.dwconv6(x32)
        x3 = torch.cat([x31, x32], dim=1)
        x3 = self.act3(self.bn3(x3)).flatten(2).transpose(1, 2)

        x = torch.cat([x1, x2, x3], dim=1)
        return x


class MultiscaleExtractor(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    
    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):
        
        def _inner_forward(query, feat):
            
            attn = self.attn(self.query_norm(query), reference_points,
                             self.feat_norm(feat), spatial_shapes,
                             level_start_index, None)
            query = query + attn
            
            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))

            return query
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)
        
        return query

class CTI_toC(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False,
                 cnn_feature_interaction=True):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        
        self.cnn_feature_interaction = cnn_feature_interaction
        if cnn_feature_interaction:
            self.cfinter = MultiscaleExtractor(dim=dim, n_levels=3, num_heads=num_heads,
                                n_points=n_points, norm_layer=norm_layer, 
                                deform_ratio=deform_ratio, with_cffn=with_cffn,
                                cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, 
                                with_cp=with_cp)
    
    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W, patch_size=14):
        
        def _inner_forward(query, feat, H, W, patch_size):
            B, N_query, C = query.shape
            B, N_feat, C = feat.shape
            
            # Calculate ViT spatial dimensions
            H_vit = (H * 16) // patch_size
            W_vit = (W * 16) // patch_size
            
            # Verify ViT dimensions
            if H_vit * W_vit != N_feat:
                sqrt_n = int(math.sqrt(N_feat))
                if sqrt_n * sqrt_n == N_feat:
                    H_vit = W_vit = sqrt_n
                else:
                    for h in range(1, int(math.sqrt(N_feat)) + 1):
                        if N_feat % h == 0:
                            w = N_feat // h
                            if abs(w/h - 2.0) < 0.5:  # Prefer 2:1 aspect ratio
                                H_vit, W_vit = h, w
                                break
            
            # Split CNN features into different scales
            n = N_query // 21
            x1 = query[:, 0:16 * n, :].contiguous()      # 8x features: H*2 * W*2 * 4 = H*W*16
            x2 = query[:, 16 * n:20 * n, :].contiguous()  # 16x features: H * W * 4 = H*W*4 -> but should be H*W
            x3 = query[:, 20 * n:, :].contiguous()        # 32x features: H//2 * W//2 * 4 = H*W//4 -> should be H*W//4
            
            # Actually, let me recalculate based on the spatial dimensions:
            # For H=42, W=84: H*W*4=14112, H*W=3528, H*W//4=882
            # Total: 14112 + 3528 + 882 = 18522, but n=N_query//21 gives different splits
            
            # Let's use the actual expected sizes:
            len_x1 = H * 2 * W * 2  # 8x downsampling: 84*168 = 14112
            len_x2 = H * W           # 16x downsampling: 42*84 = 3528  
            len_x3 = N_query - len_x1 - len_x2  # remainder
            
            x1 = query[:, 0:len_x1, :].contiguous()
            x2 = query[:, len_x1:len_x1+len_x2, :].contiguous()
            x3 = query[:, len_x1+len_x2:, :].contiguous()
            
            # Resize ViT features to match x2 (16x downsampling, H*W elements)
            if x2.shape[1] != feat.shape[1]:
                # Resize ViT features from (H_vit, W_vit) to (H, W)
                feat_spatial = feat.transpose(1, 2).view(B, C, H_vit, W_vit)
                feat_resized = F.interpolate(feat_spatial, size=(H, W), mode='bilinear', align_corners=False)
                feat_matched = feat_resized.flatten(2).transpose(1, 2)
            else:
                feat_matched = feat
                
            x2 = x2 + feat_matched
            query = torch.cat([x1, x2, x3], dim=1)

            if self.cnn_feature_interaction:               
                deform_input = deform_inputs_only_one(query, H*16, W*16)
                query = self.cfinter(query=self.query_norm(query), reference_points=deform_input[0],
                          feat=self.feat_norm(query), spatial_shapes=deform_input[1],
                          level_start_index=deform_input[2],
                          H=H, W=W)               
            
            return query
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat, H, W, patch_size)
        else:
            query = _inner_forward(query, feat, H, W, patch_size)
        
        return query


class Extractor_CTI(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False,
                 cnn_feature_interaction=True):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.cnn_feature_interaction = cnn_feature_interaction
        if cnn_feature_interaction:
            self.cfinter = MultiscaleExtractor(dim=dim, n_levels=3, num_heads=num_heads,
                                n_points=n_points, norm_layer=norm_layer, 
                                deform_ratio=deform_ratio, with_cffn=with_cffn,
                                cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, 
                                with_cp=with_cp)
    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W, patch_size=14):        
        def _inner_forward(query, feat, H, W, patch_size):
            B, N_query, C = query.shape
            B, N_feat, C = feat.shape
            
            # Calculate ViT spatial dimensions
            H_vit = (H * 16) // patch_size
            W_vit = (W * 16) // patch_size
            
            # Verify ViT dimensions
            if H_vit * W_vit != N_feat:
                sqrt_n = int(math.sqrt(N_feat))
                if sqrt_n * sqrt_n == N_feat:
                    H_vit = W_vit = sqrt_n
                else:
                    for h in range(1, int(math.sqrt(N_feat)) + 1):
                        if N_feat % h == 0:
                            w = N_feat // h
                            if abs(w/h - 2.0) < 0.5:
                                H_vit, W_vit = h, w
                                break
            
            # Split CNN features into different scales (same logic as CTI_toC)
            len_x1 = H * 2 * W * 2  # 8x downsampling
            len_x2 = H * W           # 16x downsampling  
            len_x3 = N_query - len_x1 - len_x2  # remainder
            
            x1 = query[:, 0:len_x1, :].contiguous()
            x2 = query[:, len_x1:len_x1+len_x2, :].contiguous()
            x3 = query[:, len_x1+len_x2:, :].contiguous()
            
            # Resize ViT features to match x2
            if x2.shape[1] != feat.shape[1]:
                feat_spatial = feat.transpose(1, 2).view(B, C, H_vit, W_vit)
                feat_resized = F.interpolate(feat_spatial, size=(H, W), mode='bilinear', align_corners=False)
                feat_matched = feat_resized.flatten(2).transpose(1, 2)
            else:
                feat_matched = feat
                
            x2 = x2 + feat_matched
            query = torch.cat([x1, x2, x3], dim=1)

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W)) 

            if self.cnn_feature_interaction:               
                deform_input = deform_inputs_only_one(query, H*16, W*16)
                query = self.cfinter(query=self.query_norm(query), reference_points=deform_input[0],
                        feat=self.feat_norm(query), spatial_shapes=deform_input[1],
                        level_start_index=deform_input[2],
                        H=H, W=W)               
            
            return query
        
        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat, H, W, patch_size)
        else:
            query = _inner_forward(query, feat, H, W, patch_size)
        
        return query

class CTI_toV(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=3, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False, drop=0., drop_path=0., cffn_ratio=0.25):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads,
                                 n_points=n_points, ratio=deform_ratio)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
        self.ffn_norm = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W,
                c_shapes, c_lens, patch_size=14):

        def _inner_forward(query, feat, reference_points, spatial_shapes, level_start_index, H, W,
                        c_shapes, c_lens, patch_size):
            # ... (the inner logic I provided before is correct) ...
            H_vit = (H * 16) // patch_size
            W_vit = (W * 16) // patch_size
            vit_ref_points = get_reference_points([(H_vit, W_vit)], query.device)
            
            cnn_info = self.attn(
                self.query_norm(query),
                vit_ref_points,
                self.feat_norm(feat),
                spatial_shapes,
                level_start_index,
                None
            )
            
            query = query + self.drop_path(self.gamma * cnn_info)
            return query

        # --- THIS IS THE FIX ---
        # Add 'patch_size' to the list of arguments for checkpointing.
        if self.with_cp and query.requires_grad:
            return cp.checkpoint(_inner_forward, query, feat, reference_points, spatial_shapes, level_start_index, H, W, c_shapes, c_lens, patch_size)
        else:
            return _inner_forward(query, feat, reference_points, spatial_shapes, level_start_index, H, W, c_shapes, c_lens, patch_size)


class CTIBlock(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_CTI=False, with_cp=False, 
                 use_CTI_toV=True, 
                 use_CTI_toC=True,
                 dim_ratio=6.0,
                 cnn_feature_interaction=False):
        super().__init__()

        if use_CTI_toV:
            self.cti_tov = CTI_toV(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp, drop=drop, drop_path=drop_path, cffn_ratio=cffn_ratio)
        if use_CTI_toC:
            self.cti_toc = CTI_toC(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp,
                                   cnn_feature_interaction=cnn_feature_interaction)
        
        if extra_CTI:
            self.extra_CTIs = nn.Sequential(*[
                Extractor_CTI(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp,
                                   cnn_feature_interaction=cnn_feature_interaction)
                for _ in range(4)
            ])
        else:
            self.extra_CTIs = None
        
        self.use_CTI_toV = use_CTI_toV
        self.use_CTI_toC = use_CTI_toC
        self.mrfp = MRFP(dim, hidden_features=int(dim * dim_ratio))

    def forward(self, x, c, blocks, H, W,
            c_shapes, c_lens, patch_size=14):
        B, N, C = x.shape
        H_vit = (H * 16) // patch_size
        W_vit = (W * 16) // patch_size
            
        # Verify calculation
        expected_N = H_vit * W_vit
        if expected_N != N:
            print(f"Warning: Expected ViT sequence length {expected_N}, got {N}")
            print(f"H_vit={H_vit}, W_vit={W_vit}, actual N={N}")
            # Fallback to factor-based calculation
            for h in range(1, int(math.sqrt(N)) + 1):
                if N % h == 0:
                    w = N // h
                    # Choose the pair that's closest to the expected aspect ratio
                    expected_ratio = W_vit / H_vit if H_vit > 0 else 2.0
                    actual_ratio = w / h
                    if abs(actual_ratio - expected_ratio) < 0.1:  # Allow some tolerance
                        H_vit, W_vit = h, w
                        break
        
        if self.use_CTI_toV:
            # Pre-process CNN features if needed (e.g., with MRFP)
            c_processed = self.mrfp(c, H, W)
            
            # Directly generate inputs for the attention call
            cnn_spatial_shapes = x.new_tensor(c_shapes, dtype=torch.long)
            cnn_level_start_index = torch.cat((cnn_spatial_shapes.new_zeros((1,)), cnn_spatial_shapes.prod(1).cumsum(0)[:-1]))

            # Call the corrected CTI_toV
            x = self.cti_tov(
                query=x,
                # reference_points are now generated inside CTI_toV
                reference_points=None, 
                feat=c_processed,
                spatial_shapes=cnn_spatial_shapes,
                level_start_index=cnn_level_start_index,
                H=H, W=W,
                c_shapes=c_shapes, c_lens=c_lens, patch_size=patch_size
            )

        for idx, blk in enumerate(blocks):
            x = blk(x)

        if self.use_CTI_toC:
            cnn_ref_points = get_reference_points(c_shapes, c.device)
            vit_spatial_shapes = x.new_tensor([[H_vit, W_vit]], dtype=torch.long)
            vit_level_start_index = x.new_zeros((1,))
            
            c = self.cti_toc(
                query=c,
                reference_points=cnn_ref_points,
                feat=x,
                spatial_shapes=vit_spatial_shapes,
                level_start_index=vit_level_start_index,
                H=H, W=W, patch_size=patch_size
            )
        return x, c

class CNN(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384):
        super().__init__()

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)

        return c1, c2, c3, c4