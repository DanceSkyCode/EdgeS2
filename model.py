import os
import random
import argparse
import numpy as np
import pandas as pd
import nibabel as nib

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix
)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from mamba_ssm import Mamba
from einops import rearrange
import math
class AutoDecomposition(torch.nn.Module):
    def __init__(self,in_ch=1,inter_ch=None,out_ch=None,kernel=4,scale=4):
        super(AutoDecomposition,self).__init__()
        if out_ch is None:
            if inter_ch is None:
                out_ch=in_ch*scale**3
            else:
                out_ch=inter_ch*scale**3

        self.inter_ch=inter_ch
        if self.inter_ch is None:
            self.down=torch.nn.Conv3d(in_ch,out_ch,kernel,scale,0)
        else:
            assert inter_ch>in_ch, "if enabled, inter_ch must be larger than in_ch"
            self.alter=torch.nn.Conv3d(in_ch,inter_ch-in_ch,3,1,1)
            self.down=torch.nn.Conv3d(inter_ch,out_ch,kernel,scale,0)

    def forward(self,x):
        if self.inter_ch is None:
            return self.down(x),x
        else:
            x=torch.cat((x,self.alter(x)),dim=1)
            return self.down(x),x


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
    def shell_scan_indices_3d(self, D, H, W, device="cpu"):
        coords = []
        for d in range(D):
            for h in range(H):
                for w in range(W):
                    k = min(d, h, w, D-1-d, H-1-h, W-1-w)
                    coords.append((k, d, h, w))
        coords.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

        indices = [d*H*W + h*W + w for (_, d, h, w) in coords]
        return torch.tensor(indices, dtype=torch.long, device=device)
    def mamba_forward(self, x):
        B, C, D, H, W = x.shape
        n_tokens = D * H * W
        if not hasattr(self, "shell_idx"):
            self.shell_idx = self.shell_scan_indices_3d(D, H, W, x.device)
        x = x.reshape(B, C, n_tokens)
        x = x[:, :, self.shell_idx]
        x = x.transpose(-1, -2)

        x = self.norm(x)
        x = self.mamba(x)
        inv_idx = torch.argsort(self.shell_idx)
        x = x.transpose(-1, -2)
        x = x[:, :, inv_idx]
        x = x.reshape(B, C, D, H, W)

        return x
    
    def forward(self, x):
        
        x_skip = x

        out_x_1 = self.mamba_forward(x)

        x_2 = rearrange(x, "b c d w h -> b c w d h")

        out_x_2 = self.mamba_forward(x_2)
        
        out_x_2 = rearrange(out_x_2, "b c w d h -> b c d w h")

        x_3 = rearrange(x, "b c d w h -> b c h w d")

        out_x_3 = self.mamba_forward(x_3)
        out_x_3 = rearrange(out_x_3, "b c h w d -> b c d w h")

        out = out_x_1 + out_x_2 + out_x_3

        out = out + x_skip

        return out
def gaussian_kernel_3d(size=3, sigma=1):
    coords = torch.arange(size) - size // 2
    zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2 + zz**2)/(2*sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
def scharr_3d_kernels():
    d = torch.tensor([-1., 0., 1.])
    s = torch.tensor([3., 10., 3.])
    # X direction
    kx = d[:, None, None] * s[None, :, None] * s[None, None, :]
    # Y direction
    ky = s[:, None, None] * d[None, :, None] * s[None, None, :]
    # Z direction
    kz = s[:, None, None] * s[None, :, None] * d[None, None, :]

    return kx, ky, kz
class MultiScaleResBlock3D(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.3, use_mamba=False, use_scharr=True):
        super().__init__()
        self.use_mamba = use_mamba
        self.use_scharr = use_scharr
        self.conv1 = nn.Conv3d(in_c, out_c, 3, padding=1, dilation=1, bias=False)
        self.conv2 = nn.Conv3d(in_c, out_c, 3, padding=2, dilation=2, bias=False)
        self.conv3 = nn.Conv3d(in_c, out_c, 3, padding=3, dilation=3, bias=False)
        self.bn = nn.BatchNorm3d(out_c*3)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout)
        self.res_conv = nn.Conv3d(in_c, out_c, 1) if in_c != out_c else nn.Identity()
        self.pool = nn.MaxPool3d(2)
        self.conv_merge = nn.Conv3d(out_c*3, out_c, 1, bias=False)
        self.bn_merge = nn.BatchNorm3d(out_c)
        if use_mamba:
            self.mamba = MambaLayer(dim=out_c)
        else:
            self.mamba = None
        if use_scharr:
            self.scharr_x = nn.Conv3d(out_c, out_c, 3, padding=1, groups=out_c, bias=False)
            self.scharr_y = nn.Conv3d(out_c, out_c, 3, padding=1, groups=out_c, bias=False)
            self.scharr_z = nn.Conv3d(out_c, out_c, 3, padding=1, groups=out_c, bias=False)
            kx, ky, kz = scharr_3d_kernels()
            kx = kx.unsqueeze(0).unsqueeze(0)  # (1,1,3,3,3)
            ky = ky.unsqueeze(0).unsqueeze(0)
            kz = kz.unsqueeze(0).unsqueeze(0)
            self.scharr_x.weight.data = kx.repeat(out_c, 1, 1, 1, 1)
            self.scharr_y.weight.data = ky.repeat(out_c, 1, 1, 1, 1)
            self.scharr_z.weight.data = kz.repeat(out_c, 1, 1, 1, 1)
            for m in [self.scharr_x, self.scharr_y, self.scharr_z]:
                m.weight.requires_grad = False
            self.gaussian = nn.Conv3d(out_c, out_c, 3, padding=1, groups=out_c, bias=False)
            nn.init.constant_(self.gaussian.weight, 1/27)
            self.norm_scharr = nn.BatchNorm3d(out_c)
        else:
            self.gaussian = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1,
                                    groups=out_c, bias=False)
            kernel_3d = gaussian_kernel_3d()  # (1,1,3,3,3)
            self.gaussian.weight.data = kernel_3d.repeat(out_c,1,1,1,1)

            self.norm = nn.BatchNorm3d(out_c)
            self.conv_extra = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1,
                                        groups=out_c, bias=False)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv_merge(out)
        out = self.bn_merge(out)
        out = out + self.res_conv(x)
        out = self.relu(out)
        if self.mamba is not None:
            out = self.mamba(out) + out
        if self.use_scharr:
            edges_x = self.scharr_x(out)
            edges_y = self.scharr_y(out)
            edges_z = self.scharr_z(out)

            edge = torch.sqrt(
                edges_x ** 2 +
                edges_y ** 2 +
                edges_z ** 2 + 1e-6
            )
            edge = self.gaussian(edge)
            edge = self.norm_scharr(edge)
            edge = torch.clamp(edge, -10.0, 10.0)
            edge = F.relu(edge)
            out = out + edge
        else:
            edge = self.gaussian(out) # out torch.Size([1, 32, 15, 15, 15]) edge torch.Size([1, 32, 17, 15, 15])
            edge = F.relu(self.norm(edge))
            edge = self.conv_extra(edge)
            out = out + edge
        out = self.pool(out)
        return out
# class CrossAttention3D(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.q_conv = nn.Conv3d(dim, dim, 1)
#         self.k_conv = nn.Conv3d(dim, dim, 1)
#         self.v_conv = nn.Conv3d(dim, dim, 1)
#         self.proj = nn.Conv3d(dim, dim, 1)

#     def forward(self, x_q, x_kv):
#         # x_q, x_kv: (B, C, D, H, W)
#         B, C, D, H, W = x_q.shape
#         N = D * H * W

#         q = self.q_conv(x_q).reshape(B, C, N).transpose(1, 2)   # (B, N, C)
#         k = self.k_conv(x_kv).reshape(B, C, N)                  # (B, C, N)
#         v = self.v_conv(x_kv).reshape(B, C, N).transpose(1, 2)  # (B, N, C)

#         attn = torch.bmm(q, k) / (C ** 0.5)                     # (B, N, N)
#         attn = F.softmax(attn, dim=-1)
#         out = torch.bmm(attn, v)                                # (B, N, C)
#         out = out.transpose(1, 2).reshape(B, C, D, H, W)
#         out = self.proj(out)
#         return out
class EnhancedCrossAttention3D(nn.Module):
    def __init__(self, dim, dilation_rates=(1,2), use_graph=True):
        super().__init__()
        self.use_graph = use_graph
        self.q_conv1 = nn.Conv3d(dim, dim, 1)
        self.k_conv1 = nn.Conv3d(dim, dim, 1)
        self.v_conv1 = nn.Conv3d(dim, dim, 1)
        self.dil_conv1 = nn.Conv3d(dim, dim, 3, padding=dilation_rates[1], dilation=dilation_rates[1], groups=dim, bias=False)
        self.dil_conv2 = nn.Conv3d(dim, dim, 5, padding=dilation_rates[1]*2, dilation=dilation_rates[1], groups=dim, bias=False)
        self.proj1 = nn.Conv3d(dim, dim, 1)
    def cross_attention(self, x_q, x_kv, q_conv, k_conv, v_conv, proj, dil_conv1=None, dil_conv2=None):
        B, C, D, H, W = x_q.shape
        N = D * H * W
        if dil_conv1 is not None:
            out = dil_conv1(x_q) + dil_conv2(x_q)
        q = q_conv(x_q).reshape(B, C, N).transpose(1, 2)  # (B, N, C)
        k = k_conv(x_kv).reshape(B, C, N)                # (B, C, N)
        v = v_conv(x_kv).reshape(B, C, N).transpose(1, 2)  # (B, N, C)
        attn = torch.bmm(q, k) / (C ** 0.5)  # (B, N, N)
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)             # (B, N, C)
        out = out.transpose(1, 2).reshape(B, C, D, H, W)
        out = proj(out)
        return out
    def forward(self, branch1, branch2):
        out1 = self.cross_attention(branch1, branch2,
                                    self.q_conv1, self.k_conv1, self.v_conv1,
                                    self.proj1, self.dil_conv1,self.dil_conv2)
        return out1
class LightResConv3D_MS_Mamba_Attention(nn.Module):
    def __init__(self, num_classes=4, dropout=0.3, decomp_scale=2):
        super().__init__()
        self.decomp = AutoDecomposition(in_ch=1, scale=decomp_scale)
        dummy = torch.zeros(1, 1, 124, 124, 124)
        with torch.no_grad():
            out, _ = self.decomp(dummy)
        in_ch_block1 = out.shape[1]
        self.block1 = MultiScaleResBlock3D(in_ch_block1, 16, dropout, use_mamba=False, use_scharr=True)
        self.block2 = MultiScaleResBlock3D(16, 24, dropout, use_mamba=False, use_scharr=True)
        self.block3 = MultiScaleResBlock3D(24, 32, dropout, use_mamba=True, use_scharr=False)
        self.block4 = MultiScaleResBlock3D(32, 64, dropout, use_mamba=True, use_scharr=False)
        self.edge2weight = nn.Sequential(
            nn.Conv3d(24, 64, 3, padding=1),
            nn.Sigmoid()
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, 1, 1, 1))
        self.cross_attn1 = EnhancedCrossAttention3D(64)  # branch1 ↔ branch2
        self.branch_fusion = nn.Conv3d(64*3, 64, kernel_size=1, bias=False)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x, _ = self.decomp(x)
        
        x = self.block1(x)
        edge_feat = self.block2(x) 

        x = self.block3(edge_feat)
        block4_out = self.block4(x)

        weight = self.edge2weight(edge_feat)  # (B, 128, d2, h2, w2)
        weight = F.interpolate(weight, size=block4_out.shape[2:], mode='trilinear', align_corners=False)
        inv_weight = 1 - weight

        branch1 = block4_out + self.pos_embed
        branch2 = block4_out * weight
        branch3 = block4_out * inv_weight
        fused = torch.cat([branch1, branch2, branch3], dim=1)
        fused = self.branch_fusion(fused)

        out1 = self.cross_attn1(branch1, branch2)
        out2 = self.cross_attn1(branch1, branch3)
        out3 = self.cross_attn1(branch1, fused)
        fused = out1 + out2 + out3 + fused
        return self.classifier(fused)