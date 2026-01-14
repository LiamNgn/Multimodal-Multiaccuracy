"""Model definitions for DeepMultimodalKernel.

Provides a simple 3D CNN for imaging features and an MLP for clinical
covariates, then fuses them into a joint representation suitable for
kernel-based alignment.
"""

from typing import Tuple

import torch
import torch.nn as nn


class Conv3DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ImagingEncoder(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            Conv3DBlock(in_channels, 16),
            Conv3DBlock(16, 32),
            Conv3DBlock(32, 64),
            nn.AdaptiveAvgPool3d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        return h.view(h.size(0), -1)


class ClinicalMLP(nn.Module):
    def __init__(self, in_features: int, hidden: int = 64, out_features: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepMultimodalKernel(nn.Module):
    def __init__(self, clinical_dim: int, embedding_dim: int = 64):
        super().__init__()
        self.img_encoder = ImagingEncoder()
        self.clinical_encoder = ClinicalMLP(clinical_dim, out_features=embedding_dim)
        self.proj = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(self, img: torch.Tensor, clinical: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_z = self.img_encoder(img)
        clin_z = self.clinical_encoder(clinical)
        fused = self.proj(torch.cat([img_z, clin_z], dim=1))
        return img_z, clin_z, fused
