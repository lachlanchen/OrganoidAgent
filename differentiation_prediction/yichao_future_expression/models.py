from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class B2FMultiTaskUNet(nn.Module):
    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(1, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c * 8, c * 12)
        self.up4 = nn.ConvTranspose2d(c * 12, c * 8, 2, stride=2)
        self.dec4 = ConvBlock(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c)
        self.out_image = nn.Sequential(nn.Conv2d(c, 1, 1), nn.Sigmoid())
        self.scalar_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c * 12, c * 4),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(c * 4, 3),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_image(d1), self.scalar_head(b)


class SmallImageEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128, base_channels: int = 24) -> None:
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(1, c, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.SiLU(inplace=True),
            nn.Conv2d(c * 4, c * 6, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 6),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c * 6, embedding_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FutureExpressionModel(nn.Module):
    def __init__(self, feature_dim: int, image_embedding_dim: int = 128, hidden_dim: int = 160) -> None:
        super().__init__()
        self.image_encoder = SmallImageEncoder(embedding_dim=image_embedding_dim)
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 64),
            nn.SiLU(inplace=True),
        )
        self.gru = nn.GRU(image_embedding_dim + 64, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, frames: torch.Tensor, features: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, time_steps, channels, height, width = frames.shape
        flat_frames = frames.reshape(batch * time_steps, channels, height, width)
        image_embeddings = self.image_encoder(flat_frames).reshape(batch, time_steps, -1)
        feature_embeddings = self.feature_encoder(features)
        sequence = torch.cat([image_embeddings, feature_embeddings], dim=-1)
        sequence = sequence * valid.unsqueeze(-1)
        lengths = valid.sum(dim=1).clamp_min(1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed)
        return self.head(hidden[-1])

