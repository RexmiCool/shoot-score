from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class AxisMarkerCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),

            nn.MaxPool2d(2),

            ResidualBlock(32),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),

            nn.MaxPool2d(2),

            ResidualBlock(64),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),

            nn.MaxPool2d(2),

            ResidualBlock(128),

            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )

        self.head = nn.Sequential(
            nn.Flatten(),

            nn.Linear(256, 128),
            nn.SiLU(inplace=True),

            nn.Dropout(0.2),

            nn.Linear(128, 3),
        )

    def forward(self, x):
        output = self.head(self.features(x))

        return torch.cat(
            [
                output[:, :1],
                torch.tanh(output[:, 1:])
            ],
            dim=1,
        )