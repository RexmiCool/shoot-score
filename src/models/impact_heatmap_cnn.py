"""Lightweight U-Net for dense bullet-impact heatmap prediction.

Architecture: 3-level encoder/decoder with skip connections, BatchNorm,
and ConvTranspose2d upsampling.  Input: (B, 1, 512, 512) grayscale.
Output: (B, 1, 512, 512) sigmoid heatmap in [0, 1].

The model is intentionally small (≈ 350 k params, base_ch=16) so it can be
exported to TFLite and run in real-time on mid-range Android devices.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ImpactHeatmapCNN(nn.Module):
    """U-Net with skip connections for precise impact localisation.

    Args:
        base_ch: Number of channels in the first encoder block.  All
            subsequent levels double this value.  Increased to 28 for better
            detection accuracy while staying within 3s TFLite inference budget.
    """

    def __init__(self, base_ch: int = 28) -> None:
        """Initialise all encoder, bottleneck and decoder layers."""
        super().__init__()
        ch = base_ch

        # ── Encoder ───────────────────────────────────────────────────────
        self.enc1 = self._block(1, ch)  # 512 → 512  (skip s1)
        self.enc2 = self._block(ch, ch * 2)  # 256 → 256  (skip s2)
        self.enc3 = self._block(ch * 2, ch * 4)  # 128 → 128  (skip s3)
        self.pool = nn.MaxPool2d(2)

        # ── Bottleneck ────────────────────────────────────────────────────
        self.bottleneck = self._block(ch * 4, ch * 8)  # 64 → 64

        # ── Decoder ───────────────────────────────────────────────────────
        # ConvTranspose2d gives learnable upsampling (better for localisation
        # than bilinear Upsample which blurs peak positions).
        self.up3 = nn.ConvTranspose2d(ch * 8, ch * 4, kernel_size=2, stride=2)
        self.dec3 = self._block(ch * 8, ch * 4)  # cat(up3, s3)

        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=2, stride=2)
        self.dec2 = self._block(ch * 4, ch * 2)  # cat(up2, s2)

        self.up1 = nn.ConvTranspose2d(ch * 2, ch, kernel_size=2, stride=2)
        self.dec1 = self._block(ch * 2, ch)  # cat(up1, s1)

        self.head = nn.Conv2d(ch, 1, kernel_size=1)

    # ------------------------------------------------------------------
    @staticmethod
    def _block(in_ch: int, out_ch: int) -> nn.Sequential:
        """Return a Conv-BN-ReLU x2 block.

        Args:
            in_ch: Number of input channels.
            out_ch: Number of output channels.

        Returns:
            A sequential module with two Conv-BN-ReLU sub-layers.
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Batch of grayscale images with shape ``(B, 1, H, W)``.

        Returns:
            Sigmoid heatmap with shape ``(B, 1, H, W)`` in ``[0, 1]``.
        """
        # Encoder — keep skip tensors for lateral connections
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))

        # Bottleneck
        b = self.bottleneck(self.pool(s3))

        # Decoder — concatenate upsampled feature map with skip tensor
        d3 = self.dec3(torch.cat([self.up3(b), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))

        return torch.sigmoid(self.head(d1))
