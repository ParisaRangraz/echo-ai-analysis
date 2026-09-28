"""
unet.py

Two segmentation architectures:
- SimpleUNet: a minimal 2-level U-Net (kept for reference; its receptive
  field turned out to be too small to understand whole-heart structure).
- UNet: a standard 4-level U-Net with double convolutions and BatchNorm.
"""

import torch
import torch.nn as nn


class SimpleUNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = nn.Sequential(
            nn.Conv2d(32 + 32, 16, kernel_size=3, padding=1), nn.ReLU(),
        )

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = nn.Sequential(
            nn.Conv2d(16 + 16, num_classes, kernel_size=3, padding=1),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        u1 = self.up1(p2)
        u1 = torch.cat([u1, e2], dim=1)
        d1 = self.dec1(u1)

        u2 = self.up2(d1)
        u2 = torch.cat([u2, e1], dim=1)
        d2 = self.dec2(u2)

        return d2


class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by BatchNorm and ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    A standard 4-level U-Net. Downsamples four times (256 -> 16), giving a
    much larger receptive field than SimpleUNet, so each output pixel can
    see enough context to know where it sits within the heart.
    """

    def __init__(self, num_classes=4, base=16):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Encoder
        self.enc1 = DoubleConv(1, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.bottleneck = DoubleConv(base * 8, base * 16)

        # Decoder (input channels = upsampled + skip connection)
        self.dec4 = DoubleConv(base * 16 + base * 8, base * 8)
        self.dec3 = DoubleConv(base * 8 + base * 4, base * 4)
        self.dec2 = DoubleConv(base * 4 + base * 2, base * 2)
        self.dec1 = DoubleConv(base * 2 + base, base)

                # Final 1x1 conv: one output channel per class
        self.out = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))

        return self.out(d1)