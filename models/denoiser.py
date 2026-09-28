"""
denoiser.py

A simple convolutional autoencoder for ultrasound speckle denoising.

Note: the decoder uses Upsample + Conv2d instead of ConvTranspose2d.
An earlier version used ConvTranspose2d, which produced visible
checkerboard artifacts in the output (a known issue when kernel_size
is not evenly divisible by stride). Upsample + Conv2d avoids this.
"""

import torch
import torch.nn as nn


class SimpleDenoiser(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: unchanged
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Decoder: Upsample (simple resizing) + Conv2d (learns features),
        # instead of ConvTranspose2d, to avoid checkerboard artifacts.
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),   # 64 -> 128
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),   # 128 -> 256
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded