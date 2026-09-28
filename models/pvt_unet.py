"""
pvt_unet.py

A transformer-based alternative to the plain U-Net:
- Encoder: PVTv2 (Pyramid Vision Transformer v2), pretrained on ImageNet.
  Unlike a CNN, self-attention lets every patch see the whole image from the
  first stage, so global context is available immediately rather than being
  built up layer by layer.
- Decoder: U-Net style with attention gates on the skip connections, so the
  decoder learns which regions of the encoder features matter and suppresses
  the rest (Oktay et al., 2018).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class AttentionGate(nn.Module):
    """
    Filters a skip connection using the decoder's current features as context.

    The decoder signal (g) knows roughly where the structure is; the skip
    connection (x) has the fine spatial detail. This module uses g to produce
    a per-pixel weight between 0 and 1, and scales x by it.
    """

    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.gate_conv = nn.Conv2d(gate_ch, inter_ch, kernel_size=1)
        self.skip_conv = nn.Conv2d(skip_ch, inter_ch, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1),
            nn.Sigmoid(),   # squashes the weights into [0, 1]
        )

    def forward(self, gate, skip):
        g = self.gate_conv(gate)
        x = self.skip_conv(skip)

        # The two feature maps can differ in size; bring the gate to the skip's size.
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)

        weights = self.attention(F.relu(g + x))
        return skip * weights


class DecoderBlock(nn.Module):
    """Upsample, concatenate the gated skip connection, then two convolutions."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.gate = AttentionGate(in_ch, skip_ch, skip_ch // 2)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x, skip):
        skip = self.gate(x, skip)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class PVTUNet(nn.Module):
    """
    PVTv2 encoder + attention-gated U-Net decoder.

    The pretrained encoder expects 3-channel input, so the single grayscale
    channel is repeated three times rather than retraining the first layer.
    """

    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()

        # features_only=True returns the feature map from each stage,
        # which is exactly what a U-Net decoder needs.
        self.encoder = timm.create_model(
            #"pvt_v2_b2",
            "pvt_v2_b0",
            pretrained=pretrained,
            features_only=True,
        )

        # Channel counts of the four encoder stages, read from the model itself
        # rather than hard-coded, so switching to another PVTv2 size still works.
        ch = self.encoder.feature_info.channels()   # e.g. [64, 128, 320, 512]

        self.dec3 = DecoderBlock(ch[3], ch[2], ch[2])
        self.dec2 = DecoderBlock(ch[2], ch[1], ch[1])
        self.dec1 = DecoderBlock(ch[1], ch[0], ch[0])

        # The encoder's first stage is already at 1/4 resolution, so the output
        # needs upsampling back to the input size.
        self.final = nn.Sequential(
            nn.Conv2d(ch[0], ch[0] // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch[0] // 2),
            nn.ReLU(),
            nn.Conv2d(ch[0] // 2, num_classes, kernel_size=1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        # Grayscale -> 3 channels, to match what the pretrained encoder expects.
        x = x.repeat(1, 3, 1, 1)

        f1, f2, f3, f4 = self.encoder(x)

        d3 = self.dec3(f4, f3)
        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2, f1)

        out = self.final(d1)
        return F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)