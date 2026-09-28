"""
pvt_rta_transformer.py

The transformer variant of reverse attention, as described in RTA-Former
(Li, Yi, Uneri, Niu & Jones, EMBC 2024, arXiv:2401.11671).

pvt_rta.py implements the convolutional variant. In the paper's own ablation
the convolutional version gains little while the transformer version gains
roughly 2%, so testing only the convolutional one would leave an inconclusive
result: a null result could mean either "reverse attention does not help here"
or "we tested the weak variant". Running both separates those.

The difference is inside the reverse branch: instead of a convolutional
bottleneck, the attention map is produced by a self-attention block, so the
branch can relate distant parts of the image when deciding what the edge
region contains.

This model has more parameters than pvt_rta.py, so the two are not a
like-for-like comparison of size -- which is exactly why both are run, with
pvt_unet.py as the shared baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SelfAttentionBlock(nn.Module):
    """
    A single transformer block over the spatial positions of a feature map.

    Each position attends to every other, so the attention map it produces can
    depend on the whole image rather than a local neighbourhood. This is the
    piece the paper's ablation shows to matter.

    Attention is computed at a reduced resolution and upsampled back: at full
    resolution the cost grows with the square of the pixel count, which is not
    affordable on CPU.
    """

    def __init__(self, channels, num_heads=2, reduce_to=16):
        super().__init__()
        self.reduce_to = reduce_to

        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        b, c, h, w = x.shape

        # Work at a fixed small resolution, so cost does not depend on input size.
        size = min(self.reduce_to, h, w)
        reduced = F.adaptive_avg_pool2d(x, (size, size))

        # (batch, channels, H, W) -> (batch, positions, channels), the layout
        # MultiheadAttention expects.
        tokens = reduced.flatten(2).transpose(1, 2)

        normed = self.norm(tokens)
        attended, _ = self.attention(normed, normed, normed)
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.norm2(tokens))

        # Back to a feature map, then to the original resolution.
        out = tokens.transpose(1, 2).reshape(b, c, size, size)
        return F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)


class ReverseTransformerAttention(nn.Module):
    """
    Reverse attention whose attention map comes from a self-attention block.

    Same structure as ReverseAttention in pvt_rta.py -- invert the map, modulate
    the skip, refine, add back -- with the transformer stage inserted before the
    map is produced.
    """

    def __init__(self, gate_ch, skip_ch):
        super().__init__()
        self.transformer = SelfAttentionBlock(gate_ch)
        self.to_attention = nn.Conv2d(gate_ch, 1, kernel_size=1)

        self.refine = nn.Sequential(
            nn.Conv2d(skip_ch, skip_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(skip_ch),
            nn.ReLU(),
            nn.Conv2d(skip_ch, skip_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(skip_ch),
            nn.ReLU(),
        )

    def forward(self, gate, skip):
        attention = self.to_attention(self.transformer(gate))

        if attention.shape[-2:] != skip.shape[-2:]:
            attention = F.interpolate(attention, size=skip.shape[-2:],
                                      mode="bilinear", align_corners=False)

        reverse = 1.0 - torch.sigmoid(attention)
        return self.refine(skip * reverse) + skip


class DecoderBlock(nn.Module):
    """Upsample, merge the reverse-attended skip, then two convolutions."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.reverse_attention = ReverseTransformerAttention(in_ch, skip_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x, skip):
        skip = self.reverse_attention(x, skip)
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class PVTRTAUNet(nn.Module):
    """PVTv2 encoder + reverse-transformer-attention decoder."""

    def __init__(self, num_classes=4, pretrained=True, encoder_name="pvt_v2_b0"):
        super().__init__()

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
        )

        ch = self.encoder.feature_info.channels()

        self.dec3 = DecoderBlock(ch[3], ch[2], ch[2])
        self.dec2 = DecoderBlock(ch[2], ch[1], ch[1])
        self.dec1 = DecoderBlock(ch[1], ch[0], ch[0])

        self.final = nn.Sequential(
            nn.Conv2d(ch[0], ch[0] // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch[0] // 2),
            nn.ReLU(),
            nn.Conv2d(ch[0] // 2, num_classes, kernel_size=1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        x = x.repeat(1, 3, 1, 1)
        f1, f2, f3, f4 = self.encoder(x)

        d3 = self.dec3(f4, f3)
        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2, f1)

        out = self.final(d1)
        return F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
