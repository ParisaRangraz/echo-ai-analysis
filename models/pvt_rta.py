"""
pvt_rta.py

PVTv2 encoder with a reverse-attention decoder, adapted from RTA-Former
(Li, Yi, Uneri, Niu & Jones, EMBC 2024, arXiv:2401.11671).

The difference from pvt_unet.py is the decoder's skip-connection module.

An attention gate asks "which regions matter?" and the model answers with the
structure's interior, because that is what drives most of the loss. Reverse
attention inverts that answer: subtracting the attention map from 1 leaves the
regions the model is *least* confident about, which is where the boundary is.
Those features are then refined and added back, so the decoder gets both the
confident interior and a separately processed boundary signal.

Two deliberate simplifications relative to the paper, both noted so the
comparison is honest:
  - the paper reuses a transformer stage inside the reverse-attention branch;
    here a convolutional bottleneck is used instead, to keep the parameter
    count close to pvt_unet.py so the two can be compared fairly. Their
    ablation shows the transformer version gains roughly twice what the
    convolutional one does, so this is the weaker variant.
  - the paper's Fast Feature Fusion (learnable weights + Swish) is not
    implemented; skips are concatenated as in a standard U-Net.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ReverseAttention(nn.Module):
    """
    Reverse attention on one skip connection.

    The decoder feature (gate) produces an attention map; that map is inverted,
    so what remains highlights the periphery rather than the interior. The skip
    is modulated by the inverted map, refined, and added back to itself -- a
    residual connection, so the module can only add information, never remove
    the original skip.
    """

    def __init__(self, gate_ch, skip_ch):
        super().__init__()

        # Collapse the decoder feature to a single-channel attention map.
        # One channel because an attention value is a property of a location,
        # not of a channel.
        self.to_attention = nn.Conv2d(gate_ch, 1, kernel_size=1)

        # Refines whatever survives the inversion -- i.e. the boundary region.
        self.refine = nn.Sequential(
            nn.Conv2d(skip_ch, skip_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(skip_ch),
            nn.ReLU(),
            nn.Conv2d(skip_ch, skip_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(skip_ch),
            nn.ReLU(),
        )

    def forward(self, gate, skip):
        attention = self.to_attention(gate)

        if attention.shape[-2:] != skip.shape[-2:]:
            attention = F.interpolate(attention, size=skip.shape[-2:],
                                      mode="bilinear", align_corners=False)

        # Sigmoid puts the map in [0, 1] so that "1 - map" is meaningful.
        # High where the model is confident about the interior; the inverse is
        # therefore high at the edges and in the background.
        reverse = 1.0 - torch.sigmoid(attention)

        return self.refine(skip * reverse) + skip


class DecoderBlock(nn.Module):
    """Upsample, merge the reverse-attended skip, then two convolutions."""

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.reverse_attention = ReverseAttention(in_ch, skip_ch)
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


class PVTReverseAttentionUNet(nn.Module):
    """
    PVTv2 encoder + reverse-attention decoder.

    Identical to PVTUNet except that the skip connections pass through
    ReverseAttention instead of an attention gate, so any difference in the
    results is attributable to that one change.
    """

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

        # Grayscale -> 3 channels, to match what the pretrained encoder expects.
        x = x.repeat(1, 3, 1, 1)

        f1, f2, f3, f4 = self.encoder(x)

        d3 = self.dec3(f4, f3)
        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2, f1)

        out = self.final(d1)
        return F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)
