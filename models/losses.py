"""
losses.py

Custom loss functions for segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class DiceLoss(nn.Module):
    """
    Multi-class Dice loss. Computes Dice per class, then averages.
    """

    def __init__(self, num_classes=4, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)

        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        intersection = (probs * targets_one_hot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))

        dice_per_class = (2 * intersection + self.smooth) / (union + self.smooth)

        mean_dice = dice_per_class.mean()
        return 1 - mean_dice


class CombinedLoss(nn.Module):
    """
    Combines CrossEntropyLoss and DiceLoss.
    """

    def __init__(self, num_classes=4):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes=num_classes)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return ce_loss + dice_loss

   


class BoundaryWeightedCELoss(nn.Module):
    """
    Cross-entropy that weights pixels near structure boundaries more heavily.

    Dice and plain cross-entropy treat every pixel equally, so most of the
    training signal comes from easy interior pixels. Boundaries are where the
    errors actually are -- and for a thin structure like the myocardium, they
    are almost the whole structure.

    The weight map is derived from the ground-truth mask itself: a
    morphological gradient (dilation minus erosion) marks the edges, then a
    blur spreads the weight into a band around them.
    """

    def __init__(self, boundary_weight=3.0, band_width=5):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.band_width = band_width

    def boundary_map(self, masks):
        """
        Build a per-pixel weight map: 1.0 everywhere, higher near boundaries.

        Args:
            masks: (batch, H, W) of integer class labels
        Returns:
            (batch, H, W) of float weights
        """
        x = masks.unsqueeze(1).float()   # (batch, 1, H, W) for pooling

        # Max-pooling with stride 1 is dilation; min-pooling (max of the
        # negative) is erosion. Where they differ, there is an edge.
        dilated = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        edges = (dilated != eroded).float()

        # Spread the edge into a band, so pixels near a boundary also count.
        band = F.avg_pool2d(edges, kernel_size=self.band_width,
                            stride=1, padding=self.band_width // 2)
        band = (band > 0).float()

        weights = 1.0 + (self.boundary_weight - 1.0) * band
        return weights.squeeze(1)

    def forward(self, logits, targets):
        # reduction="none" keeps the per-pixel loss, so it can be weighted.
        ce = F.cross_entropy(logits, targets, reduction="none")
        weights = self.boundary_map(targets)
        return (ce * weights).mean()


class BoundaryCombinedLoss(nn.Module):
    """
    Dice + boundary-weighted cross-entropy.

    Same structure as CombinedLoss, with the plain cross-entropy term replaced.
    Dice keeps the class balance under control; the weighted term concentrates
    the pixel-level signal where the boundaries are.
    """

    def __init__(self, num_classes=4, boundary_weight=3.0):
        super().__init__()
        self.dice = DiceLoss(num_classes=num_classes)
        self.boundary_ce = BoundaryWeightedCELoss(boundary_weight=boundary_weight)

    def forward(self, logits, targets):
        return self.dice(logits, targets) + self.boundary_ce(logits, targets)