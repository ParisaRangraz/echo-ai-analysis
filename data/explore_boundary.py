"""
explore_boundary.py

Checks the boundary weight map visually before spending hours training with it.
A weight map that highlights the wrong pixels would train the model to care
about the wrong thing, and the loss curve would not reveal it.
"""

import sys
sys.path.append("../models")

import torch
import numpy as np
import matplotlib.pyplot as plt

from loader import load_patient_frame, resize_image
from losses import BoundaryWeightedCELoss

PID = "patient0001"

img, mask = load_patient_frame(PID, view="4CH", phase="ED")
mask = resize_image(mask, (256, 256), is_mask=True)

# The loss works on batches, so add a batch dimension.
mask_tensor = torch.from_numpy(mask.astype(np.int64)).unsqueeze(0)

loss_fn = BoundaryWeightedCELoss(boundary_weight=3.0, band_width=5)
weights = loss_fn.boundary_map(mask_tensor).squeeze(0).numpy()

print(f"Weight map: min {weights.min():.1f}, max {weights.max():.1f}")
print(f"Fraction of pixels with raised weight: "
      f"{(weights > 1.0).mean() * 100:.1f}%")

plt.figure(figsize=(14, 4))

plt.subplot(1, 3, 1)
plt.imshow(mask, cmap="viridis", vmin=0, vmax=3)
plt.title("Ground-truth mask")

plt.subplot(1, 3, 2)
plt.imshow(weights, cmap="hot")
plt.colorbar()
plt.title("Boundary weight map")

# Overlay: the weight map should trace the mask's edges exactly.
plt.subplot(1, 3, 3)
plt.imshow(mask, cmap="gray")
plt.imshow(np.where(weights > 1.0, 1.0, np.nan), cmap="autumn", alpha=0.6)
plt.title("Weighted band over the mask")

plt.tight_layout()
plt.show()