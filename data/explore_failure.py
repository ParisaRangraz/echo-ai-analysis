"""
explore_failure.py

Some patients produce a negative EF, which is physically impossible: it means
the computed ES volume came out larger than the ED volume. This inspects one
such case visually, to find out what the propagated mask actually looks like.
"""

import sys
sys.path.append("../models")
sys.path.append("../evaluation")

import numpy as np
import torch
import matplotlib.pyplot as plt

from loader import load_half_sequence, load_sequence_spacing, resize_image
from processing import propagate_mask
from unet import UNet
from volume import lv_volume_ml

PID = "patient0006"   # negative EF in the calibration run
VIEW = "4CH"

model = UNet(num_classes=4)
model.load_state_dict(torch.load("../models/unet_ed_both_views.pt", weights_only=True))
model.eval()


def predict_mask(frame):
    original_shape = frame.shape
    resized = resize_image(frame.astype(np.float32), (256, 256), is_mask=False)
    tensor = torch.from_numpy(resized / 255.0).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        prediction = model(tensor).argmax(dim=1).squeeze(0).numpy()
    return resize_image(prediction.astype(np.float32), original_shape, is_mask=True)


seq = load_half_sequence(PID, view=VIEW)
seq_gt = load_half_sequence(PID, view=VIEW, gt=True)
spacing = load_sequence_spacing(PID, view=VIEW)

ed_mask = predict_mask(seq[0])
propagated = propagate_mask(seq, ed_mask, spacing)
es_mask = propagated[-1]

# Volume at every frame: a healthy curve should fall from ED to ES.
volumes = [lv_volume_ml(m, spacing) for m in propagated]
volumes_gt = [lv_volume_ml(m, spacing) for m in seq_gt]

print(f"{PID}, {VIEW}")
print(f"Predicted ED volume: {volumes[0]:.1f} mL   (truth {volumes_gt[0]:.1f})")
print(f"Predicted ES volume: {volumes[-1]:.1f} mL   (truth {volumes_gt[-1]:.1f})")

plt.figure(figsize=(14, 7))

plt.subplot(2, 3, 1)
plt.imshow(seq[0], cmap="gray")
plt.title("ED frame")

plt.subplot(2, 3, 2)
plt.imshow(ed_mask, cmap="viridis", vmin=0, vmax=3)
plt.title("Predicted ED mask")

plt.subplot(2, 3, 3)
plt.imshow(seq_gt[0], cmap="viridis", vmin=0, vmax=3)
plt.title("True ED mask")

plt.subplot(2, 3, 4)
plt.imshow(seq[-1], cmap="gray")
plt.title("ES frame")

plt.subplot(2, 3, 5)
plt.imshow(es_mask, cmap="viridis", vmin=0, vmax=3)
plt.title("Propagated ES mask")

plt.subplot(2, 3, 6)
plt.imshow(seq_gt[-1], cmap="viridis", vmin=0, vmax=3)
plt.title("True ES mask")

plt.tight_layout()
plt.show()

# The volume-time curve: where does it go wrong?
plt.figure(figsize=(8, 5))
plt.plot(volumes, label="Propagated masks")
plt.plot(volumes_gt, label="Ground truth")
plt.xlabel("Frame (0 = ED, last = ES)")
plt.ylabel("LV volume (mL)")
plt.title(f"Volume over the cardiac cycle - {PID} ({VIEW})")
plt.legend()
plt.show()