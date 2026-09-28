"""
explore_registration.py

Scratch/test script for Module 3 (registration + mask propagation).
Kept separate from explore.py so each exploration file stays focused on one topic.

Scenario: pretend only the ED mask (frame 0) is annotated, propagate it to every
later frame with frame-to-frame deformable registration, and score each frame
against the real ground-truth mask CAMUS provides for that frame.
"""

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from loader import load_half_sequence, load_sequence_spacing
from processing import to_sitk, register_bspline, warp_mask


def dice(a, b, class_id=1):
    """Dice for one class (default: LV cavity)."""
    a, b = (a == class_id), (b == class_id)
    return 2 * np.logical_and(a, b).sum() / (a.sum() + b.sum())


# --- Load everything we need for one patient ---
pid = "patient0001"
seq = load_half_sequence(pid)              # all image frames, ED -> ES: shape (frames, H, W)
seq_gt = load_half_sequence(pid, gt=True)  # the real mask for every frame (used only for scoring)
spacing = load_sequence_spacing(pid)       # real pixel size in mm, needed for registration
n_frames = seq.shape[0]                    # number of frames in this patient's sequence

print("Patient:", pid)
print("Frames:", n_frames)
print("Pixel spacing (mm):", spacing)

# --- Starting point ---
# We pretend we only have the ED mask (frame 0), as in a real clinical setting.
current_mask = to_sitk(seq_gt[0], spacing)

# Dice scores per frame, for two methods. Frame 0 is the ED mask compared
# with itself, so both start at a perfect 1.0.
dice_baseline = [1.0]     # method 1: copy the ED mask unchanged to every frame
dice_propagated = [1.0]   # method 2: carry the mask forward with registration

# --- Walk through the sequence one frame at a time ---
for k in range(1, n_frames):
    # Register the previous frame onto the current frame.
    fixed = to_sitk(seq[k], spacing)        # current frame: stays still (reference)
    moving = to_sitk(seq[k - 1], spacing)   # previous frame: gets deformed

    # Find how the heart moved between these two frames.
    transform = register_bspline(fixed, moving)

    # Apply that same motion to our mask, so it follows the heart.
    # current_mask is overwritten, so the next loop starts from THIS estimate
    # (a chain: errors at each step carry into later steps).
    current_mask = warp_mask(current_mask, transform, fixed)

    # Convert the propagated mask back to numpy, so we can score it.
    propagated = sitk.GetArrayFromImage(current_mask)

    # Score both methods against the real mask of frame k.
    dice_baseline.append(dice(seq_gt[0], seq_gt[k]))     # unchanged ED mask vs. truth
    dice_propagated.append(dice(propagated, seq_gt[k]))  # propagated mask vs. truth

    # Print progress: one line per frame.
    print(f"Frame {k:2d}: baseline {dice_baseline[-1]:.3f} | propagated {dice_propagated[-1]:.3f}")

# --- Plot both methods on the same chart ---
plt.plot(dice_baseline, label="No registration (copy ED mask)")      # line 1
plt.plot(dice_propagated, label="Frame-to-frame registration")      # line 2
plt.xlabel("Frame (0 = ED, last = ES)")                              # x-axis: time
plt.ylabel("LV cavity Dice vs. ground truth")                        # y-axis: accuracy
plt.title(f"Mask propagation - {pid}")
plt.legend()                                                         # show which line is which
plt.show()
