"""
explore_ef.py

Scratch/test script for Module 4 (volume + ejection fraction).
"""

from loader import parse_info_cfg

pid = "patient0001"

for view in ("2CH", "4CH"):
    print(f"--- {view} ---")
    for key, value in parse_info_cfg(pid, view).items():
        print(f"  {key}: {value}")

        import sys
sys.path.append("../evaluation")

import sys
sys.path.append("../evaluation")   # so Python can find volume.py

from loader import load_half_sequence, load_sequence_spacing
from volume import lv_volume_ml, ejection_fraction, classify_ef

pid = "patient0001"

# Load the REAL ground-truth masks from CAMUS (not our model's output).
# Testing the volume maths in isolation: if EF comes out close to the
# reference, the formula is sound, and any later error must come from
# the segmentation/registration masks rather than from this calculation.
seq_gt = load_half_sequence(pid, gt=True)
spacing = load_sequence_spacing(pid)       # real pixel size in mm

edv = lv_volume_ml(seq_gt[0], spacing)     # frame 0 = ED: heart relaxed, largest volume
esv = lv_volume_ml(seq_gt[-1], spacing)    # last frame = ES: heart contracted, smallest
ef = ejection_fraction(edv, esv)           # fraction of blood pumped out, in percent

print(f"\nEDV: {edv:.1f} mL")
print(f"ESV: {esv:.1f} mL")
print(f"EF (computed, 4CH only): {ef:.1f} %  -> {classify_ef(ef)}")

# The CAMUS reference EF is a biplane value (2CH + 4CH combined),
# so an exact match is not expected -- we are comparing single-plane to biplane.
print(f"EF (CAMUS reference, biplane): {parse_info_cfg(pid, '4CH')['EF']} %")

import matplotlib.pyplot as plt
import numpy as np

# --- Check whether the LV long axis is really vertical ---
# Simpson's method assumes each image row is a disk perpendicular to the
# long axis. If the LV is tilted, row widths overestimate the true diameter,
# and since diameter is squared, the volume error grows quickly.

ed_mask = seq_gt[0]
lv = (ed_mask == 1)                 # LV cavity only
row_widths = lv.sum(axis=1)         # number of LV pixels per row

# Find the horizontal centre of the LV in each row that contains LV pixels.
# If the long axis is vertical, these centres form a straight vertical line.
centres = []
rows_with_lv = []
for r in range(lv.shape[0]):
    cols = np.where(lv[r])[0]       # column indices of LV pixels in this row
    if len(cols) > 0:
        centres.append(cols.mean())
        rows_with_lv.append(r)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(ed_mask, cmap="viridis", vmin=0, vmax=3)
plt.plot(centres, rows_with_lv, "r-")   # red line: the LV centre line
plt.title("ED mask + LV centre line")

plt.subplot(1, 3, 2)
plt.imshow(lv, cmap="gray")
plt.title("LV cavity only")

plt.subplot(1, 3, 3)
plt.plot(row_widths)
plt.xlabel("Image row (top to bottom)")
plt.ylabel("LV width in pixels")
plt.title("Row widths (disk diameters)")

plt.tight_layout()
plt.show()

# --- Compare mono-plane and biplane volume estimates ---
# Testing the hypothesis that the EF gap (62.8% vs 54%) comes from using
# only one view, not from an error in the volume calculation itself.
from volume import lv_volume_biplane_ml

seq_gt_4ch = load_half_sequence(pid, view="4CH", gt=True)
seq_gt_2ch = load_half_sequence(pid, view="2CH", gt=True)
spacing_4ch = load_sequence_spacing(pid, view="4CH")
spacing_2ch = load_sequence_spacing(pid, view="2CH")

edv_bi = lv_volume_biplane_ml(seq_gt_4ch[0], spacing_4ch, seq_gt_2ch[0], spacing_2ch)
esv_bi = lv_volume_biplane_ml(seq_gt_4ch[-1], spacing_4ch, seq_gt_2ch[-1], spacing_2ch)
ef_bi = ejection_fraction(edv_bi, esv_bi)

print(f"\n--- Biplane (both views) ---")
print(f"EDV: {edv_bi:.1f} mL")
print(f"ESV: {esv_bi:.1f} mL")
print(f"EF: {ef_bi:.1f} %  -> {classify_ef(ef_bi)}")
print(f"EF reference: {parse_info_cfg(pid, '4CH')['EF']} %")