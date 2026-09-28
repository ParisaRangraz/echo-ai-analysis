"""
explore_orientation.py

Checks the long-axis detection visually before running it over 50 patients.

If the axis is found in the wrong place, the rotated mask would be wrong too,
and the volume numbers would change for the wrong reason -- with nothing in the
output to reveal it.
"""

import sys
sys.path.append("../evaluation")

import numpy as np
import matplotlib.pyplot as plt

from loader import load_sequence_ed_first, load_sequence_spacing
from volume import lv_volume_ml, lv_volume_biplane_ml
from volume_oriented import (find_long_axis, rotate_mask,
                             lv_volume_oriented_ml, lv_volume_biplane_oriented_ml)

PID = "patient0001"

gt_4ch = load_sequence_ed_first(PID, view="4CH", gt=True)
gt_2ch = load_sequence_ed_first(PID, view="2CH", gt=True)
sp_4ch = load_sequence_spacing(PID, view="4CH")
sp_2ch = load_sequence_spacing(PID, view="2CH")

ed_4ch, es_4ch = gt_4ch[0], gt_4ch[-1]

for name, mask in (("ED", ed_4ch), ("ES", es_4ch)):
    apex, base, angle = find_long_axis(mask)
    print(f"{name}: apex at row {apex[0]}, col {apex[1]:.0f}; "
          f"base at row {base[0]}, col {base[1]:.0f}; tilt {angle:+.1f} degrees")

# --- How much does the volume change? ---
edv_plain = lv_volume_biplane_ml(gt_4ch[0], sp_4ch, gt_2ch[0], sp_2ch)
esv_plain = lv_volume_biplane_ml(gt_4ch[-1], sp_4ch, gt_2ch[-1], sp_2ch)

edv_rot = lv_volume_biplane_oriented_ml(gt_4ch[0], sp_4ch, gt_2ch[0], sp_2ch)
esv_rot = lv_volume_biplane_oriented_ml(gt_4ch[-1], sp_4ch, gt_2ch[-1], sp_2ch)

print(f"\nBiplane volumes, image-aligned disks:  EDV {edv_plain:.1f}, ESV {esv_plain:.1f}, "
      f"EF {(edv_plain - esv_plain) / edv_plain * 100:.1f} %")
print(f"Biplane volumes, axis-aligned disks:   EDV {edv_rot:.1f}, ESV {esv_rot:.1f}, "
      f"EF {(edv_rot - esv_rot) / edv_rot * 100:.1f} %")

# --- Visual check ---
fig, axes = plt.subplots(2, 2, figsize=(10, 9))

for row, (name, mask) in enumerate((("ED", ed_4ch), ("ES", es_4ch))):
    apex, base, angle = find_long_axis(mask)

    axes[row, 0].imshow(mask, cmap="viridis", vmin=0, vmax=3)
    axes[row, 0].plot([apex[1], base[1]], [apex[0], base[0]], "r-", linewidth=2)
    axes[row, 0].plot(apex[1], apex[0], "wo")
    axes[row, 0].plot(base[1], base[0], "wo")
    axes[row, 0].set_title(f"{name}: detected long axis ({angle:+.1f} deg)")

    axes[row, 1].imshow(rotate_mask(mask, angle), cmap="viridis", vmin=0, vmax=3)
    axes[row, 1].set_title(f"{name}: rotated upright")

plt.tight_layout()
plt.show()
