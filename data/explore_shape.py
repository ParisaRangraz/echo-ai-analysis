"""
explore_shape.py

The LV pixel count falls from ED to ES, but the computed volume rises. Since
Simpson's method squares the width of each row, the shape of the mask matters
much more than its area. This inspects the shape directly.
"""

import numpy as np
import matplotlib.pyplot as plt

from loader import load_half_sequence, load_sequence_spacing

PID = "patient0006"
VIEW = "4CH"

gt = load_half_sequence(PID, view=VIEW, gt=True)
spacing = load_sequence_spacing(PID, view=VIEW)

ed = (gt[0] == 1)
es = (gt[-1] == 1)

for name, mask in (("ED", ed), ("ES", es)):
    rows = np.where(mask.any(axis=1))[0]
    widths = mask.sum(axis=1)

    print(f"\n{name}:")
    print(f"  pixels:        {mask.sum()}")
    print(f"  rows spanned:  {len(rows)}  (from {rows[0]} to {rows[-1]})")
    print(f"  widest row:    {widths.max()} px")
    print(f"  mean width:    {widths[widths > 0].mean():.1f} px")

plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(ed, cmap="gray")
plt.title("ED: LV cavity")

plt.subplot(1, 3, 2)
plt.imshow(es, cmap="gray")
plt.title("ES: LV cavity")

plt.subplot(1, 3, 3)
plt.plot(ed.sum(axis=1), label="ED")
plt.plot(es.sum(axis=1), label="ES")
plt.xlabel("Image row")
plt.ylabel("LV width (px)")
plt.title("Row widths")
plt.legend()

plt.tight_layout()
plt.show()