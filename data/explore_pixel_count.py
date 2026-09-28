"""
explore_pixel_count.py

Checks the raw LV pixel count per frame in the ground-truth masks -- no volume
formula involved. If the count rises from frame 0 to the last frame, the issue
is in the data or the frame ordering, not in the volume calculation.
"""

import numpy as np
from loader import load_half_sequence, parse_info_cfg

for pid in ("patient0001", "patient0006"):
    for view in ("4CH", "2CH"):
        gt = load_half_sequence(pid, view=view, gt=True)
        info = parse_info_cfg(pid, view)

        counts = [(m == 1).sum() for m in gt]

        print(f"\n{pid} {view}: ED={info['ED']}, ES={info['ES']}, NbFrame={info['NbFrame']}")
        print(f"  LV pixels, frame 0:    {counts[0]}")
        print(f"  LV pixels, last frame: {counts[-1]}")
        print(f"  smallest at frame:     {int(np.argmin(counts))}")
        print(f"  largest at frame:      {int(np.argmax(counts))}")