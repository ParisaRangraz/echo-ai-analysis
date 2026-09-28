"""
explore_check.py

Two earlier scripts reported different LV pixel counts for what should be the
same frames. This prints everything explicitly -- patient, view, frame index,
and the ED/ES frame numbers from the cfg -- so there is nothing to misread.
"""

import numpy as np
from loader import load_half_sequence, parse_info_cfg

for pid in ("patient0001", "patient0006"):
    for view in ("4CH", "2CH"):
        gt = load_half_sequence(pid, view=view, gt=True)
        info = parse_info_cfg(pid, view)
        counts = [(m == 1).sum() for m in gt]

        print(f"\n===== {pid}  {view} =====")
        print(f"cfg says:  ED frame = {info['ED']}, ES frame = {info['ES']}, "
              f"NbFrame = {info['NbFrame']}")
        print(f"array has: {len(gt)} frames")
        print(f"LV pixels at index 0:   {counts[0]}")
        print(f"LV pixels at index -1:  {counts[-1]}")
        print(f"max pixels at index {int(np.argmax(counts))}, "
              f"min at index {int(np.argmin(counts))}")