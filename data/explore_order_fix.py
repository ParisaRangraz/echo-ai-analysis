"""
explore_order_fix.py

Verifies that load_sequence_ed_first puts ED at index 0 for every sequence,
including the reversed ones.
"""

import numpy as np
from loader import read_patient_list, load_sequence_ed_first

problems = []

for split in ("training", "validation"):
    for pid in read_patient_list(split):
        for view in ("4CH", "2CH"):
            gt = load_sequence_ed_first(pid, view=view, gt=True)
            counts = [(m == 1).sum() for m in gt]

            # ED is the largest LV area, ES the smallest.
            if counts[0] <= counts[-1]:
                problems.append((split, pid, view, counts[0], counts[-1]))

    print(f"{split}: checked")

print(f"\nSequences where index 0 is not the largest: {len(problems)}")
for p in problems[:10]:
    print(f"   {p[0]} {p[1]} {p[2]}: first={p[3]}, last={p[4]}")