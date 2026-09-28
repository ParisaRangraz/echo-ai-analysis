"""
evaluate_registration.py

Evaluates ED-mask propagation (frame-to-frame B-spline registration) on
every validation patient, and compares it with the no-registration baseline
(copying the ED mask unchanged to every frame).

Writes two CSV files under results/:
  registration_validation.csv        one row per patient (summary)
  registration_per_frame.csv         one row per frame (for the Dice-vs-time curve)

Patients have different numbers of frames, so each frame also gets a
"cycle_position" between 0 (ED) and 1 (ES). That shared scale is what makes
curves from different patients comparable and averageable.
"""

import sys
sys.path.append("data")
sys.path.append("evaluation")

import os
import csv
import time
import numpy as np

from loader import read_patient_list, load_half_sequence, load_sequence_spacing
from processing import propagate_mask
from metrics import dice_score

LV = 1  # class id of the LV cavity

patient_ids = read_patient_list("validation")
# patient_ids = patient_ids[:2]   # uncomment for a quick test run on 2 patients

os.makedirs("results", exist_ok=True)
summary_rows = []
frame_rows = []
start = time.time()

for i, pid in enumerate(patient_ids):
    print(f"Starting {pid} ...")

    seq = load_half_sequence(pid)              # image frames, ED -> ES
    seq_gt = load_half_sequence(pid, gt=True)  # real mask for every frame (scoring only)
    spacing = load_sequence_spacing(pid)       # physical pixel size (mm)

    # Pretend only the ED mask is known, and propagate it forward.
    propagated = propagate_mask(seq, seq_gt[0], spacing)
    n = len(propagated)

    # Score both methods on every frame.
    per_frame_baseline = []
    per_frame_propagated = []

    for k in range(n):
        d_base = dice_score(seq_gt[0], seq_gt[k], LV)        # ED mask copied unchanged
        d_prop = dice_score(propagated[k], seq_gt[k], LV)    # mask carried by registration

        per_frame_baseline.append(d_base)
        per_frame_propagated.append(d_prop)

        frame_rows.append({
            "patient": pid,
            "frame": k,
            # 0.0 at ED, 1.0 at ES -- lets patients with different frame counts
            # be plotted and averaged on one common axis.
            "cycle_position": round(k / (n - 1), 4),
            "dice_baseline": round(d_base, 4),
            "dice_propagated": round(d_prop, 4),
        })

    summary_rows.append({
        "patient": pid,
        "n_frames": n,
        "es_dice_baseline": round(per_frame_baseline[-1], 4),
        "es_dice_propagated": round(per_frame_propagated[-1], 4),
        # Mean over all frames after ED (frame 0 is trivially perfect).
        "mean_dice_propagated": round(float(np.mean(per_frame_propagated[1:])), 4),
    })

    elapsed = (time.time() - start) / 60
    print(f"[{i+1:2d}/{len(patient_ids)}] {pid}: "
          f"ES baseline {per_frame_baseline[-1]:.3f} | "
          f"ES propagated {per_frame_propagated[-1]:.3f} "
          f"| elapsed {elapsed:.1f} min")


def write_csv(path, rows):
    """Write a list of dictionaries to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {path}")


write_csv(os.path.join("results", "registration_validation.csv"), summary_rows)
write_csv(os.path.join("results", "registration_per_frame.csv"), frame_rows)


# --- Summary across all patients ---
def summarize(key):
    values = np.array([r[key] for r in summary_rows])
    return f"{values.mean():.3f} +/- {values.std():.3f}"


print(f"\nValidation patients: {len(summary_rows)}")
print("LV cavity Dice at ES, no registration:  ", summarize("es_dice_baseline"))
print("LV cavity Dice at ES, with registration:", summarize("es_dice_propagated"))
print("LV cavity Dice, mean over all frames:   ", summarize("mean_dice_propagated"))

improved = sum(r["es_dice_propagated"] > r["es_dice_baseline"] for r in summary_rows)
print(f"Registration better than baseline at ES for {improved}/{len(summary_rows)} patients")
