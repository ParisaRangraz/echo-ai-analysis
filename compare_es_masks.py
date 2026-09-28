"""
compare_es_masks.py

The end-to-end run showed EF systematically too low (bias -9.4%). The likely
cause is the ES mask: Dice drops from 0.946 at ED to 0.903 at ES after
propagation, and an ES mask that stays too large makes ESV too large, which
makes EF too small.

This script tests a simpler alternative: run the U-Net directly on the ES frame
instead of carrying the ED mask there with registration. It compares both
approaches on the same patients, and recomputes EF for each.

Registration remains necessary for the intermediate frames (a volume-time curve
needs all of them) -- but EF only needs ED and ES, so if direct segmentation is
better at ES, EF should use it.

Fast: no registration involved, so this runs in minutes rather than hours.
"""

import sys
sys.path.append("data")
sys.path.append("models")
sys.path.append("evaluation")

import os
import csv
import numpy as np
import torch

from loader import (read_patient_list, load_half_sequence,
                    load_sequence_spacing, parse_info_cfg, resize_image)
from unet import UNet
from volume import lv_volume_biplane_ml, ejection_fraction, classify_ef
from metrics import dice_score

MODEL_PATH = "models/unet_ed_both_views.pt"
LV = 1

model = UNet(num_classes=4)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()


def predict_mask(frame):
    """Run the U-Net on one frame and return the mask at the frame's own size."""
    original_shape = frame.shape

    resized = resize_image(frame.astype(np.float32), (256, 256), is_mask=False)
    normalized = resized / 255.0

    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        prediction = model(tensor).argmax(dim=1).squeeze(0).numpy()

    return resize_image(prediction.astype(np.float32), original_shape, is_mask=True)


patient_ids = read_patient_list("validation")
os.makedirs("results", exist_ok=True)
rows = []

for i, pid in enumerate(patient_ids):
    masks = {}
    spacings = {}
    dice_es = {}

    for view in ("4CH", "2CH"):
        seq = load_half_sequence(pid, view=view)
        seq_gt = load_half_sequence(pid, view=view, gt=True)
        spacings[view] = load_sequence_spacing(pid, view=view)

        # Segment ED and ES directly -- no registration in this path.
        masks[(view, "ED")] = predict_mask(seq[0])
        masks[(view, "ES")] = predict_mask(seq[-1])

        dice_es[view] = dice_score(masks[(view, "ES")], seq_gt[-1], LV)

    edv = lv_volume_biplane_ml(masks[("4CH", "ED")], spacings["4CH"],
                               masks[("2CH", "ED")], spacings["2CH"])
    esv = lv_volume_biplane_ml(masks[("4CH", "ES")], spacings["4CH"],
                               masks[("2CH", "ES")], spacings["2CH"])

    ef_raw = ejection_fraction(edv, esv)
    ef_ref = float(parse_info_cfg(pid, "4CH")["EF"])

    rows.append({
        "patient": pid,
        "dice_es_4ch_direct": round(dice_es["4CH"], 4),
        "dice_es_2ch_direct": round(dice_es["2CH"], 4),
        "edv": round(edv, 1),
        "esv": round(esv, 1),
        "ef_raw": round(ef_raw, 1),
        "ef_reference": ef_ref,
        "error_raw": round(ef_raw - ef_ref, 1),
    })

    print(f"[{i+1:2d}/{len(patient_ids)}] {pid}: "
          f"EF {ef_raw:5.1f} vs reference {ef_ref:5.1f} "
          f"| ES Dice 4CH {dice_es['4CH']:.3f}")

csv_path = os.path.join("results", "es_direct_validation.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"\nSaved {len(rows)} rows to {csv_path}")


def summarize(values):
    values = np.array(values)
    return f"{values.mean():.3f} +/- {values.std():.3f}"


errors = np.array([r["error_raw"] for r in rows])

print(f"\n=== Direct ES segmentation ({len(rows)} patients) ===")
print(f"ES Dice, 4CH: {summarize([r['dice_es_4ch_direct'] for r in rows])}")
print(f"ES Dice, 2CH: {summarize([r['dice_es_2ch_direct'] for r in rows])}")
print("(for comparison, registration-propagated: 4CH 0.903, 2CH 0.878)")

print(f"\nEF before any calibration:")
print(f"  bias:                  {errors.mean():+.1f} %")
print(f"  std of the difference: {errors.std():.1f} %")
print(f"  mean absolute error:   {np.abs(errors).mean():.1f} %")

# What a calibration would look like, if the same offset were applied.
calibrated_errors = errors - errors.mean()
same_class = sum(
    classify_ef(r["ef_raw"] - errors.mean()) == classify_ef(r["ef_reference"])
    for r in rows
)
print(f"\nIf calibrated by its own mean offset ({errors.mean():+.2f}):")
print(f"  std stays:                 {calibrated_errors.std():.1f} %")
print(f"  mean absolute error:       {np.abs(calibrated_errors).mean():.1f} %")
print(f"  same Normal/Reduced class: {same_class}/{len(rows)}")
print("\nNote: that offset is fitted on validation, so it is shown only as an")
print("upper bound on what calibration could achieve. A real correction must")
print("be measured on the training split.")
