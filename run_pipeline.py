"""
run_pipeline.py

The full pipeline, end to end, with no ground-truth masks used as input:

    ED frame (2CH and 4CH)
        -> U-Net             -> predicted ED mask
        -> registration      -> predicted ES mask
        -> Simpson biplane   -> EDV, ESV
        -> calibration       -> EF -> Normal / Reduced

Ground-truth masks are used only to score the result at the end.

Sequences are loaded with load_sequence_ed_first, which reads the true ED/ES
frame numbers from Info_*.cfg and reverses the sequence where needed: about 2%
of sequences run ES -> ED, and it can differ between the two views of the same
patient. Index 0 is therefore always ED and the last index always ES.
"""

import sys
sys.path.append("data")
sys.path.append("models")
sys.path.append("evaluation")

import os
import csv
import time
import numpy as np
import torch

from loader import (read_patient_list, load_sequence_ed_first,
                    load_sequence_spacing, parse_info_cfg, resize_image)
from processing import propagate_mask
#from unet import UNet
#from pvt_unet import PVTUNet
from pvt_rta import PVTReverseAttentionUNet

from volume import lv_volume_biplane_ml, ejection_fraction, classify_ef
from metrics import dice_score

#MODEL_PATH = "models/unet_ed_both_views.pt"
#EF_OFFSET = 0.0    # to be set once calibrate_pipeline_ef.py has been re-run
#EF_OFFSET = 1.13   # mean offset measured on 50 training patients
#MODEL_PATH = "models/pvt_ed_both_views.pt"
#EF_OFFSET = 1.04
MODEL_PATH = "models/pvt_rta_both_views.pt"
EF_OFFSET = 0.95
LV = 1


# Set to a small number for a quick test run, or None to process everyone.
LIMIT = None

#model = UNet(num_classes=4)
#model = PVTUNet(num_classes=4, pretrained=False)
model = PVTReverseAttentionUNet(num_classes=4, pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()


def predict_mask(frame):
    """
    Run the U-Net on one frame.

    The model works on 256x256 inputs, but registration and volume measurement
    need the original resolution and pixel spacing, so the predicted mask is
    resized back to the frame's own size (nearest-neighbor, to keep the class
    labels as whole numbers).
    """
    original_shape = frame.shape

    resized = resize_image(frame.astype(np.float32), (256, 256), is_mask=False)
    normalized = resized / 255.0

    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        prediction = model(tensor).argmax(dim=1).squeeze(0).numpy()

    return resize_image(prediction.astype(np.float32), original_shape, is_mask=True)


def process_view(pid, view):
    """
    Predict the ED mask, propagate it to ES, and score both against ground truth.

    Returns the predicted ED and ES masks, the spacing, and the two Dice scores.
    """
    seq = load_sequence_ed_first(pid, view=view)
    seq_gt = load_sequence_ed_first(pid, view=view, gt=True)
    spacing = load_sequence_spacing(pid, view=view)

    # Stage 1: segmentation on the ED frame only.
    ed_mask = predict_mask(seq[0])

    # Stage 2: carry that predicted mask through the sequence.
    propagated = propagate_mask(seq, ed_mask, spacing)
    es_mask = propagated[-1]

    # Scoring only -- these ground-truth masks were not used above.
    dice_ed = dice_score(ed_mask, seq_gt[0], LV)
    dice_es = dice_score(es_mask, seq_gt[-1], LV)

    return ed_mask, es_mask, spacing, dice_ed, dice_es


#patient_ids = read_patient_list("validation")
patient_ids = read_patient_list("testing")
if LIMIT:
    patient_ids = patient_ids[:LIMIT]

os.makedirs("results", exist_ok=True)
#csv_path = os.path.join("results", "pipeline_validation.csv")
#csv_path = os.path.join("results", "pipeline_validation_pvt.csv")
#csv_path = os.path.join("results", "pipeline_validation_rta.csv")
csv_path = os.path.join("results", "pipeline_test_rta.csv")
rows = []
start = time.time()

for i, pid in enumerate(patient_ids):
    print(f"Starting {pid} ...")

    ed_4ch, es_4ch, sp_4ch, dice_ed_4ch, dice_es_4ch = process_view(pid, "4CH")
    ed_2ch, es_2ch, sp_2ch, dice_ed_2ch, dice_es_2ch = process_view(pid, "2CH")

    # Stage 3: volumes from the predicted masks.
    edv = lv_volume_biplane_ml(ed_4ch, sp_4ch, ed_2ch, sp_2ch)
    esv = lv_volume_biplane_ml(es_4ch, sp_4ch, es_2ch, sp_2ch)

    ef_raw = ejection_fraction(edv, esv)
    ef_calibrated = ef_raw - EF_OFFSET

    info = parse_info_cfg(pid, "4CH")
    ef_ref = float(info["EF"])

    rows.append({
        "patient": pid,
        "image_quality": info.get("ImageQuality", ""),
        "dice_ed_4ch": round(dice_ed_4ch, 4),
        "dice_es_4ch": round(dice_es_4ch, 4),
        "dice_ed_2ch": round(dice_ed_2ch, 4),
        "dice_es_2ch": round(dice_es_2ch, 4),
        "edv": round(edv, 1),
        "esv": round(esv, 1),
        "ef_raw": round(ef_raw, 1),
        "ef_calibrated": round(ef_calibrated, 1),
        "ef_reference": ef_ref,
        "error": round(ef_calibrated - ef_ref, 1),
        "predicted_class": classify_ef(ef_calibrated),
        "reference_class": classify_ef(ef_ref),
    })

    elapsed = (time.time() - start) / 60
    print(f"[{i+1:2d}/{len(patient_ids)}] {pid}: "
          f"EF {ef_calibrated:5.1f} vs reference {ef_ref:5.1f} "
          f"| Dice ED {dice_ed_4ch:.3f} ES {dice_es_4ch:.3f} (4CH) "
          f"| elapsed {elapsed:.1f} min")

    # Rewrite the CSV after every patient, so a long run is never lost.
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

print(f"\nSaved {len(rows)} rows to {csv_path}")


# --- Summary ---
def stats(key):
    values = np.array([r[key] for r in rows])
    return f"{values.mean():.3f} +/- {values.std():.3f}"


errors = np.array([r["error"] for r in rows])
same_class = sum(r["predicted_class"] == r["reference_class"] for r in rows)

print(f"\n=== End-to-end results ({len(rows)} patients) ===")
print(f"LV Dice, predicted ED mask (4CH): {stats('dice_ed_4ch')}")
print(f"LV Dice, propagated ES mask (4CH): {stats('dice_es_4ch')}")
print(f"LV Dice, predicted ED mask (2CH): {stats('dice_ed_2ch')}")
print(f"LV Dice, propagated ES mask (2CH): {stats('dice_es_2ch')}")
print(f"\nEF bias:                   {errors.mean():+.1f} %")
print(f"EF std of the difference:  {errors.std():.1f} %")
print(f"EF mean absolute error:    {np.abs(errors).mean():.1f} %")
print(f"Same Normal/Reduced class: {same_class}/{len(rows)}")
