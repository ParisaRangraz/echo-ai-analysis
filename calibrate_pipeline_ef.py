"""
calibrate_pipeline_ef.py

Measures the EF calibration offset under the conditions the pipeline actually
runs in: model-predicted masks, propagated to ES by registration.

An earlier offset was measured with ground-truth masks and did not transfer --
a correction fitted under one set of conditions does not apply to another.

Sequences are loaded with load_sequence_ed_first, so index 0 is always ED even
for the ~2% of sequences stored in reverse order.

Runs on TRAINING patients only. The offset is then applied to validation, which
keeps the reported validation numbers honest.

A subset is used rather than all 400 patients: each one takes around 3 minutes
(two views, ~19 registrations each), so 400 would take about 20 hours.
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
from volume import lv_volume_biplane_ml, ejection_fraction

#MODEL_PATH = "models/unet_ed_both_views.pt"
#N_PATIENTS = 50
#MODEL_PATH = "models/pvt_ed_both_views.pt"
MODEL_PATH = "models/pvt_rta_both_views.pt"
N_PATIENTS = 50

#model = UNet(num_classes=4)
#model = PVTUNet(num_classes=4, pretrained=False)
model = PVTReverseAttentionUNet(num_classes=4, pretrained=False)
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


def pipeline_ef(pid):
    """Run the full pipeline for one patient and return the uncalibrated EF."""
    masks = {}
    spacings = {}

    for view in ("4CH", "2CH"):
        seq = load_sequence_ed_first(pid, view=view)
        spacings[view] = load_sequence_spacing(pid, view=view)

        ed_mask = predict_mask(seq[0])
        propagated = propagate_mask(seq, ed_mask, spacings[view])

        masks[(view, "ED")] = ed_mask
        masks[(view, "ES")] = propagated[-1]

    edv = lv_volume_biplane_ml(masks[("4CH", "ED")], spacings["4CH"],
                               masks[("2CH", "ED")], spacings["2CH"])
    esv = lv_volume_biplane_ml(masks[("4CH", "ES")], spacings["4CH"],
                               masks[("2CH", "ES")], spacings["2CH"])

    return ejection_fraction(edv, esv)


patient_ids = read_patient_list("training")[:N_PATIENTS]

os.makedirs("results", exist_ok=True)
#csv_path = os.path.join("results", "pipeline_calibration_training.csv")
#csv_path = os.path.join("results", "pipeline_calibration_training_pvt.csv")
csv_path = os.path.join("results", "pipeline_calibration_training_rta.csv")
rows = []
start = time.time()

for i, pid in enumerate(patient_ids):
    ef = pipeline_ef(pid)
    ef_ref = float(parse_info_cfg(pid, "4CH")["EF"])

    rows.append({
        "patient": pid,
        "ef_pipeline": round(ef, 1),
        "ef_reference": ef_ref,
        "error": round(ef - ef_ref, 1),
    })

    elapsed = (time.time() - start) / 60
    print(f"[{i+1:2d}/{len(patient_ids)}] {pid}: "
          f"EF {ef:5.1f} vs reference {ef_ref:5.1f} "
          f"| elapsed {elapsed:.1f} min")

    # Rewrite after every patient, so a long run is never lost.
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

errors = np.array([r["error"] for r in rows])

print(f"\nSaved {len(rows)} rows to {csv_path}")
print(f"\n=== Calibration measured on {len(rows)} training patients ===")
print(f"Mean offset (pipeline - reference):   {errors.mean():+.2f} %")
print(f"Median offset:                        {np.median(errors):+.2f} %")
print(f"Std of the difference:                {errors.std():.1f} %")
print(f"\nThe median is also reported because a single badly failed patient can")
print("move the mean by several points, while the median is unaffected.")
print(f"\nSet EF_OFFSET in run_pipeline.py, then re-run it on validation.")
