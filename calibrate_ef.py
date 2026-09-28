"""
calibrate_ef.py

Simpson's biplane method systematically overestimates EF on this dataset.
This script measures that offset on the TRAINING patients only, then applies
it as a fixed correction to the validation patients.

Fitting the correction on training and reporting it on validation is what
makes the reported numbers honest: fitting it on the validation set and then
reporting improvement on that same set would be circular.
"""

import sys
sys.path.append("data")
sys.path.append("evaluation")

import numpy as np

from loader import (read_patient_list, load_half_sequence,
                    load_sequence_spacing, parse_info_cfg)
from volume import lv_volume_biplane_ml, ejection_fraction, classify_ef


def compute_ef(pid):
    """Biplane EF from ground-truth masks, plus the reference EF."""
    gt_4ch = load_half_sequence(pid, view="4CH", gt=True)
    gt_2ch = load_half_sequence(pid, view="2CH", gt=True)
    sp_4ch = load_sequence_spacing(pid, view="4CH")
    sp_2ch = load_sequence_spacing(pid, view="2CH")

    edv = lv_volume_biplane_ml(gt_4ch[0], sp_4ch, gt_2ch[0], sp_2ch)
    esv = lv_volume_biplane_ml(gt_4ch[-1], sp_4ch, gt_2ch[-1], sp_2ch)

    ef = ejection_fraction(edv, esv)
    ef_ref = float(parse_info_cfg(pid, "4CH")["EF"])
    return ef, ef_ref


def collect(split_name):
    """Compute EF for every patient in a split."""
    ids = read_patient_list(split_name)
    computed, reference = [], []

    for i, pid in enumerate(ids):
        ef, ef_ref = compute_ef(pid)
        computed.append(ef)
        reference.append(ef_ref)
        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(ids)}")

    return np.array(computed), np.array(reference)


# --- Step 1: measure the offset on training data ---
print("Computing EF on training patients ...")
train_computed, train_reference = collect("training")
offset = (train_computed - train_reference).mean()
print(f"\nCalibration offset measured on training: {offset:+.2f} %")

# --- Step 2: apply it to validation data ---
print("\nComputing EF on validation patients ...")
val_computed, val_reference = collect("validation")
val_calibrated = val_computed - offset


def report(name, computed, reference):
    errors = computed - reference
    same_class = sum(classify_ef(c) == classify_ef(r)
                     for c, r in zip(computed, reference))

    print(f"\n--- {name} ---")
    print(f"Bias:                      {errors.mean():+.1f} %")
    print(f"Std of the difference:     {errors.std():.1f} %")
    print(f"Mean absolute error:       {np.abs(errors).mean():.1f} %")
    print(f"Same Normal/Reduced class: {same_class}/{len(computed)}")


print("\n=== Validation results ===")
report("Before calibration", val_computed, val_reference)
report("After calibration", val_calibrated, val_reference)