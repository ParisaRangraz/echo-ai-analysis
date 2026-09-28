"""
evaluate_ef.py

Computes ejection fraction for every validation patient and compares it with
the CAMUS reference EF.

Ground-truth masks are used here on purpose: this isolates the volume/EF
calculation from segmentation and registration error. If the calculation agrees
well with the reference here, any later disagreement can be attributed to the
masks rather than to the geometry.

Both estimates are reported:
  mono-plane  -- 4CH only, disks assumed circular
  biplane     -- 4CH + 2CH, disks assumed elliptical (the clinical standard)

Agreement is assessed the way method-comparison studies do it (Bland & Altman,
1986): not by asking whether two methods give identical numbers, but by
measuring the systematic offset (bias) and the spread around it.
"""

import sys
sys.path.append("data")
sys.path.append("evaluation")

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from loader import (read_patient_list, load_half_sequence,
                    load_sequence_spacing, parse_info_cfg)
from volume import lv_volume_ml, lv_volume_biplane_ml, ejection_fraction, classify_ef

patient_ids = read_patient_list("validation")
os.makedirs("results", exist_ok=True)
rows = []

for pid in patient_ids:
    # Ground-truth masks for both views; only ED (first) and ES (last) frames
    # are needed for EF.
    gt_4ch = load_half_sequence(pid, view="4CH", gt=True)
    gt_2ch = load_half_sequence(pid, view="2CH", gt=True)
    sp_4ch = load_sequence_spacing(pid, view="4CH")
    sp_2ch = load_sequence_spacing(pid, view="2CH")

    # Mono-plane: 4CH width used as both disk diameters.
    edv_mono = lv_volume_ml(gt_4ch[0], sp_4ch)
    esv_mono = lv_volume_ml(gt_4ch[-1], sp_4ch)
    ef_mono = ejection_fraction(edv_mono, esv_mono)

    # Biplane: one diameter from each view.
    edv_bi = lv_volume_biplane_ml(gt_4ch[0], sp_4ch, gt_2ch[0], sp_2ch)
    esv_bi = lv_volume_biplane_ml(gt_4ch[-1], sp_4ch, gt_2ch[-1], sp_2ch)
    ef_bi = ejection_fraction(edv_bi, esv_bi)

    info = parse_info_cfg(pid, "4CH")
    ef_ref = float(info["EF"])

    rows.append({
        "patient": pid,
        "image_quality": info.get("ImageQuality", ""),
        "edv_mono": round(edv_mono, 1),
        "esv_mono": round(esv_mono, 1),
        "ef_mono": round(ef_mono, 1),
        "edv_biplane": round(edv_bi, 1),
        "esv_biplane": round(esv_bi, 1),
        "ef_biplane": round(ef_bi, 1),
        "ef_reference": ef_ref,
        "error_mono": round(ef_mono - ef_ref, 1),
        "error_biplane": round(ef_bi - ef_ref, 1),
    })

    print(f"{pid}: mono {ef_mono:5.1f} | biplane {ef_bi:5.1f} | reference {ef_ref:5.1f}")

csv_path = os.path.join("results", "ef_validation.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"\nSaved {len(rows)} rows to {csv_path}")


def report(name, computed_key, error_key):
    """Print agreement statistics for one method."""
    computed = np.array([r[computed_key] for r in rows])
    reference = np.array([r["ef_reference"] for r in rows])
    errors = np.array([r[error_key] for r in rows])

    bias = errors.mean()            # systematic offset
    sd = errors.std()               # spread around that offset
    mae = np.abs(errors).mean()     # typical size of the error
    corr = np.corrcoef(computed, reference)[0, 1]  # do they move together?

    # Agreement between the Normal/Reduced labels, which is what a clinician acts on.
    same_class = sum(classify_ef(c) == classify_ef(r) for c, r in zip(computed, reference))

    print(f"\n--- {name} ---")
    print(f"Bias (computed - reference): {bias:+.1f} %")
    print(f"Std of the difference:       {sd:.1f} %")
    print(f"Mean absolute error:         {mae:.1f} %")
    print(f"Correlation with reference:  {corr:.3f}")
    print(f"Same Normal/Reduced class:   {same_class}/{len(rows)}")
    return computed, reference, errors


mono = report("Mono-plane (4CH only)", "ef_mono", "error_mono")
biplane = report("Biplane (4CH + 2CH)", "ef_biplane", "error_biplane")


# --- Bland-Altman plots ---
# x-axis: the average of the two methods (best estimate of the true value)
# y-axis: their difference. Flat scatter around a horizontal line means a
# constant offset; a slope would mean the error depends on the EF value.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (computed, reference, errors), title in zip(
        axes, [mono, biplane], ["Mono-plane", "Biplane"]):

    mean_values = (computed + reference) / 2
    bias = errors.mean()
    sd = errors.std()

    ax.scatter(mean_values, errors, alpha=0.7)
    ax.axhline(bias, color="red", label=f"bias {bias:+.1f}")
    # Limits of agreement: the range containing ~95% of differences.
    ax.axhline(bias + 1.96 * sd, color="gray", linestyle="--", label="+/- 1.96 SD")
    ax.axhline(bias - 1.96 * sd, color="gray", linestyle="--")
    ax.set_xlabel("Mean of computed and reference EF (%)")
    ax.set_ylabel("Computed - reference EF (%)")
    ax.set_title(f"Bland-Altman: {title}")
    ax.legend()

plt.tight_layout()
plt.show()
