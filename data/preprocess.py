"""
preprocess.py

Loads, resizes, and caches CAMUS patient data to disk as .npz files
(one per patient per view), so training scripts don't need to re-read
and re-resize the raw NIfTI files every time they run.
"""

import os
import numpy as np

from loader import load_patient_frame, resize_image, read_patient_list

#OUTPUT_ROOT = "processed"  # will create processed/training/, processed/validation/, etc.
# __file__ is the path to preprocess.py itself.
# os.path.dirname(__file__) gives the folder that file is in (data/),
# regardless of which folder you ran the script from.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "processed")

def preprocess_one_patient(patient_id, view="4CH", phase="ED", target_size=(256, 256)):
    """
    Load, resize, and normalize one patient's frame and mask.

    Returns:
        img_resized (np.ndarray): normalized image, values in [0, 1]
        gt_resized (np.ndarray): mask, integer class labels (0-3)
    """
    img_arr, gt_arr = load_patient_frame(patient_id, view=view, phase=phase)

    img_resized = resize_image(img_arr, target_size=target_size, is_mask=False)
    gt_resized = resize_image(gt_arr, target_size=target_size, is_mask=True)

    # Normalize image pixel values from [0, 255] to [0, 1].
    # We do NOT normalize the mask -- its values are class labels, not intensities.
    img_normalized = img_resized.astype(np.float32) / 255.0

    return img_normalized, gt_resized

def preprocess_split(subgroup_name, view="4CH", phase="ED", target_size=(256, 256)):
    """
    Preprocess every patient in a split and save each as a .npz file.

    The view is part of the filename, so 2CH and 4CH can live side by side
    in the same folder and be trained on together.
    """
    patient_ids = read_patient_list(subgroup_name)
    out_dir = os.path.join(OUTPUT_ROOT, subgroup_name)
    os.makedirs(out_dir, exist_ok=True)

    for i, pid in enumerate(patient_ids):
        img, mask = preprocess_one_patient(pid, view=view, phase=phase, target_size=target_size)

        save_path = os.path.join(out_dir, f"{pid}_{view}.npz")
        np.savez(save_path, image=img, mask=mask)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(patient_ids)} patients ({view})...")

    print(f"Done. Saved {len(patient_ids)} patients ({view}) to {out_dir}")