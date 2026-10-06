"""
loader.py

Reusable functions for loading and preprocessing CAMUS patient data.
Import these into other scripts (exploration, preprocessing, training)
instead of copy-pasting them.
"""

import os

import SimpleITK as sitk
import numpy as np
from skimage.transform import resize as sk_resize


# Where the CAMUS dataset lives. Set the ECHO_DATA_ROOT environment variable to
# the folder that contains database_nifti/ and database_split/. If it is not
# set, a path relative to the repository root is used, which matches the layout
# in data/README.md. No absolute path is hard-coded here, so the repository
# carries no machine-specific path and runs on any machine once the data is in
# place or the variable is set.
_DEFAULT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "CAMUS_public", "CAMUS_public",
)
_DATA_HOME = os.environ.get("ECHO_DATA_ROOT", _DEFAULT_ROOT)

DATA_ROOT = os.path.join(_DATA_HOME, "database_nifti")
SPLIT_ROOT = os.path.join(_DATA_HOME, "database_split")


def _require(path):
    """Return `path`, or raise a clear error naming ECHO_DATA_ROOT if it is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CAMUS file not found:\n  {path}\n"
            f"Looked under data root: {_DATA_HOME}\n"
            "Set the ECHO_DATA_ROOT environment variable to the folder that "
            "contains database_nifti/ and database_split/, or place the data "
            "there (see data/README.md)."
        )
    return path


def load_patient_frame(patient_id, view="4CH", phase="ED"):
    """Load a single echo frame and its ground-truth mask for one patient."""
    img_path = _require(os.path.join(DATA_ROOT, patient_id, f"{patient_id}_{view}_{phase}.nii.gz"))
    gt_path = _require(os.path.join(DATA_ROOT, patient_id, f"{patient_id}_{view}_{phase}_gt.nii.gz"))

    img = sitk.ReadImage(img_path)
    gt_img = sitk.ReadImage(gt_path)

    img_arr = sitk.GetArrayFromImage(img)
    gt_arr = sitk.GetArrayFromImage(gt_img)

    return img_arr, gt_arr


def resize_image(arr, target_size=(256, 256), is_mask=False):
    """Resize a 2D numpy array. Nearest-neighbor for masks, linear for images."""
    order = 0 if is_mask else 1
    resized = sk_resize(arr, target_size, order=order, preserve_range=True, anti_aliasing=not is_mask)
    return resized.astype(arr.dtype)


def read_patient_list(subgroup_name):
    """
    Read a subgroup file by name (e.g. 'training', 'validation', 'testing')
    from the official CAMUS split, and return a list of patient IDs.
    """
    txt_path = _require(os.path.join(SPLIT_ROOT, f"subgroup_{subgroup_name}.txt"))
    with open(txt_path, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids

# #frames
# def load_half_sequence(patient_id, view="4CH"):
#     """
#     Load the ED-to-ES sequence of frames (multiple frames, not just one).

#     Args:
#         patient_id (str): e.g. "patient0001"
#         view (str): "2CH" or "4CH"

#     Returns:
#         np.ndarray of shape (num_frames, height, width)
#     """
#     path = _require(os.path.join(DATA_ROOT, patient_id, f"{patient_id}_{view}_half_sequence.nii.gz"))
#     img = sitk.ReadImage(path)
#     array = sitk.GetArrayFromImage(img)
#     return array
def load_half_sequence(patient_id, view="4CH", gt=False):
    """
    Load the ED-to-ES sequence of frames.

    Args:
        patient_id (str): e.g. "patient0001"
        view (str): "2CH" or "4CH"
        gt (bool): if True, load the ground-truth mask sequence instead of the images

    Returns:
        np.ndarray of shape (num_frames, height, width)
    """
    suffix = "_gt" if gt else ""
    path = _require(os.path.join(DATA_ROOT, patient_id, f"{patient_id}_{view}_half_sequence{suffix}.nii.gz"))
    img = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(img)
    return array
def load_sequence_spacing(patient_id, view="4CH"):
    """
    Return the physical pixel spacing (x, y) of the half_sequence,
    needed for registration. The third spacing value is time, so it is dropped.
    """
    path = _require(os.path.join(DATA_ROOT, patient_id, f"{patient_id}_{view}_half_sequence.nii.gz"))
    img = sitk.ReadImage(path)
    return img.GetSpacing()[:2]

def parse_info_cfg(patient_id, view="4CH"):
    """
    Read the Info_2CH.cfg / Info_4CH.cfg file for one patient.
    Contains reference values such as ED/ES frame numbers, LV volumes,
    ejection fraction, image quality, age and sex.

    Returns a dict; numeric values are converted to int or float.
    """
    path = _require(os.path.join(DATA_ROOT, patient_id, f"Info_{view}.cfg"))
    info = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key, value = key.strip(), value.strip()
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            info[key] = value

    return info

def get_ed_es_indices(patient_id, view="4CH"):
    """
    Return the (ED, ES) frame indices for one sequence.

    half_sequence does not always run ED -> ES: in about 2% of sequences it
    runs the other way, and it can differ between the two views of the same
    patient. Info_*.cfg states the true frame numbers, so they are read rather
    than assumed.

    cfg numbers frames from 1; numpy indexes from 0.
    """
    info = parse_info_cfg(patient_id, view)
    return info["ED"] - 1, info["ES"] - 1


def load_sequence_ed_first(patient_id, view="4CH", gt=False):
    """
    Load a half_sequence, reversed if necessary so that index 0 is always ED
    and the last index is always ES.

    This lets every downstream step assume that order, instead of each script
    having to handle both cases.
    """
    seq = load_half_sequence(patient_id, view=view, gt=gt)
    ed_index, es_index = get_ed_es_indices(patient_id, view)

    if ed_index > es_index:
        seq = seq[::-1]

    return seq
