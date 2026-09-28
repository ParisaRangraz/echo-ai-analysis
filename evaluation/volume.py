"""
volume.py

Clinical measurements derived from segmentation masks: LV volume
(Simpson's method of disks) and ejection fraction.
"""

import numpy as np


def lv_volume_ml(mask, spacing, lv_class=1):
    """
    Estimate LV volume from a 2D mask using Simpson's method of disks.

    Each image row that contains LV pixels is treated as a thin disk.
    The row's width gives the disk diameter; the disk is assumed circular,
    which is the standard (approximate) assumption of the method.

    Args:
        mask: 2D array of class labels
        spacing: (x, y) pixel spacing in mm
        lv_class: which label is the LV cavity

    Returns:
        volume in millilitres (mL)
    """
    spacing_x, spacing_y = spacing

    lv = (mask == lv_class)

    # Number of LV pixels in each row -> width of the LV at that height.
    pixels_per_row = lv.sum(axis=1)

    # Convert pixel counts to real diameters in mm.
    diameters = pixels_per_row * spacing_x

    # Volume of each disk: circle area x thickness.
    # Thickness is one pixel height (spacing_y).
    disk_volumes = (np.pi / 4) * (diameters ** 2) * spacing_y

    volume_mm3 = disk_volumes.sum()

    # 1 mL = 1000 mm^3
    return volume_mm3 / 1000.0


def ejection_fraction(edv, esv):
    """
    Ejection fraction as a percentage.

    Args:
        edv: end-diastolic volume (largest, heart relaxed)
        esv: end-systolic volume (smallest, heart contracted)

    Returns:
        EF in percent
    """
    if edv <= 0:
        return float("nan")
    return (edv - esv) / edv * 100.0


def classify_ef(ef, threshold=50.0):
    """
    Standard clinical cut-off: EF below 50% is considered reduced.
    This is a fixed clinical rule, not a trained model.
    """
    return "Reduced" if ef < threshold else "Normal"


def lv_volume_biplane_ml(mask_4ch, spacing_4ch, mask_2ch, spacing_2ch, lv_class=1):
    """
    Estimate LV volume with Simpson's biplane method.

    Each disk is an ellipse: one diameter from the 4-chamber view, the other
    from the 2-chamber view. This is the standard clinical method, and the
    one the CAMUS reference EF values are based on.

    The two views have different image heights, so the LV is split into the
    same number of disks in both views and matched level by level.

    Returns:
        volume in millilitres (mL)
    """
    def diameters(mask, spacing, n_disks):
        """Row widths of the LV, in mm, resampled to n_disks levels."""
        lv = (mask == lv_class)
        rows = np.where(lv.any(axis=1))[0]   # rows that contain LV pixels
        if len(rows) == 0:
            return np.zeros(n_disks), 0.0

        widths = lv[rows[0]:rows[-1] + 1].sum(axis=1) * spacing[0]
        long_axis = (rows[-1] - rows[0] + 1) * spacing[1]   # LV height in mm

        # Resample onto a common number of levels, so the two views can be
        # matched disk by disk even though their pixel heights differ.
        positions = np.linspace(0, len(widths) - 1, n_disks)
        return np.interp(positions, np.arange(len(widths)), widths), long_axis

    n_disks = 20  # standard in clinical practice

    d_4ch, long_4ch = diameters(mask_4ch, spacing_4ch, n_disks)
    d_2ch, long_2ch = diameters(mask_2ch, spacing_2ch, n_disks)

    # Clinical convention: use the shorter of the two measured long axes.
    long_axis = min(long_4ch, long_2ch)
    thickness = long_axis / n_disks

    # Elliptical disks: pi/4 * d1 * d2 * thickness
    volume_mm3 = (np.pi / 4) * np.sum(d_4ch * d_2ch) * thickness

    return volume_mm3 / 1000.0