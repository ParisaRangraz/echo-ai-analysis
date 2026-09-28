"""
test_core.py

Unit tests for the functions the pipeline depends on most.

These test properties that must hold, not exact outputs: that resizing a mask
never invents a class label, that Dice behaves correctly at its extremes, that
volume scales the way geometry says it should. A function can run without
error and still be wrong; these catch that.

Run from the repository root:
    pytest
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))

import numpy as np

from loader import resize_image
from metrics import dice_score
from volume import ejection_fraction, classify_ef, lv_volume_ml


# --- resize_image ---

def test_resize_changes_shape():
    image = np.random.rand(100, 200).astype(np.float32)
    assert resize_image(image, (256, 256)).shape == (256, 256)


def test_resize_mask_keeps_only_original_labels():
    """
    The critical property: resizing a mask must never produce a label that
    was not there before. Linear interpolation would create values like 1.5,
    which correspond to no anatomical structure.
    """
    mask = np.zeros((100, 100), dtype=np.float32)
    mask[20:50, 20:50] = 1
    mask[50:80, 20:50] = 3

    resized = resize_image(mask, (256, 256), is_mask=True)

    assert set(np.unique(resized)).issubset({0.0, 1.0, 3.0})


def test_resize_image_uses_smooth_interpolation():
    """An intensity image, unlike a mask, is allowed to gain intermediate values."""
    image = np.zeros((100, 100), dtype=np.float32)
    image[:50] = 1.0

    resized = resize_image(image, (256, 256), is_mask=False)

    # Somewhere along the edge there should be a value strictly between 0 and 1.
    assert ((resized > 0.01) & (resized < 0.99)).any()


# --- dice_score ---

def test_dice_identical_masks_is_one():
    mask = np.zeros((10, 10))
    mask[2:5, 2:5] = 1
    assert dice_score(mask, mask, 1) == 1.0


def test_dice_no_overlap_is_zero():
    a = np.zeros((10, 10))
    a[0:3, 0:3] = 1

    b = np.zeros((10, 10))
    b[7:10, 7:10] = 1

    assert dice_score(a, b, 1) == 0.0


def test_dice_half_overlap():
    """
    Two 4x4 squares overlapping in a 2x4 strip: intersection 8, areas 16 + 16.
    Dice = 2*8 / 32 = 0.5
    """
    a = np.zeros((10, 10))
    a[0:4, 0:4] = 1

    b = np.zeros((10, 10))
    b[2:6, 0:4] = 1

    assert abs(dice_score(a, b, 1) - 0.5) < 1e-9


def test_dice_class_absent_from_both_is_one():
    """Perfect agreement: both agree the class is not present."""
    empty = np.zeros((10, 10))
    assert dice_score(empty, empty, 2) == 1.0


def test_dice_only_scores_the_requested_class():
    """A difference in another class must not affect the score."""
    a = np.zeros((10, 10))
    a[0:4, 0:4] = 1

    b = a.copy()
    b[6:9, 6:9] = 3  # extra region, but a different class

    assert dice_score(a, b, 1) == 1.0


# --- ejection fraction ---

def test_ejection_fraction_basic():
    assert abs(ejection_fraction(100.0, 40.0) - 60.0) < 1e-9


def test_ejection_fraction_no_contraction_is_zero():
    assert ejection_fraction(100.0, 100.0) == 0.0


def test_ejection_fraction_handles_zero_volume():
    """Must not raise ZeroDivisionError on an empty mask."""
    assert np.isnan(ejection_fraction(0.0, 0.0))


def test_classify_ef_threshold():
    assert classify_ef(60.0) == "Normal"
    assert classify_ef(40.0) == "Reduced"
    assert classify_ef(50.0) == "Normal"   # the cut-off itself counts as normal
    assert classify_ef(49.9) == "Reduced"


# --- volume ---

def test_volume_scales_with_spacing():
    """
    Volume is proportional to diameter squared times thickness. Doubling the
    pixel size in every direction should therefore multiply volume by 8.
    """
    mask = np.zeros((50, 50))
    mask[10:40, 15:35] = 1

    small = lv_volume_ml(mask, (0.5, 0.5))
    large = lv_volume_ml(mask, (1.0, 1.0))

    assert abs(large / small - 8.0) < 1e-6


def test_volume_of_empty_mask_is_zero():
    assert lv_volume_ml(np.zeros((50, 50)), (0.3, 0.3)) == 0.0


def test_larger_region_gives_larger_volume():
    small = np.zeros((50, 50))
    small[20:30, 20:30] = 1

    large = np.zeros((50, 50))
    large[15:35, 15:35] = 1

    assert lv_volume_ml(large, (0.3, 0.3)) > lv_volume_ml(small, (0.3, 0.3))
