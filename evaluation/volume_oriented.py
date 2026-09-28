"""
volume_oriented.py

Simpson's method with the disks oriented along the LV's own long axis, rather
than along the image rows.

The standard implementation in volume.py treats each image row as a disk. That
is only correct if the LV long axis happens to be vertical in the image. The
clinical definition places the axis between the apex and the midpoint of the
mitral annulus, and stacks the disks perpendicular to it.

Ferraz et al. (2022) measured the cost of this approximation directly and found
that re-orienting the disks improves both accuracy and precision, which makes it
the most likely remaining source of the EF bias measured in this project.

Implementation: find the axis, rotate the mask so the axis becomes vertical,
then reuse the existing row-based calculation. Rotating the mask is equivalent
to tilting the disks, and far simpler than resampling along an arbitrary
direction.
"""

import numpy as np
from scipy.ndimage import rotate

from volume import lv_volume_ml, lv_volume_biplane_ml
from scipy.ndimage import binary_dilation


# def find_long_axis(mask, lv_class=1):
#     """
#     Locate the LV long axis: apex to the midpoint of the mitral annulus.

#     The apex is the topmost point of the cavity. The annulus is the open end at
#     the base, where the CAMUS masks cut the ventricle off with a straight line,
#     so its midpoint is the centre of the lowest row containing LV pixels.

#     Returns:
#         (apex_row, apex_col), (base_row, base_col), angle_degrees
#         angle_degrees is how far the axis is tilted from vertical; positive
#         means the base sits to the right of the apex.
#     """
#     lv = (mask == lv_class)
#     rows = np.where(lv.any(axis=1))[0]

#     if len(rows) < 2:
#         return None, None, 0.0

#     top_row, bottom_row = rows[0], rows[-1]

#     # Centre of the LV pixels in the topmost and bottommost rows.
#     apex_col = np.where(lv[top_row])[0].mean()
#     base_col = np.where(lv[bottom_row])[0].mean()

#     # Angle from vertical. atan2 takes the horizontal offset over the vertical
#     # one, so a perfectly vertical axis gives 0.
#     d_col = base_col - apex_col
#     d_row = bottom_row - top_row
#     angle = np.degrees(np.arctan2(d_col, d_row))

#     return (top_row, apex_col), (bottom_row, base_col), angle
# def find_long_axis(mask, lv_class=1, atrium_class=3):
#     """
#     Locate the LV long axis: apex to the midpoint of the mitral annulus.

#     The annulus is the boundary between the LV cavity and the left atrium --
#     the straight cut at the base of the ventricle. Taking the lowest row of the
#     LV instead finds only one corner of that cut when the ventricle is tilted,
#     which overestimates the tilt badly.

#     The apex is the point of the cavity furthest from the annulus midpoint,
#     rather than simply the topmost pixel, for the same reason.

#     Returns:
#         (apex_row, apex_col), (base_row, base_col), angle_degrees
#     """
#     lv = (mask == lv_class)
#     atrium = (mask == atrium_class)

#     if not lv.any():
#         return None, None, 0.0

#     # Dilate the atrium by one pixel and intersect with the LV: the result is
#     # the strip of LV pixels lying directly against the atrium, i.e. the annulus.
    
#     annulus = lv & binary_dilation(atrium)

#     if not annulus.any():
#         # No atrium in this mask -- fall back to the lowest row of the LV.
#         rows = np.where(lv.any(axis=1))[0]
#         base = (rows[-1], np.where(lv[rows[-1]])[0].mean())
#     else:
#         coords = np.argwhere(annulus)
#         base = (coords[:, 0].mean(), coords[:, 1].mean())

#     # Apex: the LV pixel furthest from the annulus midpoint.
#     lv_coords = np.argwhere(lv)
#     distances = np.hypot(lv_coords[:, 0] - base[0], lv_coords[:, 1] - base[1])
#     apex = tuple(lv_coords[distances.argmax()])

#     d_row = base[0] - apex[0]
#     d_col = base[1] - apex[1]
#     angle = np.degrees(np.arctan2(d_col, d_row))

#     return apex, base, angle
def find_long_axis(mask, lv_class=1, atrium_class=3):
    """
    Locate the LV long axis by fitting a line through the cavity's centres.

    Two earlier attempts used extreme points -- the lowest row, then the
    furthest pixel from the base -- and both failed for the same reason: an
    extreme point of a wide, rounded shape lands on a corner, not on the axis.

    Fitting through the row centres uses the whole shape instead, so no single
    pixel can tilt the result.
    """
    lv = (mask == lv_class)
    if not lv.any():
        return None, None, 0.0

    rows = np.where(lv.any(axis=1))[0]

    # Horizontal centre of the cavity in every row it occupies.
    centres = np.array([np.where(lv[r])[0].mean() for r in rows])

    # Least-squares line through those centres: col = slope * row + intercept.
    slope, intercept = np.polyfit(rows, centres, 1)

    apex = (rows[0], slope * rows[0] + intercept)
    base = (rows[-1], slope * rows[-1] + intercept)

    # The slope is d_col / d_row, which is exactly the tangent of the tilt.
    angle = np.degrees(np.arctan(slope))

    return apex, base, angle


def rotate_mask(mask, angle):
    """
    Rotate a mask so its long axis becomes vertical.

    order=0 is nearest-neighbor, which keeps the class labels as whole numbers.
    reshape=True lets the output grow, so no part of the mask is cut off.
    """
    if abs(angle) < 0.1:
        return mask   # already vertical; skip the resampling and its error

    return rotate(mask, angle, order=0, reshape=True, mode="constant", cval=0)


def lv_volume_oriented_ml(mask, spacing, lv_class=1):
    """Mono-plane volume with the disks perpendicular to the LV long axis."""
    _, _, angle = find_long_axis(mask, lv_class)
    return lv_volume_ml(rotate_mask(mask, angle), spacing, lv_class)


def lv_volume_biplane_oriented_ml(mask_4ch, spacing_4ch, mask_2ch, spacing_2ch,
                                  lv_class=1):
    """
    Biplane volume with each view's mask rotated to its own long axis first.

    Each view is rotated independently: the LV can be tilted by different
    amounts in the two planes, since they are acquired at different probe
    rotations.
    """
    _, _, angle_4ch = find_long_axis(mask_4ch, lv_class)
    _, _, angle_2ch = find_long_axis(mask_2ch, lv_class)

    return lv_volume_biplane_ml(
        rotate_mask(mask_4ch, angle_4ch), spacing_4ch,
        rotate_mask(mask_2ch, angle_2ch), spacing_2ch,
        lv_class,
    )
