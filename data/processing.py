"""
processing.py

Image processing operations applied to ultrasound frames: denoising
(Lee filter) and deformable registration / mask propagation. Distinct
from loader.py (raw I/O) and preprocess.py (the pipeline that combines
loading + processing + saving).
"""

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import uniform_filter


def lee_filter(img, window_size=5):
    """
    Apply the Lee filter for speckle noise reduction.

    The idea: compute local mean and variance in a sliding window. In
    uniform regions (low variance), smooth heavily. Near edges (high
    variance), smooth less -- this preserves anatomical boundaries
    while reducing speckle in flat areas.
    """
    img_mean = uniform_filter(img, size=window_size)
    img_sqr_mean = uniform_filter(img**2, size=window_size)
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = np.var(img)

    weight = img_variance / (img_variance + overall_variance)
    filtered = img_mean + weight * (img - img_mean)

    return filtered


def to_sitk(arr, spacing):
    """Convert a 2D numpy array into a SimpleITK image with real physical spacing."""
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing(spacing)
    return img


def register_bspline(fixed, moving, mesh_size=8, iterations=100, sampling_percentage=0.1):
    """
    Deformable (B-spline) registration of `moving` onto `fixed`.
    Returns a transform that can then be applied to anything that lives
    in the moving image's space -- e.g. its segmentation mask.

    sampling_percentage: fraction of pixels used to evaluate the similarity
    metric at each step. Using a random 10% instead of every pixel is standard
    practice and makes registration roughly 10x faster with little accuracy loss.
    """
    initial_tx = sitk.BSplineTransformInitializer(fixed, [mesh_size, mesh_size])

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMeanSquares()

    # Evaluate the metric on a random subset of pixels (fixed seed = reproducible).
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(sampling_percentage, 42)

    reg.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5,
                             numberOfIterations=iterations)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetInitialTransform(initial_tx, inPlace=True)

    # Multi-resolution: solve on a coarse image first, then refine.
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])

    return reg.Execute(fixed, moving)

def warp_mask(mask_img, transform, reference):
    """
    Apply a registration transform to a mask. Nearest-neighbor interpolation
    keeps class labels as whole numbers (0-3), exactly like during resizing.
    """
    return sitk.Resample(mask_img, reference, transform,
                         sitk.sitkNearestNeighbor, 0.0, mask_img.GetPixelID())

def propagate_mask(seq, first_mask, spacing):
    """
    Carry the first frame's mask through the whole sequence with
    frame-to-frame registration (a chain: frame 0 -> 1 -> 2 -> ...).

    Args:
        seq: image frames, shape (frames, H, W)
        first_mask: mask of frame 0 (ED), shape (H, W)
        spacing: physical pixel spacing (x, y) in mm

    Returns:
        list of propagated masks (numpy arrays), one per frame
    """
    current = to_sitk(first_mask, spacing)
    masks = [first_mask]

    for k in range(1, seq.shape[0]):
        fixed = to_sitk(seq[k], spacing)        # current frame (reference)
        moving = to_sitk(seq[k - 1], spacing)   # previous frame (deformed)
        transform = register_bspline(fixed, moving)
        current = warp_mask(current, transform, fixed)
        masks.append(sitk.GetArrayFromImage(current))

    return masks