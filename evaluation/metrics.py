"""
metrics.py

Evaluation metrics shared across the project, so the same Dice definition
is used everywhere (segmentation, registration, and later EF evaluation)
instead of being copy-pasted into each script.
"""

import numpy as np


def dice_score(pred, target, class_id):
    """
    Dice for one class: 2 * overlap / (predicted area + true area).

    Args:
        pred: predicted mask (numpy array of class labels)
        target: ground-truth mask (same shape)
        class_id: which class to score (e.g. 1 = LV cavity)

    Returns:
        float between 0 (no overlap) and 1 (perfect overlap)
    """
    p = (pred == class_id)
    t = (target == class_id)
    total = p.sum() + t.sum()
    if total == 0:
        return 1.0  # class absent in both: perfect agreement
    return 2 * np.logical_and(p, t).sum() / total
