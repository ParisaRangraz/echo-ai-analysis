"""
run_preprocessing.py

Preprocess both views for the training and validation splits.
Run once; the cached .npz files are then used by all training scripts.
"""

from preprocess import preprocess_split

for split in ("training", "validation"):
    for view in ("2CH", "4CH"):
        preprocess_split(split, view=view)