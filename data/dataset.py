"""
dataset.py

PyTorch Dataset class that reads the preprocessed .npz files (created by
preprocess.py) and serves them in the format PyTorch's DataLoader expects.

Files are named patient0001_4CH.npz / patient0001_2CH.npz, so both views
live in the same folder and can be used together or filtered by view.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CamusDataset(Dataset):
    """
    Reads preprocessed .npz files from a split folder (e.g. processed/training/)
    and returns (image, mask) pairs as tensors.
    """

    def __init__(self, split_dir, view=None):
        """
        Args:
            split_dir (str): path to processed/training, processed/validation, etc.
            view (str or None): "2CH" or "4CH" to use only that view.
                                None uses both views together.
        """
        self.split_dir = split_dir
        self.view = view

        files = [f for f in os.listdir(split_dir) if f.endswith(".npz")]

        # Filter by view when asked. The underscore and dot make the match exact,
        # so "_4CH." cannot accidentally match some other part of the name.
        if view is not None:
            files = [f for f in files if f"_{view}." in f]

        self.file_names = sorted(files)

    def __len__(self):
        """Total number of samples in this dataset."""
        return len(self.file_names)

    def __getitem__(self, idx):
        """
        Load and return the sample at index idx.
        PyTorch calls this automatically, once per sample, when building a batch.
        """
        file_path = os.path.join(self.split_dir, self.file_names[idx])
        data = np.load(file_path)

        image = data["image"]  # shape (256, 256), values in [0, 1]
        mask = data["mask"]    # shape (256, 256), values 0-3

        # Add the channel dimension: (256, 256) -> (1, 256, 256),
        # since PyTorch expects (channels, H, W).
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return image_tensor, mask_tensor
