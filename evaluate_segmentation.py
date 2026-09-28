"""
evaluate_segmentation.py

# Loads a trained U-Net and evaluates it on the validation set -- patients the
# model never saw during training. Reports the Dice score per heart structure.

# Results are reported separately for each view and then combined. Separating
# them matters: the model trained on both views must be compared with the earlier
# 4CH-only model on 4CH data alone, otherwise the model and the evaluation set
# would both have changed at once and the comparison would mean nothing.
a trained segmentation model
"""
import sys
sys.path.append("data")
sys.path.append("models")
sys.path.append("evaluation")

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import CamusDataset
#from unet import UNet
#from pvt_unet import PVTUNet
#from pvt_rta import PVTReverseAttentionUNet
from pvt_rta_transformer import PVTRTAUNet
from metrics import dice_score

#MODEL_PATH = "models/unet_ed_both_views.pt"

# Background (class 0) is excluded on purpose: it covers most of the image,
# so including it would make the average look better than it is.
CLASS_NAMES = {1: "LV cavity", 2: "Myocardium", 3: "Left atrium"}

# --- Load the trained model (no retraining needed) ---
#model = UNet(num_classes=4)
#MODEL_PATH = "models/pvt_ed_both_views.pt"
#MODEL_PATH = "models/pvt_boundary_both_views.pt"
#model = PVTUNet(num_classes=4, pretrained=False)
#MODEL_PATH = "models/pvt_rta_both_views.pt"
#model = PVTReverseAttentionUNet(num_classes=4, pretrained=False)
MODEL_PATH = "models/pvt_rta_transformer_both_views.pt"
model = PVTRTAUNet(num_classes=4, pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()
print(f"Loaded {MODEL_PATH}\n")


def evaluate(view=None):
    """Run the model over one view (or both) and return Dice scores per class."""
    dataset = CamusDataset("data/processed/validation", view=view)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    scores = {c: [] for c in CLASS_NAMES}

    with torch.no_grad():
        for images, masks in loader:
            preds = model(images).argmax(dim=1).numpy()
            targets = masks.squeeze(1).long().numpy()

            for p, t in zip(preds, targets):
                for c in CLASS_NAMES:
                    scores[c].append(dice_score(p, t, c))

    label = view if view else "Both views"
    print(f"--- {label} ({len(dataset)} samples) ---")
    for c, name in CLASS_NAMES.items():
        arr = np.array(scores[c])
        print(f"{name:12s} Dice: {arr.mean():.3f} +/- {arr.std():.3f}")
    print()

    return scores


# 4CH first, so it can be compared directly with the earlier 4CH-only model.
evaluate("4CH")
evaluate("2CH")
evaluate(None)
