"""
train_pvt_boundary.py

Trains the same PVTv2 + attention-gated decoder as train_pvt.py, with one
change: the cross-entropy term is weighted to penalise errors near structure
boundaries more heavily.

Everything else -- data, architecture, learning rates, epochs, batch size --
is identical, so any difference in the results can be attributed to the loss.

Motivation: Dice and plain cross-entropy treat every pixel equally, so most of
the training signal comes from easy interior pixels. The myocardium is a thin
ring whose Dice depends almost entirely on boundary precision, and it is the
one structure where the transformer trailed the plain U-Net.

Note: the loss values printed here are NOT comparable with train_pvt.py's --
the formula is different. Only the validation Dice scores can be compared.
"""

import sys
sys.path.append("data")
sys.path.append("models")

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import CamusDataset
from pvt_unet import PVTUNet
from losses import BoundaryCombinedLoss

train_dataset = CamusDataset("data/processed/training")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

model = PVTUNet(num_classes=4, pretrained=True)

# boundary_weight=3.0 means pixels within the band around an edge count three
# times as much as interior pixels.
criterion = BoundaryCombinedLoss(num_classes=4, boundary_weight=3.0)

# Two learning rates: the pretrained encoder needs small updates so its
# ImageNet features are not washed out, while the decoder and attention gates
# start from random weights and need to learn much faster.
encoder_params = list(model.encoder.parameters())
encoder_ids = {id(p) for p in encoder_params}
decoder_params = [p for p in model.parameters() if id(p) not in encoder_ids]

optimizer = optim.Adam([
    {"params": encoder_params, "lr": 1e-4},
    {"params": decoder_params, "lr": 1e-3},
])

num_epochs = 15
CHECKPOINT_PATH = "models/pvt_boundary_checkpoint.pt"

start_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"]
    print(f"Resuming from epoch {start_epoch}")

for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0.0

    for images, masks in train_loader:
        masks = masks.squeeze(1).long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")

    # Save after every epoch, so an interrupted run is not lost.
    torch.save({
        "epoch": epoch + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, CHECKPOINT_PATH)

torch.save(model.state_dict(), "models/pvt_boundary_both_views.pt")
print("Model saved to models/pvt_boundary_both_views.pt")
