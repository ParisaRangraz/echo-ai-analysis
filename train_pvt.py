"""
train_pvt.py

Trains the PVTv2 + attention-gated U-Net on the same data and with the same
loss as the plain U-Net, so the two can be compared fairly. The only things
that change are the architecture and the learning rate.

A lower learning rate is used because the encoder is pretrained: large updates
would destroy the features it already learned on ImageNet.
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
from losses import CombinedLoss

train_dataset = CamusDataset("data/processed/training")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

model = PVTUNet(num_classes=4, pretrained=True)
criterion = CombinedLoss(num_classes=4)

# 1e-4 rather than 1e-3: the encoder starts from pretrained weights, and a
# large learning rate would wash them out in the first few batches.
#optimizer = optim.Adam(model.parameters(), lr=1e-4)
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
CHECKPOINT_PATH = "models/pvt_checkpoint.pt"

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

    torch.save({
        "epoch": epoch + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, CHECKPOINT_PATH)

torch.save(model.state_dict(), "models/pvt_ed_both_views.pt")
print("Model saved to models/pvt_ed_both_views.pt")