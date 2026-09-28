"""
train_pvt_rta.py

Trains the PVTv2 + reverse-attention decoder (pvt_rta.py).

Everything except the decoder's skip module is identical to train_pvt.py --
same data, same loss, same learning rates, same epochs -- so any difference in
the validation Dice is attributable to reverse attention alone.

The loss here IS comparable with train_pvt.py's, unlike the boundary-loss
experiment: the loss function is unchanged.
"""

import sys
sys.path.append("data")
sys.path.append("models")

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import CamusDataset
from pvt_rta import PVTReverseAttentionUNet
from losses import CombinedLoss

train_dataset = CamusDataset("data/processed/training")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

model = PVTReverseAttentionUNet(num_classes=4, pretrained=True)
criterion = CombinedLoss(num_classes=4)

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params/1e6:.1f} M")

# Two learning rates: the pretrained encoder needs small updates so its
# ImageNet features are not washed out, while the decoder starts from random
# weights and needs to learn much faster.
encoder_params = list(model.encoder.parameters())
encoder_ids = {id(p) for p in encoder_params}
decoder_params = [p for p in model.parameters() if id(p) not in encoder_ids]

optimizer = optim.Adam([
    {"params": encoder_params, "lr": 1e-4},
    {"params": decoder_params, "lr": 1e-3},
])

num_epochs = 15
CHECKPOINT_PATH = "models/pvt_rta_checkpoint.pt"

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

torch.save(model.state_dict(), "models/pvt_rta_both_views.pt")
print("Model saved to models/pvt_rta_both_views.pt")
