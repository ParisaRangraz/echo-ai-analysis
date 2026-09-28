"""
train_pvt_rta_transformer.py

Trains the transformer variant of reverse attention (pvt_rta_transformer.py).

Together with train_pvt.py (attention gates) and train_pvt_rta.py
(convolutional reverse attention), this reproduces the RTA-Former ablation on
a different dataset and task: does reverse attention help, and does the
transformer version help more than the convolutional one?

Training settings are identical to the other two runs.
"""

import sys
sys.path.append("data")
sys.path.append("models")

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import CamusDataset
from pvt_rta_transformer import PVTRTAUNet
from losses import CombinedLoss

train_dataset = CamusDataset("data/processed/training")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

model = PVTRTAUNet(num_classes=4, pretrained=True)
criterion = CombinedLoss(num_classes=4)

n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params/1e6:.1f} M")

encoder_params = list(model.encoder.parameters())
encoder_ids = {id(p) for p in encoder_params}
decoder_params = [p for p in model.parameters() if id(p) not in encoder_ids]

optimizer = optim.Adam([
    {"params": encoder_params, "lr": 1e-4},
    {"params": decoder_params, "lr": 1e-3},
])

num_epochs = 15
CHECKPOINT_PATH = "models/pvt_rta_transformer_checkpoint.pt"

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

torch.save(model.state_dict(), "models/pvt_rta_transformer_both_views.pt")
print("Model saved to models/pvt_rta_transformer_both_views.pt")
