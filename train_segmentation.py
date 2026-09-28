"""
train_segmentation.py

Trains a U-Net to segment the LV cavity, myocardium, and left atrium
on ED frames from both apical views (2CH and 4CH).
Uses CombinedLoss (CrossEntropy + Dice) to handle class imbalance.
Saves a checkpoint after every epoch so an interrupted run can resume.
"""

import sys
sys.path.append("data")
sys.path.append("models")

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from dataset import CamusDataset
from unet import UNet
from losses import CombinedLoss

# --- Setup ---
train_dataset = CamusDataset("data/processed/training")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

model = UNet(num_classes=4)
criterion = CombinedLoss(num_classes=4)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- Training loop ---
num_epochs = 15
num_epochs = 15
CHECKPOINT_PATH = "models/unet_both_views_checkpoint.pt"

# If a checkpoint exists, continue from it instead of starting over.
start_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"]
    print(f"Resuming from epoch {start_epoch}")

#for epoch in range(num_epochs):
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
    # start_epoch is stored too, so training can resume from where it stopped.
    torch.save({
        "epoch": epoch + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, CHECKPOINT_PATH)
# --- Save the trained model ---
#torch.save(model.state_dict(), "models/unet_ed.pt")
#print("Model saved to models/unet_ed.pt")
torch.save(model.state_dict(), "models/unet_ed_both_views.pt")
print("Model saved to models/unet_ed_both_views.pt")

# --- Visualize a prediction ---
model.eval()

sample_image, sample_mask = train_dataset[0]
sample_batch = sample_image.unsqueeze(0)

with torch.no_grad():
    output = model(sample_batch)
    predicted_mask = output.argmax(dim=1)

print("Predicted classes:", torch.unique(predicted_mask))

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(sample_image.squeeze().numpy(), cmap="gray")
plt.title("Input Image")

plt.subplot(1, 3, 2)
plt.imshow(sample_mask.squeeze().numpy(), cmap="viridis", vmin=0, vmax=3)
plt.title("Ground Truth")

plt.subplot(1, 3, 3)
plt.imshow(predicted_mask.squeeze().numpy(), cmap="viridis", vmin=0, vmax=3)
plt.title("Model Prediction")

plt.show()