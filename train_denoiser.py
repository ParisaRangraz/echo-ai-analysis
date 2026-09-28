"""
train_denoiser.py

Trains the SimpleDenoiser model (self-supervised: the model learns to
reconstruct the input image through a compressed bottleneck, which
naturally suppresses speckle noise).
"""

import sys
sys.path.append("data")
sys.path.append("models")

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import CamusDataset
from denoiser import SimpleDenoiser

# --- Setup ---
train_dataset = CamusDataset("data/processed/training")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

model = SimpleDenoiser()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training loop ---
num_epochs = 5

for epoch in range(num_epochs):
    total_loss = 0.0

    for images, masks in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")

# --- Visualize the trained denoiser's output ---
model.eval()

sample_image, _ = train_dataset[0]
sample_batch = sample_image.unsqueeze(0)

with torch.no_grad():
    denoised = model(sample_batch)

original_np = sample_image.squeeze().numpy()
denoised_np = denoised.squeeze().numpy()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_np, cmap="gray")
plt.title("Original")
plt.subplot(1, 2, 2)
plt.imshow(denoised_np, cmap="gray")
plt.title("Denoised (trained model)")
plt.show()