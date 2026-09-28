import sys
sys.path.append("../models")

import torch
from pvt_unet import PVTUNet

model = PVTUNet(num_classes=4)

fake_image = torch.rand(1, 1, 256, 256)
output = model(fake_image)

print("Input shape:", fake_image.shape)
print("Output shape:", output.shape)

# How much bigger is this than the plain U-Net?
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params/1e6:.1f} M")