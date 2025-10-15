import os
import torch

DIR = "weights"  # change to your folder path

for fname in os.listdir(DIR):
   
    if not fname.endswith(".pt"):
        continue

    path = os.path.join(DIR, fname)
    try:
        ckpt = torch.load(path, map_location="cpu")
        print(f"epoch: {ckpt['epoch']}; train loss: {ckpt['training_loss']}; val loss: {ckpt['validation_loss']}") 
    except Exception as e:
        print(f"\nError reading {fname}: {e}")

