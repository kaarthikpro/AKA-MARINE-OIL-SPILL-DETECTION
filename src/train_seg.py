# src/train_seg.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import inspect

# Import relative modules
from src.dataset import SegmentationDataset, NORMALIZE_TRANSFORM
from src.models import UNet

# --- Path Utility (Robust Fix) ---
# Get the directory where train_seg.py lives
CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# Go up one level to the project root
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) 

def get_full_data_path(relative_path):
    """Constructs an absolute path from the project root."""
    return os.path.join(PROJECT_ROOT, relative_path)

# --- Metrics Utility ---
def iou_score(pred, mask, eps=1e-6):
    """Calculates Intersection over Union (IoU) on a batch."""
    pred = (pred > 0.5).float()
    inter = (pred * mask).sum()
    union = pred.sum() + mask.sum() - inter
    return (inter + eps) / (union + eps)

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Use robust path utility to locate CSVs
train_csv = get_full_data_path("data/splits/train_seg.csv")
val_csv   = get_full_data_path("data/splits/val_seg.csv")

BATCH_SIZE = 4
EPOCHS = 5 # Set to 5 for quick run
LR = 1e-4

# --- Data Loaders ---
# Note: SegmentationDataset handles reading the CSV using the full path provided
train_ds = SegmentationDataset(train_csv, transform=NORMALIZE_TRANSFORM)
val_ds   = SegmentationDataset(val_csv, transform=NORMALIZE_TRANSFORM)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# --- Model, Loss, Optimizer ---
model = UNet().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_iou = 0.0
train_losses, val_losses = [], []

# --- TRAIN LOOP ---
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    val_iou, val_loss = 0, 0
    preds_all, masks_all = [], []
    
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, masks)
            val_loss += loss.item()
            
            # Continuous predictions
            preds_prob = torch.sigmoid(logits)
            val_iou += iou_score(preds_prob, masks).item()
            
            # FIX: Convert both predictions and masks to discrete binary (0 or 1)
            # This prevents the ValueError in sklearn metrics
            preds_binary = (preds_prob > 0.5).float().cpu().numpy().flatten()
            masks_binary = (masks > 0.5).float().cpu().numpy().flatten()
            
            preds_all.extend(preds_binary)
            masks_all.extend(masks_binary)

    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    avg_val_iou = val_iou / len(val_loader)
    
    # Metrics
    # Check if there are any non-zero pixels before calculating metrics
    if np.sum(masks_all) > 0 and np.sum(preds_all) > 0:
        acc = accuracy_score(masks_all, preds_all)
        prec = precision_score(masks_all, preds_all, zero_division=0)
        rec = recall_score(masks_all, preds_all, zero_division=0)
        f1 = f1_score(masks_all, preds_all, zero_division=0)
    else:
        # Avoid errors if the validation batch contains no oil (all zeros)
        acc, prec, rec, f1 = 0.0, 0.0, 0.0, 0.0
    
    print(f"\nEpoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val IoU={avg_val_iou:.4f}")
    print(f"Metrics (Pixel-level): Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f}")

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    # Save best model
    if avg_val_iou > best_val_iou:
        best_val_iou = avg_val_iou
        # Save model checkpoint in the project root
        torch.save(model.state_dict(), get_full_data_path("best_seg_model.pth"))
        print(f"✅ Saved best segmentation model with IoU={best_val_iou:.4f}")

print("🎯 Segmentation Training complete.")

# --- PLOT LOSS ---
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Segmentation Training Curve')
# Save loss curve in the project root
plt.savefig(get_full_data_path('seg_loss_curve.png'))
# plt.show()