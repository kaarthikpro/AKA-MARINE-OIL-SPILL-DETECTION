# src/train_cls.py (FINAL VERSION with 5 EPOCHS)

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torchvision import transforms
import cv2 
import numpy as np
import torch.optim as optim
import inspect

from src.models import get_classifier
# Assuming ClassificationDataset and NORMALIZE_TRANSFORM are defined in src/dataset.py
from src.dataset import ClassificationDataset, NORMALIZE_TRANSFORM 


# --- Path Utility (Robust Fix) ---
# Get the directory where train_cls.py lives
CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# Go up one level to the project root
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) 

def get_full_data_path(relative_path):
    """Constructs an absolute path from the project root."""
    return os.path.join(PROJECT_ROOT, relative_path)
# --- End Path Utility ---


# --- CONFIG ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Use robust path utility to locate CSVs
train_csv = get_full_data_path("data/splits/train_cls.csv")
val_csv   = get_full_data_path("data/splits/val_cls.csv")

BATCH_SIZE = 8
EPOCHS = 5  # <-- CHANGED TO 5 EPOCHS
LR = 1e-5

# --- Transforms (Normalize for pre-trained ResNet) ---
# Classification transforms for ResNet-50 input (224x224)
train_tfms = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.Resize(224), # Ensure standard input size
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(), 
    NORMALIZE_TRANSFORM 
])

val_tfms   = transforms.Compose([
    transforms.ToPILImage(), 
    transforms.Resize(224),
    transforms.ToTensor(), 
    NORMALIZE_TRANSFORM 
])

# --- Load Data & Model ---
train_ds = ClassificationDataset(train_csv, transform=train_tfms)
val_ds   = ClassificationDataset(val_csv, transform=val_tfms)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

model = get_classifier(num_classes=1).to(device) # ResNet-50
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_auc = 0.0

# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    avg_train_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()
            
            all_probs.extend(probs)
            all_labels.extend(labels_np)

    # Metrics
    auc = roc_auc_score(all_labels, all_probs)
    
    print(f"\nEpoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val AUC={auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        # Save model checkpoint using full path
        torch.save(model.state_dict(), get_full_data_path("best_cls_model.pth"))
        print(f"✅ Saved best classification model with AUC={best_auc:.4f}")

print("🎯 Classification Training complete.")