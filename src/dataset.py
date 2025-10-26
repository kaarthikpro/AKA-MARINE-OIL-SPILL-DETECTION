# src/dataset.py
import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

# --- Base Augmentations (to be used with both Seg/Cls) ---
IMAGE_SIZE = 256
NORMALIZE_TRANSFORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# --- Segmentation Dataset ---
class SegmentationDataset(Dataset):
    def __init__(self, csv_file, image_dir="data/oil_spill", mask_dir="data/masks", transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.iloc[idx]['filename']
        img_path = os.path.join(self.image_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)

        # Read RGB image and grayscale mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Preprocessing (Resize and Normalize)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0

        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0) # (1, H, W)

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        
        # Apply transform (e.g., normalization)
        if self.transform:
            image = self.transform(image)
        
        return image, mask

# --- Classification Dataset ---
class ClassificationDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # CSV must contain 'path' (full path to image) and 'label' (0 or 1)
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['path']
        label = row['label']

        # Read RGB image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocessing
        image = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0 # ResNet size
        image = np.transpose(image, (2, 0, 1)) # HWC -> CHW

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor([label], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label