# src/split_data.py (Updated with Path Fix)
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Path Utility ---
# Get the absolute path to the project root directory
# This makes paths reliable regardless of how the script is executed.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_full_path(relative_path):
    return os.path.join(PROJECT_ROOT, relative_path)

# --- Configuration ---
# Use the full path utility for all directories
IMG_DIR_OIL = get_full_path("data/oil_spill")
IMG_DIR_NON = get_full_path("data/non_oil_spill")
OUT_DIR = get_full_path("data/splits")
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Segmentation Splits (Only uses oil_spill for mask training)
# Check if the directory exists before listing files
if not os.path.isdir(IMG_DIR_OIL):
    print(f"❌ Error: Oil spill directory not found at {IMG_DIR_OIL}")
    exit()

files = [f for f in os.listdir(IMG_DIR_OIL) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
seg_df = pd.DataFrame({'filename': files})
# ... (rest of segmentation split logic) ...

# 2. Classification Splits (Uses both oil_spill and non_oil_spill)
# ... (rest of classification logic) ...
oil_files = [{'filename': f, 'label': 1, 'path': os.path.join(IMG_DIR_OIL, f)} 
             for f in os.listdir(IMG_DIR_OIL) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
non_oil_files = [{'filename': f, 'label': 0, 'path': os.path.join(IMG_DIR_NON, f)} 
                 for f in os.listdir(IMG_DIR_NON) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
# NOTE: The 'path' column in classification CSVs now uses the full absolute path.
# This makes ClassificationDataset more robust.

# ... (Final CSV saving logic remains the same) ...

train_seg_df, temp_seg_df = train_test_split(seg_df, test_size=0.2, random_state=42)
val_seg_df, test_seg_df = train_test_split(temp_seg_df, test_size=0.5, random_state=42)

train_seg_df.to_csv(os.path.join(OUT_DIR, 'train_seg.csv'), index=False)
val_seg_df.to_csv(os.path.join(OUT_DIR, 'val_seg.csv'), index=False)
test_seg_df.to_csv(os.path.join(OUT_DIR, 'test_seg.csv'), index=False)

print("✅ Segmentation splits complete (train_seg.csv, etc.)")


# 2. Classification Splits
# (Assuming the logic below is the same, using the new IMG_DIR_OIL and IMG_DIR_NON)
cls_df = pd.DataFrame(oil_files + non_oil_files)

train_cls_df, temp_cls_df = train_test_split(cls_df, test_size=0.2, random_state=42, stratify=cls_df['label'])
val_cls_df, test_cls_df = train_test_split(temp_cls_df, test_size=0.5, random_state=42, stratify=temp_cls_df['label'])

train_cls_df.to_csv(os.path.join(OUT_DIR, 'train_cls.csv'), index=False)
val_cls_df.to_csv(os.path.join(OUT_DIR, 'val_cls.csv'), index=False)
test_cls_df.to_csv(os.path.join(OUT_DIR, 'test_cls.csv'), index=False)

print("✅ Classification splits complete (train_cls.csv, etc.)")