# split_cls_data.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

IMG_DIR_OIL = "data/oil_spill"
IMG_DIR_NON = "data/non_oil_spill"
OUT_DIR = "data/splits"

# 1. Get oil spill files (Label 1)
oil_files = [{'filename': f, 'label': 1, 'path': os.path.join(IMG_DIR_OIL, f)} 
             for f in os.listdir(IMG_DIR_OIL) if f.lower().endswith(('.jpg','.jpeg','.png'))]
# 2. Get non-oil spill files (Label 0)
non_oil_files = [{'filename': f, 'label': 0, 'path': os.path.join(IMG_DIR_NON, f)} 
                 for f in os.listdir(IMG_DIR_NON) if f.lower().endswith(('.jpg','.jpeg','.png'))]

df = pd.DataFrame(oil_files + non_oil_files)

# Split 80/10/10
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

train_df.to_csv(os.path.join(OUT_DIR, 'cls_train.csv'), index=False)
val_df.to_csv(os.path.join(OUT_DIR, 'cls_val.csv'), index=False)
test_df.to_csv(os.path.join(OUT_DIR, 'cls_test.csv'), index=False)

print("✅ Classification splits complete — CSVs saved in data/splits/ (cls_train.csv, etc.)")