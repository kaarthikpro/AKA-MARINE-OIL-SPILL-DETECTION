# src/infer_flask.py
import torch
import numpy as np
import cv2
from src.models import UNet, get_classifier
from src.data_utils import preprocess_for_inference
from src.dataset import NORMALIZE_TRANSFORM

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Models (global scope for speed) ---
SEG_MODEL = UNet().to(DEVICE)
CLS_MODEL = get_classifier(num_classes=1).to(DEVICE)

try:
    SEG_MODEL.load_state_dict(torch.load("best_seg_model.pth", map_location=DEVICE))
    SEG_MODEL.eval()
except Exception as e:
    print(f"Error loading segmentation model: {e}")
    SEG_MODEL = None

try:
    CLS_MODEL.load_state_dict(torch.load("best_cls_model.pth", map_location=DEVICE))
    CLS_MODEL.eval()
except Exception as e:
    print(f"Error loading classification model: {e}")
    CLS_MODEL = None

# --- Core Inference Function ---
def run_inference(image_np_rgb):
    results = {
        'oil_prob': -1,
        'prob_mask': None,
        'binary_mask': None
    }
    
    # 1. Classification (Probability)
    if CLS_MODEL:
        cls_input = preprocess_for_inference(image_np_rgb, size=(224, 224), normalize=NORMALIZE_TRANSFORM).to(DEVICE)
        with torch.no_grad():
            cls_logits = CLS_MODEL(cls_input)
            results['oil_prob'] = torch.sigmoid(cls_logits).item() * 100 # Convert to percentage

    # 2. Segmentation (Mask)
    if SEG_MODEL:
        # Preprocess without classification normalization
        seg_input = preprocess_for_inference(image_np_rgb, size=(256, 256)).to(DEVICE) 
        with torch.no_grad():
            seg_logits = SEG_MODEL(seg_input)
            prob_mask = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
            
            results['prob_mask'] = prob_mask # float 0-1
            # Convert mask to 0/255 for utility display
            results['binary_mask'] = (prob_mask > 0.5).astype(np.uint8) * 255
    
    return results