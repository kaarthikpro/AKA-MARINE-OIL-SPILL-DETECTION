# flask_app/app.py (FINAL VERSION with Heatmap Generation)

import os
import cv2
import numpy as np
import torch
import time
import inspect
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
import sys

# --- Path Utility ---
CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) 

sys.path.append(PROJECT_ROOT)

# --- Imports from src/ ---
from src.data_utils import get_hog_visualization, overlay_mask, preprocess_for_inference
from src.models import UNet, get_classifier
from src.dataset import NORMALIZE_TRANSFORM # Import the transform directly

# --- Configuration ---
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(CURRENT_DIR, 'static/uploads')
RESULTS_FOLDER = os.path.join(CURRENT_DIR, 'static/results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model variables (will be populated in load_models)
seg_model = None
cls_model = None

# --- Load Models Function (Called once at startup) ---
def load_models():
    print("Loading models...")
    global seg_model
    global cls_model

    # 1. Segmentation Model (U-Net)
    seg_model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    try:
        seg_model_path = os.path.join(PROJECT_ROOT, "best_seg_model.pth")
        seg_model.load_state_dict(torch.load(seg_model_path, map_location=DEVICE))
        seg_model.eval()
        print("✅ Segmentation Model Loaded.")
    except Exception as e:
        print(f"❌ Error loading segmentation model: {e}. Running without segmentation.")
        seg_model = None

    # 2. Classification Model (ResNet)
    cls_model = get_classifier(num_classes=1).to(DEVICE)
    try:
        cls_model_path = os.path.join(PROJECT_ROOT, "best_cls_model.pth")
        cls_model.load_state_dict(torch.load(cls_model_path, map_location=DEVICE))
        cls_model.eval()
        print("✅ Classification Model Loaded.")
    except Exception as e:
        print(f"❌ Error loading classification model: {e}. Running without classification.")
        cls_model = None
        

# --- Inference Function ---
def process_image(img_path):
    
    start_time = time.time() # Start benchmark timer
    
    img_np = cv2.imread(img_path)
    if img_np is None:
        raise FileNotFoundError(f"Could not read image at {img_path}")
        
    img_np_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    
    # 1. Classification (Probability)
    oil_prob = -1
    if cls_model:
        # Preprocess with ResNet-specific normalization
        cls_input = preprocess_for_inference(img_np_rgb, size=(224, 224), normalize=NORMALIZE_TRANSFORM).to(DEVICE)
        with torch.no_grad():
            cls_logits = cls_model(cls_input)
            oil_prob = torch.sigmoid(cls_logits).item() * 100
    
    # 2. Segmentation and Feature Visualization
    mask_overlay_file, hog_file, bw_file, heatmap_file = None, None, None, None # <--- ADDED HEATMAP FILE
    base_name = os.path.basename(img_path).split('.')[0]

    # Feature Visualizations (BW, HOG)
    bw_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    bw_path = os.path.join(app.config['RESULTS_FOLDER'], f"{base_name}_bw.png")
    cv2.imwrite(bw_path, bw_img)
    bw_file = os.path.basename(bw_path)
    
    hog_img = get_hog_visualization(img_np_rgb)
    hog_path = os.path.join(app.config['RESULTS_FOLDER'], f"{base_name}_hog.png")
    cv2.imwrite(hog_path, hog_img)
    hog_file = os.path.basename(hog_path)

    if seg_model:
        # Segmentation Inference
        seg_input = preprocess_for_inference(img_np_rgb, size=(256, 256)).to(DEVICE)
        with torch.no_grad():
            seg_logits = seg_model(seg_input)
            prob_mask = torch.sigmoid(seg_logits).squeeze().cpu().numpy()
        
        # --- GENERATE HEATMAP ---
        # Resize probability mask to original image dimensions for heatmap
        prob_mask_display = cv2.resize(prob_mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Normalize to 0-255 and apply a colormap (COLORMAP_JET is good for heatmaps)
        heatmap = (prob_mask_display * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Save heatmap result
        heatmap_path = os.path.join(app.config['RESULTS_FOLDER'], f"{base_name}_heatmap.png")
        cv2.imwrite(heatmap_path, heatmap_colored)
        heatmap_file = os.path.basename(heatmap_path) # <--- STORE HEATMAP FILE NAME
        # --- END HEATMAP GENERATION ---

        # Convert mask to 0/255 for overlay
        binary_mask = (prob_mask > 0.5).astype(np.uint8) * 255
        
        # Overlay mask (resize original RGB to mask size for overlay, then resize back)
        overlay_img = overlay_mask(img_np_rgb, binary_mask, color=(255, 0, 0), alpha=0.5)
        
        # Save mask overlay result
        mask_overlay_path = os.path.join(app.config['RESULTS_FOLDER'], f"{base_name}_overlay.png")
        cv2.imwrite(mask_overlay_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
        mask_overlay_file = os.path.basename(mask_overlay_path)

    end_time = time.time() # Stop benchmark timer
    inference_time = f"{end_time - start_time:.2f} seconds" # Calculate and format time

    return oil_prob, mask_overlay_file, hog_file, bw_file, heatmap_file, inference_time # <--- RETURN HEATMAP FILE

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            
            # Process the image, includes benchmarking
            oil_prob, overlay_file, hog_file, bw_file, heatmap_file, inference_time = process_image(upload_path) # <--- RECEIVE HEATMAP FILE
            
            # Check if models failed to load, set placeholder probability
            prob_display = f"{oil_prob:.2f}%" if oil_prob >= 0 else "N/A (Model Error)"
            
            return render_template('result.html', 
                                   original_file=os.path.basename(upload_path),
                                   oil_probability=prob_display,
                                   overlay_file=overlay_file,
                                   hog_file=hog_file,
                                   bw_file=bw_file,
                                   heatmap_file=heatmap_file, # <--- PASS HEATMAP FILE TO TEMPLATE
                                   inference_time=inference_time) 
    
    return render_template('index.html')

@app.route('/static/<folder>/<filename>')
def serve_file(folder, filename):
    full_path = os.path.join(CURRENT_DIR, 'static', folder)
    return send_from_directory(full_path, filename)

if __name__ == '__main__':
    load_models() 
    
    print("\n--- Flask App Ready ---")
    app.run(debug=True, host='0.0.0.0', port=5000)