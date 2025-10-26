# src/evaluate.py (Final Version with Metrics File Saving)

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import json # New import for saving metrics
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import inspect
from torchvision import transforms 
# Import necessary files
from src.dataset import SegmentationDataset, ClassificationDataset, NORMALIZE_TRANSFORM
from src.models import UNet, get_classifier
from src.train_seg import iou_score # Reusing IoU utility

# --- Path Utility ---
CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) 
def get_full_data_path(relative_path):
    return os.path.join(PROJECT_ROOT, relative_path)

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEG_MODEL_PATH = get_full_data_path("best_seg_model.pth")
CLS_MODEL_PATH = get_full_data_path("best_cls_model.pth")
METRICS_OUTPUT_FILE = get_full_data_path("test_metrics_results.json") # New output file

# Global dictionary to store all results
ALL_RESULTS = {}


# --- Function to Save Metrics to JSON ---
def save_metrics_to_file():
    """Saves the contents of the ALL_RESULTS dictionary to a JSON file."""
    try:
        with open(METRICS_OUTPUT_FILE, 'w') as f:
            # Use indent for readability
            json.dump(ALL_RESULTS, f, indent=4)
        print(f"\n✅ All metrics saved successfully to: {METRICS_OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ Error saving metrics to JSON: {e}")


# --- 1. Segmentation Evaluation (Pixel-level Metrics) ---
def evaluate_segmentation():
    print("\n--- Evaluating Segmentation Model (U-Net) on TEST SET ---")
    
    test_csv = get_full_data_path("data/splits/test_seg.csv")
    test_ds = SegmentationDataset(test_csv, transform=NORMALIZE_TRANSFORM)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    
    model = UNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load(SEG_MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"❌ Error loading segmentation weights: {e}")
        return
        
    model.eval()

    total_iou = 0
    all_preds_seg, all_masks_seg = [], []

    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="Seg Eval"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits = model(imgs)
            preds_prob = torch.sigmoid(logits)
            
            total_iou += iou_score(preds_prob, masks).item() * len(imgs)
            
            preds_binary = (preds_prob > 0.5).float().cpu().numpy().flatten()
            masks_binary = (masks > 0.5).float().cpu().numpy().flatten()
            
            all_preds_seg.extend(preds_binary)
            all_masks_seg.extend(masks_binary)

    avg_iou = total_iou / len(test_ds)
    
    # Calculate Dice coefficient robustly from overall pixels
    inter = np.sum(np.array(all_preds_seg) * np.array(all_masks_seg))
    union = np.sum(np.array(all_preds_seg)) + np.sum(np.array(all_masks_seg))
    avg_dice = (2.0 * inter + 1e-6) / (union + 1e-6)
    
    # Pixel-level Metrics
    acc = accuracy_score(all_masks_seg, all_preds_seg)
    prec = precision_score(all_masks_seg, all_preds_seg, zero_division=0)
    rec = recall_score(all_masks_seg, all_preds_seg, zero_division=0)
    f1 = f1_score(all_masks_seg, all_preds_seg, zero_division=0)
    cm = confusion_matrix(all_masks_seg, all_preds_seg)


    print(f"\n[SEGMENTATION RESULTS (TEST SET)]")
    print(f"Avg IoU (Jaccard Index): {avg_iou:.4f}")
    print(f"Avg Dice Coefficient: {avg_dice:.4f}")
    print(f"Pixel Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1 Score: {f1:.4f}")
    print("\nPixel-level Confusion Matrix (TN/FP; FN/TP):\n", cm)

    # Store results in the global dictionary
    ALL_RESULTS['segmentation'] = {
        'Avg_IoU': round(avg_iou, 4),
        'Avg_Dice_Coefficient': round(avg_dice, 4),
        'Pixel_Accuracy': round(acc, 4),
        'Pixel_Precision': round(prec, 4),
        'Pixel_Recall': round(rec, 4),
        'Pixel_F1_Score': round(f1, 4),
        'Confusion_Matrix': cm.tolist() # Convert NumPy array to list for JSON
    }


# --- 2. Classification Evaluation (Image-level Metrics + Plots) ---
def evaluate_classification():
    print("\n--- Evaluating Classification Model (ResNet) on TEST SET ---")
    
    test_csv = get_full_data_path("data/splits/test_cls.csv")
    # Classification transforms for ResNet-50 input (224x224)
    cls_transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize(224), transforms.ToTensor(), NORMALIZE_TRANSFORM
    ])
    test_ds = ClassificationDataset(test_csv, transform=cls_transform)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    model = get_classifier(num_classes=1).to(DEVICE)
    try:
        model.load_state_dict(torch.load(CLS_MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"❌ Error loading classification weights: {e}")
        return
        
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Cls Eval"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()
            
            all_probs.extend(probs)
            all_labels.extend(labels_np)

    # Metrics
    auc = roc_auc_score(all_labels, all_probs)
    preds_binary = (np.array(all_probs) > 0.5).astype(int) # Binary predictions at default threshold
    
    acc = accuracy_score(all_labels, preds_binary)
    prec = precision_score(all_labels, preds_binary, zero_division=0)
    rec = recall_score(all_labels, preds_binary, zero_division=0)
    f1 = f1_score(all_labels, preds_binary, zero_division=0)
    cm = confusion_matrix(all_labels, preds_binary)
    
    print(f"\n[CLASSIFICATION RESULTS (TEST SET)]")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1 Score: {f1:.4f}")
    print("\nImage-level Confusion Matrix (Threshold 0.5):\n", cm)

    # Plotting ROC/PR Curves for Journal
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Classification ROC Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Classification Precision-Recall Curve')
    plt.legend()
    
    # Save high-resolution PNG in project root
    plt.tight_layout()
    plt.savefig(get_full_data_path('journal_metrics.png'))
    
    # Store results in the global dictionary
    ALL_RESULTS['classification'] = {
        'AUC': round(auc, 4),
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1_Score': round(f1, 4),
        'Confusion_Matrix': cm.tolist() 
    }


if __name__ == '__main__':
    evaluate_segmentation()
    evaluate_classification()
    # Call the save function after both evaluations are complete
    save_metrics_to_file()
    plt.show() # Display the plot only after saving both metrics and plot