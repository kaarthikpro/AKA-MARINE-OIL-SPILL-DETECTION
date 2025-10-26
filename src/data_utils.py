# src/data_utils.py (Final Version with HOG Fix and Correct Function Name)
import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
import torch

# --- Preprocessing for Flask App ---
def preprocess_for_inference(image_np, size=(256, 256), normalize=None):
    """
    Preprocesses a NumPy image for the model. 
    Handles resizing, normalization (0-1), HWC -> CHW conversion,
    and applies optional external normalization (like ResNet normalization).
    """
    img = cv2.resize(image_np, size)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1)) # HWC -> CHW
    tensor = torch.tensor(img, dtype=torch.float32)
    
    if normalize:
        # Apply torchvision.transforms.Normalize object (e.g., NORMALIZE_TRANSFORM)
        tensor = normalize(tensor)
    
    return tensor.unsqueeze(0) # Add batch dim

# --- HOG Feature Extraction ---
def get_hog_visualization(image_np):
    """Computes and visualizes HOG features."""
    if len(image_np.shape) == 3:
        # Convert RGB to Grayscale
        image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        image = image_np
        
    # Standard HOG parameters
    # FIX: Removed 'multichannel=False' to prevent TypeError in newer scikit-image
    fd, hog_image = hog(image.astype('uint8'), 
                        orientations=9, 
                        pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), 
                        visualize=True 
                        ) 
    
    # Rescale HOG image to 0-255 for visualization
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 1))
    hog_image_rescaled = (hog_image_rescaled * 255).astype(np.uint8)
    
    return hog_image_rescaled

# --- Mask Overlay Utility ---
def overlay_mask(image_np, mask_np, color=(255, 0, 0), alpha=0.5):
    """Overlays a binary mask onto the original image."""
    h, w = image_np.shape[:2]
    # Ensure mask is the same size
    mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Convert to binary mask (0 or 255)
    if mask_resized.max() <= 1.0:
        binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
    else:
        binary_mask = (mask_resized > 127).astype(np.uint8) * 255
    
    # Create a colored overlay layer
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    color_mask[:] = color
    
    # Apply the mask
    overlay = cv2.bitwise_and(color_mask, color_mask, mask=binary_mask)
    
    # Blend with original image
    image_float = image_np.astype(np.float32)
    overlay_float = overlay.astype(np.float32)
    
    combined = cv2.addWeighted(image_float, 1.0 - alpha, overlay_float, alpha, 0)
    
    return combined.astype(np.uint8)