import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from scipy import ndimage

def generate_mask_advanced(image_path, out_path, method='multispec'):
    """
    Fully automated mask generation without manual intervention.
    Methods:
    - 'multispec': Multi-spectral thresholding (most reliable)
    - 'kmeans': K-means clustering
    - 'statistical': Statistical anomaly detection
    """
    img = cv2.imread(image_path)
    if img is None:
        return
    
    h, w = img.shape[:2]
    
    if method == 'multispec':
        mask = multispec_thresholding(img)
    elif method == 'kmeans':
        mask = kmeans_segmentation(img)
    elif method == 'statistical':
        mask = statistical_anomaly(img)
    else:
        mask = multispec_thresholding(img)
    
    # Post-processing: remove small objects, fill holes
    mask = post_process_mask(mask)
    
    cv2.imwrite(out_path, mask)
    print(f"Generated mask for {os.path.basename(image_path)}")

def multispec_thresholding(img):
    """
    Multi-channel thresholding combining color info.
    Oil typically appears as dark (low L) + desaturated or brownish.
    """
    # Convert to different color spaces
    bgr = img.astype(np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Channel extraction
    L = lab[:,:,0]           # Lightness (0-255)
    a = lab[:,:,1]           # Green-Red (-128 to 127, shifted to 0-255)
    b = lab[:,:,2]           # Blue-Yellow (-128 to 127, shifted to 0-255)
    
    H = hsv[:,:,0] * 2       # Hue (0-180 in OpenCV, scale to 0-360)
    S = hsv[:,:,1]           # Saturation (0-255)
    V = hsv[:,:,2]           # Value (0-255)
    
    # Oil detection heuristics:
    # 1. Dark (low L)
    dark = L < 120
    
    # 2. Not pure water blue/cyan (hue range 80-180 degrees)
    not_water_hue = (H < 80) | (H > 200)
    
    # 3. Water has higher saturation consistency; oil is often brownish/desaturated
    brownish = (H >= 15) & (H <= 40)  # Brown/reddish hues
    desaturated_dark = (L < 100) & (S < 100)
    
    # 4. Contrast in a-b channels indicates objects
    ab_contrast = (np.abs(a - 128) > 15) | (np.abs(b - 128) > 15)
    
    # Combine conditions
    mask = (dark & (brownish | desaturated_dark | not_water_hue)) | (desaturated_dark & ab_contrast)
    mask = mask.astype(np.uint8) * 255
    
    # Adaptive thresholding refinement
    L_gray = L.astype(np.uint8)
    adaptive = cv2.adaptiveThreshold(L_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 41, 15)
    
    # Combine with voting (AND for intersection of high-confidence regions)
    mask = cv2.bitwise_and(mask, adaptive)
    
    return mask

def kmeans_segmentation(img):
    """
    K-means clustering on color features.
    Assumes oil is one distinct cluster.
    """
    h, w = img.shape[:2]
    
    # Feature extraction: color + position
    bgr = img.reshape(-1, 3).astype(np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    
    # Combine features (color dominates)
    features = np.hstack([lab, bgr * 0.3])  # Lab has more weight
    
    # K-means (typically 3 clusters: water, oil, sky/mixed)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    # Identify oil cluster (darkest cluster by L channel)
    centers = kmeans.cluster_centers_
    L_vals = centers[:, 0]
    oil_cluster = np.argmin(L_vals)
    
    # Create mask
    mask = (labels == oil_cluster).astype(np.uint8) * 255
    mask = mask.reshape(h, w)
    
    return mask

def statistical_anomaly(img):
    """
    Detect oil as anomalies in water statistics.
    Use local statistics: oil regions deviate from water background.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:,:,0]
    
    # Compute local statistics (mean and std of L channel)
    kernel_size = 51
    local_mean = cv2.blur(L, (kernel_size, kernel_size))
    sq_diff = (L - local_mean) ** 2
    local_var = cv2.blur(sq_diff, (kernel_size, kernel_size))
    local_std = np.sqrt(np.maximum(local_var, 0))
    
    # Anomaly: regions significantly darker than local mean
    anomaly_score = (local_mean - L) / (local_std + 1e-5)
    
    # Threshold on anomaly score
    mask = (anomaly_score > 2.5).astype(np.uint8) * 255
    
    return mask

def post_process_mask(mask):
    """
    Clean up mask: remove noise, fill gaps, ensure connectivity.
    """
    # Morphological closing to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Morphological opening to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Remove small connected components (noise)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    
    min_area = 300  # Tune based on image size
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(clean_mask, [contour], -1, 255, -1)
    
    # Dilate slightly to ensure oil regions are fully captured
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_mask = cv2.dilate(clean_mask, kernel, iterations=1)
    
    return clean_mask

if __name__ == '__main__':
    src_dir = 'data/oil_spill'
    dst_dir = 'data/masks'
    os.makedirs(dst_dir, exist_ok=True)
    
    # Choose method: 'multispec' (recommended), 'kmeans', or 'statistical'
    method = 'multispec'
    
    for fname in os.listdir(src_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(dst_dir, fname)
            generate_mask_advanced(src_path, dst_path, method=method)
    
    print(f"Mask generation complete using {method} method.")