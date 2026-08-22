import numpy as np
from PIL import Image
from matplotlib.colors import rgb_to_hsv
from skimage.feature import graycomatrix, graycoprops
import config

def extract_hsv_color_features(image_path):
    """
    Extracts 9 HSV color features (mean, std, skewness for each channel H, S, V).
    
    Args:
        image_path (Path or str): Path to the image file.
        
    Returns:
        np.ndarray: Array of shape (9,) containing channel statistics.
    """
    # Load image and convert to RGB
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img, dtype=np.float32) / 255.0
    
    # Convert to HSV using matplotlib
    hsv_img = rgb_to_hsv(img_np)
    
    features = []
    for i in range(3):
        channel = hsv_img[:, :, i].ravel()
        mean = np.mean(channel)
        std = np.std(channel)
        
        # Calculate skewness safely
        if std < 1e-6:
            skew = 0.0
        else:
            skew = np.mean((channel - mean) ** 3) / (std ** 3)
            
        features.extend([mean, std, skew])
        
    return np.array(features, dtype=np.float32)

def extract_glcm_texture_features(image_path):
    """
    Extracts 4 GLCM texture features (contrast, correlation, energy, homogeneity)
    averaged across angles 0, 45, 90, 135 degrees.
    
    Args:
        image_path (Path or str): Path to the image file.
        
    Returns:
        np.ndarray: Array of shape (4,) containing averaged texture properties.
    """
    # Load image and convert to grayscale (uint8)
    img = Image.open(image_path).convert('L')
    img_np = np.array(img, dtype=np.uint8)
    
    # Convert angles to radians
    angles_rad = [np.deg2rad(a) for a in config.GLCM_ANGLES_DEG]
    
    # Compute GLCM
    glcm = graycomatrix(
        img_np, 
        distances=[config.GLCM_DISTANCE], 
        angles=angles_rad, 
        levels=256, 
        symmetric=True, 
        normed=True
    )
    
    # Extract properties and average over orientations
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    correlation = np.mean(graycoprops(glcm, 'correlation'))
    energy = np.mean(graycoprops(glcm, 'energy'))
    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
    
    return np.array([contrast, correlation, energy, homogeneity], dtype=np.float32)

def extract_all_features(image_path):
    """
    Extracts both HSV color features and GLCM texture features for an image.
    
    Args:
        image_path (Path or str): Path to the image file.
        
    Returns:
        tuple: (color_features, texture_features)
    """
    color_feats = extract_hsv_color_features(image_path)
    texture_feats = extract_glcm_texture_features(image_path)
    return color_feats, texture_feats
