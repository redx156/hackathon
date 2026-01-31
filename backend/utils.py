"""
Image Preprocessing Pipeline
- Loads and preprocesses X-ray images for ResNet18 inference
- Handles various input formats (file upload, file path)
- Applies ImageNet normalization (required for pretrained weights)
"""

import io
import base64
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms


# ImageNet normalization (REQUIRED for pretrained ResNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_preprocessing_transform():
    """
    Returns the preprocessing pipeline for X-ray images.
    
    Steps:
    1. Resize to 224x224 (ResNet input size)
    2. Convert to tensor (0-1 range)
    3. Normalize with ImageNet stats
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def preprocess_image(image_bytes: bytes):
    """
    Preprocess image bytes for model inference.
    
    Args:
        image_bytes: Raw image bytes from upload
    
    Returns:
        tensor: Preprocessed tensor (1, 3, 224, 224)
        original_image: Original image as numpy array (for heatmap overlay)
    """
    # Load image from bytes
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB (X-rays may be grayscale)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Keep original for heatmap overlay
    original_image = np.array(image)
    
    # Apply preprocessing
    transform = get_preprocessing_transform()
    tensor = transform(image)
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor, original_image


def preprocess_image_from_path(image_path: str):
    """
    Preprocess image from file path.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        tensor: Preprocessed tensor (1, 3, 224, 224)
        original_image: Original image as numpy array
    """
    with open(image_path, 'rb') as f:
        return preprocess_image(f.read())


def numpy_to_base64(image: np.ndarray) -> str:
    """
    Convert numpy image to base64 string for API response.
    
    Args:
        image: RGB image as numpy array
    
    Returns:
        base64 encoded PNG string
    """
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Encode as PNG
    _, buffer = cv2.imencode('.png', image_bgr)
    
    # Convert to base64
    base64_str = base64.b64encode(buffer).decode('utf-8')
    
    return base64_str


def base64_to_data_uri(base64_str: str, mime_type: str = "image/png") -> str:
    """Convert base64 string to data URI for direct embedding"""
    return f"data:{mime_type};base64,{base64_str}"
