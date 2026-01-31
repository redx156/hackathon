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


# ============ IMAGE QUALITY CHECKS ============
# Lightweight, rule-based heuristics to flag potentially unreliable scans

# Thresholds (conservative to avoid over-flagging clinical images)
BLUR_THRESHOLD = 100.0  # Laplacian variance below this = blurry
CONTRAST_THRESHOLD = 30.0  # Grayscale std below this = low contrast


def check_blur(image: np.ndarray) -> tuple[bool, float]:
    """
    Detect blur using Laplacian variance.
    
    Lower variance = more blur (fewer edges detected).
    Medical X-rays should have clear lung boundaries.
    
    Returns:
        is_blurry: True if image appears blurry
        score: Laplacian variance (higher = sharper)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Compute Laplacian variance (edge detection sensitivity)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    
    is_blurry = variance < BLUR_THRESHOLD
    return is_blurry, float(variance)


def check_contrast(image: np.ndarray) -> tuple[bool, float]:
    """
    Detect low contrast using grayscale intensity spread.
    
    Low std = pixel values clustered (low contrast).
    X-rays should have good contrast between lung/tissue/bone.
    
    Returns:
        is_low_contrast: True if image has poor contrast
        score: Standard deviation of pixel intensities
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Compute intensity spread
    std = float(np.std(gray))
    
    is_low_contrast = std < CONTRAST_THRESHOLD
    return is_low_contrast, std


def check_image_quality(image: np.ndarray) -> dict:
    """
    Run all quality checks on an image.
    
    Args:
        image: RGB or grayscale numpy array
    
    Returns:
        dict with:
            - is_low_quality: True if any check failed
            - issues: List of detected problems
            - blur_score: Laplacian variance
            - contrast_score: Intensity std
            - warning: Human-readable warning (or None if quality OK)
    """
    is_blurry, blur_score = check_blur(image)
    is_low_contrast, contrast_score = check_contrast(image)
    
    issues = []
    if is_blurry:
        issues.append("blurry/motion artifact")
    if is_low_contrast:
        issues.append("low contrast")
    
    is_low_quality = len(issues) > 0
    
    warning = None
    if is_low_quality:
        issue_text = " and ".join(issues)
        warning = (
            f"⚠️ Image quality notice: {issue_text} detected. "
            f"Prediction confidence may be less reliable. "
            f"Consider re-scanning if clinically feasible."
        )
    
    return {
        "is_low_quality": is_low_quality,
        "issues": issues,
        "blur_score": round(blur_score, 2),
        "contrast_score": round(contrast_score, 2),
        "warning": warning
    }


# ============ TEST BLOCK ============
if __name__ == "__main__":
    print("🧪 Testing utils.py preprocessing pipeline...\n")
    
    # Create a dummy test image (224x224 RGB)
    print("1. Creating dummy 224x224 RGB image...")
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    print(f"   ✅ Image shape: {dummy_image.shape}")
    
    # Convert to bytes (simulating file upload)
    print("\n2. Converting to bytes (simulating upload)...")
    pil_image = Image.fromarray(dummy_image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    image_bytes = buffer.getvalue()
    print(f"   ✅ Image bytes size: {len(image_bytes)} bytes")
    
    # Test preprocessing
    print("\n3. Running preprocess_image()...")
    tensor, original = preprocess_image(image_bytes)
    print(f"   ✅ Tensor shape: {tensor.shape}")
    print(f"   ✅ Tensor dtype: {tensor.dtype}")
    print(f"   ✅ Original image shape: {original.shape}")
    
    # Test base64 encoding
    print("\n4. Testing numpy_to_base64()...")
    b64_str = numpy_to_base64(original)
    print(f"   ✅ Base64 string length: {len(b64_str)} chars")
    
    # Test data URI
    print("\n5. Testing base64_to_data_uri()...")
    data_uri = base64_to_data_uri(b64_str)
    print(f"   ✅ Data URI starts with: {data_uri[:30]}...")
    
    print("\n" + "=" * 40)
    print("✅ All utils.py tests passed!")
    print("=" * 40)
