"""
PyTorch Model Loading and Configuration
- Uses pretrained ResNet18 (ImageNet weights)
- Modifies final layer for binary classification (Normal vs Pneumonia)
- Runs on GPU if available, else CPU
"""

import torch
import torch.nn as nn
from torchvision import models


def get_device():
    """Returns CUDA if available, else CPU"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(weights_path: str = None):
    """
    Load pretrained ResNet18 and modify for binary classification.
    
    Args:
        weights_path: Path to fine-tuned weights (.pth file). 
                      If None, uses ImageNet pretrained weights.
    
    Returns:
        model: The loaded model in eval mode
        device: The device (CPU/CUDA) being used
    """
    device = get_device()
    
    # Load ResNet18 with pretrained ImageNet weights
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Replace final fully connected layer for binary classification
    # Original: 512 -> 1000 classes
    # New: 512 -> 1 output (sigmoid for probability)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)
    
    # Load fine-tuned weights if provided
    if weights_path:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded fine-tuned weights from {weights_path}")
    else:
        print("Using ImageNet pretrained weights (no fine-tuning)")
    
    # Move to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    return model, device


# Global model instance (loaded once at startup)
_model = None
_device = None


def get_model():
    """Get the global model instance (singleton pattern)"""
    global _model, _device
    if _model is None:
        _model, _device = load_model()
    return _model, _device
