"""
Grad-CAM Implementation for Explainability
- Captures gradients from the last conv layer (layer4 in ResNet18)
- Generates heatmap showing regions influencing the prediction
- Overlays heatmap on original X-ray image in red
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM: Visual Explanations from Deep Networks
    
    For ResNet18, we hook into 'layer4' (the last convolutional block)
    to capture feature maps and gradients.
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture gradients and activations
        self._register_hooks()
    
    def _register_hooks(self):
        """Attach forward and backward hooks to target layer"""
        
        def forward_hook(module, input, output):
            # Store the output (feature maps) of the target layer
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            # Store the gradient flowing back through target layer
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap for the input image.
        
        Args:
            input_tensor: Preprocessed image tensor (1, 3, 224, 224)
            class_idx: Target class (0 or 1). If None, uses predicted class.
        
        Returns:
            cam: Normalized heatmap (224, 224), values 0-1
            prediction: Model's sigmoid output (probability)
        """
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        prediction = torch.sigmoid(output).item()
        
        # Use predicted class if not specified
        if class_idx is None:
            # For binary: if prediction > 0.5, class = 1 (pneumonia)
            class_idx = 1 if prediction > 0.5 else 0
        
        # Backward pass - compute gradients w.r.t. target class
        # For binary classification with sigmoid:
        # - If class_idx == 1 (pneumonia), we want regions that increase output
        # - If class_idx == 0 (normal), we want regions that decrease output
        if class_idx == 1:
            output.backward()
        else:
            (-output).backward()
        
        # Get gradients and activations
        gradients = self.gradients  # Shape: (1, 512, 7, 7)
        activations = self.activations  # Shape: (1, 512, 7, 7)
        
        # Global average pooling of gradients -> channel weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (1, 512, 1, 1)
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1).squeeze()  # (7, 7)
        
        # ReLU - only keep positive contributions
        cam = F.relu(cam)
        
        # Normalize to 0-1 range
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        # Resize to input image size (224x224)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        
        return cam, prediction


def create_heatmap_overlay(original_image: np.ndarray, cam: np.ndarray, alpha: float = 0.4):
    """
    Overlay Grad-CAM heatmap on original X-ray image.
    
    Args:
        original_image: Original image as numpy array (H, W, 3), uint8
        cam: Grad-CAM heatmap (H, W), values 0-1
        alpha: Blend factor (0=original, 1=heatmap)
    
    Returns:
        overlay: Blended image with red heatmap highlighting infected regions
    """
    # Resize CAM to match original image
    cam_resized = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))
    
    # Convert CAM to heatmap (red colormap for medical imaging)
    # Using JET colormap - blue (low) to red (high)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Ensure original image is RGB
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    elif original_image.shape[2] == 4:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)
    
    # Blend original with heatmap
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
    
    return overlay
