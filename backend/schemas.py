
from pydantic import BaseModel
from typing import Optional


class PredictionResponse(BaseModel):
    
    # Prediction result: "Normal" or "Pneumonia"
    prediction: str
    
    # Confidence score (0-1), represents P(pneumonia)
    confidence: float
    
    # Risk level based on threshold
    risk_level: str  # "Low Risk", "Medium Risk", "High Risk"
    
    # Base64 encoded heatmap overlay image
    heatmap_image: str
    
    # Original image with overlay as data URI (ready for <img src>)
    heatmap_data_uri: str
    
    # Device used for inference
    device: str
    
    # Image quality flag (informational, not decision-altering)
    low_image_quality: bool = False
    
    # Additional info for doctors
    note: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
