"""
FastAPI Application Entry Point
- Pneumonia Detection API for Chest X-ray Analysis
- Uses pretrained ResNet18 + Grad-CAM for explainability

Run with: uvicorn backend.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .routes import router
from .models import get_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown lifecycle.
    Pre-loads the model at startup to avoid first-request delay.
    """
    print("🚀 Starting Pneumonia Detection API...")
    print("📦 Loading ResNet18 model...")
    model, device = get_model()
    print(f"✅ Model loaded on {device}")
    yield
    print("👋 Shutting down...")


app = FastAPI(
    title="Pneumonia Detection API",
    description="AI-Assisted Chest X-ray Analysis with Grad-CAM Explainability",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for hackathon demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(router, prefix="/api", tags=["Prediction"])


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "message": "Pneumonia Detection API",
        "docs": "/docs",
        "predict_endpoint": "/api/predict",
        "health_endpoint": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
