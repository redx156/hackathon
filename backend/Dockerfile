# ===================================================================
# Backend Dockerfile for Pneumonia Detection API
# ===================================================================
# Key fixes:
# 1. ALL dependencies installed at build time (not runtime)
# 2. numpy<2 enforced BEFORE torch to prevent conflicts
# 3. OpenCV headless + system libs for slim image compatibility
# 4. No runtime pip install required
# ===================================================================

FROM python:3.12-slim-bookworm

WORKDIR /app

# -----------------------------------------------------------------
# 1. Install system dependencies for OpenCV and image processing
# -----------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------
# 2. Install numpy<2 FIRST to prevent torch/opencv conflicts
# -----------------------------------------------------------------
RUN pip install --no-cache-dir "numpy<2"

# -----------------------------------------------------------------
# 3. Install PyTorch and TorchVision
# Using standard versions - will install CPU-only on Linux slim
# -----------------------------------------------------------------
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

# -----------------------------------------------------------------
# 4. Install OpenCV headless (no GUI dependencies needed)
# -----------------------------------------------------------------
RUN pip install --no-cache-dir opencv-python-headless==4.8.0.76

# -----------------------------------------------------------------
# 5. Install FastAPI and other Python dependencies
# -----------------------------------------------------------------
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic \
    sqlalchemy \
    python-multipart \
    pillow \
    python-dotenv \
    psycopg2-binary

# -----------------------------------------------------------------
# 6. Copy application code and model checkpoints
# -----------------------------------------------------------------
COPY backend ./backend
COPY checkpoints ./checkpoints

# -----------------------------------------------------------------
# 7. Expose API port
# -----------------------------------------------------------------
EXPOSE 8000

# -----------------------------------------------------------------
# 8. Run FastAPI via Uvicorn
# -----------------------------------------------------------------
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
