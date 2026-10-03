# ==============================================================================
# Dockerfile: Dynamic Named Entity Recognition (NER) Microservice
# ==============================================================================
# Multi-stage production container configuration for serving GLiNER/spaCy models.
# Optimized for minimal image size, build layer caching, and reproducible runtime.
# ==============================================================================

# 1. Base Image: Use official Debian slim Python runtime for lightweight footprint
FROM python:3.10-slim

# 2. Environment Variables:
#    - PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files to disk
#    - PYTHONUNBUFFERED: Flushes stdout/stderr immediately for real-time Docker logging
#    - MODEL_PATH: Configures the location where ModelRunner searches for local weights
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/model

# 3. Working Directory: Set root project context inside container
WORKDIR /app

# 4. Dependency Installation (Optimized Layer Caching):
#    Copy requirements.txt independently so pip installation is re-run ONLY when
#    dependencies change, not when application source code is updated.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Model Artifacts:
#    Copy the local model directory (contains weights, tokenizers, and configs).
#    Note: In production with dynamic models, you can also mount external volumes
#    to /app/model at container runtime: `docker run -v /host/model:/app/model ...`
COPY ./model ./model

# 6. Application Code:
#    Copy the FastAPI router, data schemas, and text processing utilities
COPY ./app ./app

# 7. Network Port:
#    Expose HTTP port 8000 for ingress microservice communication
EXPOSE 8000

# 8. Service Entry Point:
#    Launch Uvicorn ASGI server binding to all network interfaces (0.0.0.0)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]