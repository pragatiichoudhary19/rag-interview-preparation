
FROM python:3.11-slim

# Python environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements from backend build context
COPY requirements.txt /srv/requirements.txt

# Install Python dependencies
RUN pip install -r /srv/requirements.txt

# Copy backend application
COPY app /srv/app

# Create application user and data directory
RUN useradd -m appuser && \
    mkdir -p /srv/data && \
    chown -R appuser:appuser /srv

USER appuser

# Application environment
ENV PYTHONPATH=/srv \
    HF_HOME=/srv/data/hf_cache

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
    CMD curl -fs http://localhost:8000/health || exit 1

# Start FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

