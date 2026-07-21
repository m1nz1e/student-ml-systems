# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder — install dependencies
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements first (for layer caching)
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Also install extras needed for survival analysis
RUN pip install --user --no-cache-dir scikit-survival

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime — minimal production image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

LABEL org.opencontainers.image.title="Student ML Systems"
LABEL org.opencontainers.image.description="Early warning system for student dropout prediction"
LABEL org.opencontainers.image.version="1.0.0"

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ /app/src/
COPY ui/ /app/ui/
COPY configs/ /app/configs/
COPY data/ /app/data/
COPY models/ /app/models/

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV MLFLOW_TRACKING_URI=postgresql://user:pass@db:5432/student_ml

# Expose ports
# 8501 — Streamlit
# 5000 — MLflow UI
EXPOSE 8501 5000

# Default: run Streamlit app
# Override with docker run <image> mlflow ui for just MLflow
CMD ["streamlit", "run", "ui/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
