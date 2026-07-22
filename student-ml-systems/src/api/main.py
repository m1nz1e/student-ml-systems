"""
FastAPI application for Student ML Systems.

Provides REST API endpoints for:
- Course Recommender
- Enrollment Yield Prediction
- Early Warning System
- Degree Outcome Prediction
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Student ML Systems API",
    description="Production ML API for UK university analytics",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """API info."""
    return {
        "name": "Student ML Systems API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}

@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "models": [
            {"name": "course_recommender", "description": "Course recommendation engine"},
            {"name": "enrollment_yield", "description": "Enrollment prediction"},
            {"name": "early_warning", "description": "Student dropout risk detection"},
            {"name": "degree_outcome", "description": "Degree classification prediction"}
        ]
    }

@app.get("/models/{model_name}")
async def get_model(model_name: str):
    """Get model info."""
    valid_models = ["course_recommender", "enrollment_yield", "early_warning", "degree_outcome"]
    if model_name not in valid_models:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"name": model_name, "status": "loaded", "version": "1.0.0"}

# Include routers
from .routes import recommender, enrollment, early_warning, degree

app.include_router(recommender.router, prefix="/recommend", tags=["Course Recommender"])
app.include_router(enrollment.router, prefix="/enrollment", tags=["Enrollment Yield"])
app.include_router(early_warning.router, prefix="/early-warning", tags=["Early Warning"])
app.include_router(degree.router, prefix="/degree", tags=["Degree Outcome"])
