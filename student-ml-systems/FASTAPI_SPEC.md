# FastAPI Model Serving — Phase 1 Specification

## Objective
Build production REST API endpoints for all 4 ML models:
1. Course Recommender
2. Enrollment Yield
3. Early Warning
4. Degree Outcome

## Files to Create
Location: `/home/m1nz/.openclaw/workspace/student-ml-systems/src/api/`

### Required Files
1. `main.py` — FastAPI app with all routes
2. `models.py` — Pydantic request/response schemas
3. `predictors.py` — Model wrapper classes
4. `dependencies.py` — Dependency injection (model loading, caching)
5. `__init__.py` — Exports

## API Endpoints

### Health & Info
```
GET /health                    # Health check
GET /                          # API info
GET /models                    # List available models
GET /models/{name}            # Model info
```

### Course Recommender
```
POST /recommend                # Get course recommendations
POST /recommend/batch          # Batch recommendations
```

### Enrollment Yield
```
POST /enrollment/predict      # Predict enrollment probability
POST /enrollment/batch        # Batch predictions
```

### Early Warning
```
POST /early-warning/predict    # Predict dropout risk
POST /early-warning/batch     # Batch predictions
GET /early-warning/risks       # Get all at-risk students
```

### Degree Outcome
```
POST /degree/predict          # Predict degree classification
POST /degree/batch            # Batch predictions
```

## Request/Response Format

### Course Recommender
```python
# Request
{
    "student_id": "SPR12345",
    "n_recommendations": 5,
    "exclude_enrolled": true
}

# Response
{
    "student_id": "SPR12345",
    "recommendations": [
        {"course_id": "CRS001", "course_name": "Computer Science", "score": 0.95},
        {"course_id": "CRS002", "course_name": "Data Science", "score": 0.88}
    ],
    "model_version": "1.0.0"
}
```

### Enrollment Yield
```python
# Request
{
    "student_id": "SPR12345",
    "course_id": "CRS001"
}

# Response
{
    "student_id": "SPR12345",
    "course_id": "CRS001",
    "enrollment_probability": 0.82,
    "risk_level": "low",
    "recommendations": ["High likelihood of enrollment"],
    "model_version": "1.0.0"
}
```

### Early Warning
```python
# Request
{
    "student_id": "SPR12345"
}

# Response
{
    "student_id": "SPR12345",
    "risk_score": 0.75,
    "risk_level": "high",
    "weeks_remaining": 6,
    "factors": [
        {"factor": "attendance_decline", "impact": 0.3},
        {"factor": "low_engagement", "impact": 0.2}
    ],
    "intervention_recommended": true,
    "model_version": "1.0.0"
}
```

### Degree Outcome
```python
# Request
{
    "student_id": "SPR12345"
}

# Response
{
    "student_id": "SPR12345",
    "predicted_class": "2:1",
    "predicted_class_ordinal": 3,
    "probabilities": {
        "Fail": 0.02,
        "Third": 0.08,
        "2:2": 0.25,
        "2:1": 0.55,
        "First": 0.10
    },
    "confidence": "high",
    "model_version": "1.0.0"
}
```

## Features Required

### Model Loading & Caching
- Lazy loading on first request
- Singleton pattern for loaded models
- Version tracking

### Batch Processing
- Accept list of students
- Parallel processing
- Return results with IDs

### Error Handling
- Proper HTTP status codes
- Validation errors (422)
- Model not loaded (503)
- Invalid input (400)

### Logging & Monitoring
- Request/response logging
- Latency tracking
- Error tracking

### Documentation
- Auto-generated OpenAPI/Swagger UI
- Request/response examples

## FastAPI App Structure

```python
# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import *

app = FastAPI(
    title="Student ML Systems API",
    description="Production ML API for UK university analytics",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(recommender.router, prefix="/recommend", tags=["Course Recommender"])
app.include_router(enrollment.router, prefix="/enrollment", tags=["Enrollment Yield"])
app.include_router(early_warning.router, prefix="/early-warning", tags=["Early Warning"])
app.include_router(degree.router, prefix="/degree", tags=["Degree Outcome"])

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

## Dependencies
```python
# requirements addition
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pydantic>=2.0.0
```

## Testing Requirements
- Unit tests for each endpoint
- Mock model predictions
- Validation tests

## Success Criteria
- [ ] All 4 models have REST endpoints
- [ ] Batch processing works
- [ ] Error handling returns proper codes
- [ ] OpenAPI docs accessible at `/docs`
- [ ] Health check works
- [ ] Unit tests pass

## Delegation Tasks

### Task 1: Core FastAPI Setup (Backend Dev)
1. Create `src/api/` directory structure
2. Create `main.py` with FastAPI app
3. Add CORS middleware
4. Add health check
5. Add base routes

### Task 2: Pydantic Models (Backend Dev)
1. Create `models.py` with all request/response schemas
2. Include validation
3. Add examples

### Task 3: Predictor Wrappers (ML Engineer)
1. Create `predictors.py` with model wrapper classes
2. Lazy loading
3. Singleton pattern

### Task 4: API Routes (Backend Dev)
1. Create route handlers for each model
2. Error handling
3. Logging

### Task 5: Testing (QA)
1. Unit tests for endpoints
2. Mock model tests
