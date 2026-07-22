"""
Enrollment yield routes.
"""

from fastapi import APIRouter, HTTPException
from ..models import EnrollmentRequest, EnrollmentResponse, RiskLevel, BatchEnrollmentRequest
from ..predictors import get_predictor

router = APIRouter()

@router.post("/", response_model=EnrollmentResponse)
async def predict_enrollment(request: EnrollmentRequest):
    """Predict enrollment probability."""
    try:
        predictor = get_predictor("enrollment_yield")
        result = predictor.predict({
            "student_id": request.student_id,
            "course_id": request.course_id
        })
        
        # Map risk level string to enum
        risk_level_map = {
            "low": RiskLevel.LOW,
            "medium": RiskLevel.MEDIUM,
            "high": RiskLevel.HIGH,
            "critical": RiskLevel.CRITICAL
        }
        
        return EnrollmentResponse(
            student_id=result["student_id"],
            course_id=result["course_id"],
            enrollment_probability=result["enrollment_probability"],
            risk_level=risk_level_map.get(result["risk_level"], RiskLevel.LOW),
            recommendations=result.get("recommendations", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def predict_batch(request: BatchEnrollmentRequest):
    """Batch enrollment predictions."""
    predictor = get_predictor("enrollment_yield")
    results = []
    for req in request.requests:
        try:
            result = predictor.predict({
                "student_id": req.student_id,
                "course_id": req.course_id
            })
            results.append({"student_id": req.student_id, "course_id": req.course_id, 
                          "probability": result["enrollment_probability"]})
        except Exception as e:
            results.append({"student_id": req.student_id, "course_id": req.course_id, "error": str(e)})
    return {"results": results}
