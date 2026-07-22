"""
Early warning routes.
"""

from fastapi import APIRouter, HTTPException
from ..models import (
    EarlyWarningRequest, 
    EarlyWarningResponse, 
    RiskLevel, 
    RiskFactor,
    BatchEarlyWarningRequest
)
from ..predictors import get_predictor

router = APIRouter()

@router.post("/", response_model=EarlyWarningResponse)
async def predict_risk(request: EarlyWarningRequest):
    """Predict student dropout risk."""
    try:
        predictor = get_predictor("early_warning")
        result = predictor.predict({"student_id": request.student_id})
        
        risk_level_map = {
            "low": RiskLevel.LOW,
            "medium": RiskLevel.MEDIUM,
            "high": RiskLevel.HIGH,
            "critical": RiskLevel.CRITICAL
        }
        
        return EarlyWarningResponse(
            student_id=result["student_id"],
            risk_score=result["risk_score"],
            risk_level=risk_level_map.get(result["risk_level"], RiskLevel.LOW),
            weeks_remaining=result.get("weeks_remaining"),
            factors=[RiskFactor(**f) for f in result.get("factors", [])],
            intervention_recommended=result.get("intervention_recommended", False)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/risks")
async def get_at_risk_students(threshold: float = 0.5):
    """Get all at-risk students above threshold."""
    # TODO: Implement with actual model
    return {"students": [], "threshold": threshold}
