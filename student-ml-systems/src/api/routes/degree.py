"""
Degree outcome routes.
"""

from fastapi import APIRouter, HTTPException
from ..models import (
    DegreeOutcomeRequest, 
    DegreeOutcomeResponse, 
    DegreeClass,
    BatchDegreeRequest
)
from ..predictors import get_predictor

router = APIRouter()

@router.post("/", response_model=DegreeOutcomeResponse)
async def predict_degree(request: DegreeOutcomeRequest):
    """Predict degree classification."""
    try:
        predictor = get_predictor("degree_outcome")
        result = predictor.predict({"student_id": request.student_id})
        
        # Map string to enum
        class_map = {
            "Fail": DegreeClass.FAIL,
            "Third": DegreeClass.THIRD,
            "2:2": DegreeClass.TWO_TWO,
            "2:1": DegreeClass.TWO_ONE,
            "First": DegreeClass.FIRST
        }
        
        return DegreeOutcomeResponse(
            student_id=result["student_id"],
            predicted_class=class_map.get(result["predicted_class"], DegreeClass.TWO_TWO),
            predicted_class_ordinal=result["predicted_class_ordinal"],
            probabilities={DegreeClass(k): v for k, v in result["probabilities"].items()},
            confidence=result.get("confidence", "medium")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
