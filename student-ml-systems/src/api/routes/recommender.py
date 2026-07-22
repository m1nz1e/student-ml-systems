"""
Course recommender routes.
"""

from fastapi import APIRouter, HTTPException
from ..models import (
    RecommendationRequest,
    RecommendationResponse,
    CourseRecommendation,
    BatchRecommendationRequest
)
from ..predictors import get_predictor

router = APIRouter()

@router.post("/", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """Get course recommendations for a student."""
    try:
        predictor = get_predictor("course_recommender")
        result = predictor.predict({"student_id": request.student_id})
        
        return RecommendationResponse(
            student_id=result["student_id"],
            recommendations=[
                CourseRecommendation(**r) for r in result["recommendations"][:request.n_recommendations]
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def recommend_batch(request: BatchRecommendationRequest):
    """Batch course recommendations."""
    predictor = get_predictor("course_recommender")
    results = []
    for student_id in request.student_ids:
        try:
            result = predictor.predict({"student_id": student_id})
            results.append({"student_id": student_id, "recommendations": result["recommendations"]})
        except Exception as e:
            results.append({"student_id": student_id, "error": str(e)})
    return {"results": results}
