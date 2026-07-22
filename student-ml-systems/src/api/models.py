"""
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DegreeClass(str, Enum):
    FAIL = "Fail"
    THIRD = "Third"
    TWO_TWO = "2:2"
    TWO_ONE = "2:1"
    FIRST = "First"

# === Course Recommender ===

class RecommendationRequest(BaseModel):
    student_id: str
    n_recommendations: int = Field(default=5, ge=1, le=20)
    exclude_enrolled: bool = True

class CourseRecommendation(BaseModel):
    course_id: str
    course_name: str
    score: float
    department: Optional[str] = None

class RecommendationResponse(BaseModel):
    student_id: str
    recommendations: List[CourseRecommendation]
    model_version: str = "1.0.0"

# === Enrollment Yield ===

class EnrollmentRequest(BaseModel):
    student_id: str
    course_id: str

class EnrollmentResponse(BaseModel):
    student_id: str
    course_id: str
    enrollment_probability: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    recommendations: List[str] = []
    model_version: str = "1.0.0"

# === Early Warning ===

class EarlyWarningRequest(BaseModel):
    student_id: str
    threshold: float = Field(default=0.5, ge=0, le=1)

class RiskFactor(BaseModel):
    factor: str
    impact: float

class EarlyWarningResponse(BaseModel):
    student_id: str
    risk_score: float = Field(..., ge=0, le=1)
    risk_level: RiskLevel
    weeks_remaining: Optional[int] = None
    factors: List[RiskFactor] = []
    intervention_recommended: bool
    model_version: str = "1.0.0"

# === Degree Outcome ===

class DegreeOutcomeRequest(BaseModel):
    student_id: str

class DegreeOutcomeResponse(BaseModel):
    student_id: str
    predicted_class: DegreeClass
    predicted_class_ordinal: int
    probabilities: Dict[DegreeClass, float]
    confidence: str  # low, medium, high
    model_version: str = "1.0.0"

# === Batch Requests ===

class BatchRecommendationRequest(BaseModel):
    student_ids: List[str]
    n_recommendations: int = 5

class BatchEnrollmentRequest(BaseModel):
    requests: List[EnrollmentRequest]

class BatchEarlyWarningRequest(BaseModel):
    student_ids: List[str]

class BatchDegreeRequest(BaseModel):
    student_ids: List[str]
