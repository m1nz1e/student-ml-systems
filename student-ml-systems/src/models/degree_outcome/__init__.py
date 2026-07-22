"""Degree Outcome Prediction module.

Ordinal classification of degree outcomes:
0=Fail, 1=Third, 2=2:2, 3=2:1, 4=First
"""

from src.models.degree_outcome.data_prep import (
    DegreeOutcomeFeatureEngineer,
    DEGREE_CLASSIFICATION_MAP,
    DEGREE_CLASS_NAMES,
)
from src.models.degree_outcome.ordinal_classifier import DegreeOutcomeClassifier
from src.models.degree_outcome.metrics import evaluate_degree_outcome
from src.models.degree_outcome.fairness import DegreeOutcomeFairnessAuditor

__all__ = [
    "DegreeOutcomeFeatureEngineer",
    "DegreeOutcomeClassifier",
    "evaluate_degree_outcome",
    "DegreeOutcomeFairnessAuditor",
    "DEGREE_CLASSIFICATION_MAP",
    "DEGREE_CLASS_NAMES",
]
