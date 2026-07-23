"""
NSS Prediction.

Predicts student satisfaction across 7 themes:
- Teaching, Assessment, Feedback, Support
- Organisation, Learning Resources, Student Voice
"""

from .data_prep import NSSFeatureEngineer
from .multi_task_model import NSSMultiTaskModel
from .metrics import evaluate_nss_predictions

__all__ = [
    'NSSFeatureEngineer',
    'NSSMultiTaskModel',
    'evaluate_nss_predictions',
]
