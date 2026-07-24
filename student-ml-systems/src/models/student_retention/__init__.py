"""
Student Retention Prediction.

Predicts student dropout risk:
- Binary retention risk (will they leave?)
- Risk score (0-100)
- Risk category (Low/Medium/High/Critical)
- Departure year
"""

from .data_prep import RetentionFeatureEngineer
from .multi_task_model import RetentionPredictor
from .metrics import evaluate_retention

__all__ = [
    'RetentionFeatureEngineer',
    'RetentionPredictor',
    'evaluate_retention',
]
