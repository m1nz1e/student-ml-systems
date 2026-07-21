"""Early Warning System module."""

from .data_prep import EarlyWarningFeatureEngineer
from .lstm import LSTMEarlyWarning, generate_synthetic_data

try:
    from .survival import SurvivalAnalyzer
except ImportError:
    SurvivalAnalyzer = None

try:
    from .risk_scorer import RiskScorer
except ImportError:
    RiskScorer = None

__all__ = [
    "EarlyWarningFeatureEngineer",
    "LSTMEarlyWarning",
    "SurvivalAnalyzer",
    "RiskScorer",
    "generate_synthetic_data",
]
