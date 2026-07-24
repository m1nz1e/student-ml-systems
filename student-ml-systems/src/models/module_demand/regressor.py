"""Module Demand Regressor — XGBoost for enrollment count and fill rate."""
from typing import Dict, Optional
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)

class ModuleDemandRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators: int = 100, max_depth: int = 4, learning_rate: float = 0.1, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.enrollment_model = None
        self.fill_rate_model = None
    
    def fit(self, X: np.ndarray, y_dict: Dict[str, np.ndarray], sample_weight: Optional[np.ndarray] = None) -> 'ModuleDemandRegressor':
        logger.info("Fitting module demand regressor")
        self.enrollment_model = xgb.XGBRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, random_state=self.random_state, n_jobs=-1
        )
        self.enrollment_model.fit(X, y_dict['enrollment_count'], sample_weight=sample_weight)
        self.fill_rate_model = xgb.XGBRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, random_state=self.random_state, n_jobs=-1
        )
        self.fill_rate_model.fit(X, y_dict['fill_rate'], sample_weight=sample_weight)
        return self
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        return {'enrollment_count': self.enrollment_model.predict(X), 'fill_rate': self.fill_rate_model.predict(X)}
