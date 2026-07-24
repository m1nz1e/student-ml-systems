"""Module Demand Classifier — XGBoost for demand category."""
from typing import Dict, Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)

class ModuleDemandClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators: int = 100, max_depth: int = 4, learning_rate: float = 0.1, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.classes_ = np.array([0, 1, 2, 3])
    
    def fit(self, X: np.ndarray, y_dict: Dict[str, np.ndarray], sample_weight: Optional[np.ndarray] = None) -> 'ModuleDemandClassifier':
        logger.info("Fitting module demand classifier")
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, random_state=self.random_state,
            use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1
        )
        self.model.fit(X, y_dict['demand_category'], sample_weight=sample_weight)
        return self
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        return {'demand_category': self.model.predict(X)}
    
    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        return {'demand_category_proba': self.model.predict_proba(X)}
