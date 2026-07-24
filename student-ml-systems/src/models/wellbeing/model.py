"""Student Wellbeing Multi-task Model."""
from typing import Dict, Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)


class WellbeingPredictor(BaseEstimator, ClassifierMixin, RegressorMixin):
    """
    Multi-task model for student wellbeing prediction.
    
    Predicts:
    - Wellbeing score (regression)
    - At-risk flag (binary classification)
    - Risk level (ordinal classification)
    - Support need (ordinal classification)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.score_regressor = None
        self.risk_classifier = None
        self.support_classifier = None
    
    def fit(
        self,
        X: np.ndarray,
        y_dict: Dict[str, np.ndarray],
        sample_weight: Optional[np.ndarray] = None,
    ) -> 'WellbeingPredictor':
        logger.info("Fitting wellbeing predictor")
        
        # Wellbeing score regressor
        logger.info("  Training wellbeing score regressor")
        self.score_regressor = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.score_regressor.fit(X, y_dict['wellbeing_score'], sample_weight=sample_weight)
        
        # At-risk binary classifier
        logger.info("  Training at-risk classifier")
        self.risk_classifier = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='auc',
            n_jobs=-1
        )
        self.risk_classifier.fit(X, y_dict['at_risk'], sample_weight=sample_weight)
        
        # Support need classifier (4 classes: 0,1,2,3)
        logger.info("  Training support need classifier")
        self.support_classifier = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=-1
        )
        self.support_classifier.fit(X, y_dict['support_need'], sample_weight=sample_weight)
        
        logger.info("Wellbeing training complete")
        return self
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            'wellbeing_score': self.score_regressor.predict(X),
            'at_risk': self.risk_classifier.predict(X),
            'support_need': self.support_classifier.predict(X)
        }
    
    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            'risk_proba': self.risk_classifier.predict_proba(X),
            'support_proba': self.support_classifier.predict_proba(X)
        }
