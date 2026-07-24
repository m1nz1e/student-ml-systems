"""Exam Performance Multi-task Model."""
from typing import Dict, Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)


class ExamPredictor(BaseEstimator, ClassifierMixin, RegressorMixin):
    """
    Multi-task model for exam performance prediction.
    
    Predicts:
    - Exam mark (regression)
    - Pass/Fail (binary classification)
    - Grade class (ordinal classification)
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
        self.mark_regressor = None
        self.pass_classifier = None
        self.grade_classifier = None
    
    def fit(
        self,
        X: np.ndarray,
        y_dict: Dict[str, np.ndarray],
        sample_weight: Optional[np.ndarray] = None,
    ) -> 'ExamPredictor':
        logger.info("Fitting exam predictor")
        
        # Exam mark regressor
        logger.info("  Training mark regressor")
        self.mark_regressor = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.mark_regressor.fit(X, y_dict['exam_mark'], sample_weight=sample_weight)
        
        # Pass/Fail classifier
        logger.info("  Training pass/fail classifier")
        self.pass_classifier = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='auc',
            n_jobs=-1
        )
        self.pass_classifier.fit(X, y_dict['pass_fail'], sample_weight=sample_weight)
        
        # Grade class classifier (4 classes)
        logger.info("  Training grade classifier")
        self.grade_classifier = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=-1
        )
        self.grade_classifier.fit(X, y_dict['grade_class'], sample_weight=sample_weight)
        
        logger.info("Exam training complete")
        return self
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            'exam_mark': self.mark_regressor.predict(X),
            'pass_fail': self.pass_classifier.predict(X),
            'grade_class': self.grade_classifier.predict(X)
        }
    
    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            'pass_proba': self.pass_classifier.predict_proba(X),
            'grade_proba': self.grade_classifier.predict_proba(X)
        }
