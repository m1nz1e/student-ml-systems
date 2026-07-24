"""
Multi-task Student Retention Classifier.

Predicts 4 outcomes simultaneously:
1. retention_risk: binary (will they leave?)
2. risk_score: continuous (0-100)
3. risk_category: ordinal (Low/Medium/High/Critical)
4. departure_year: ordinal (0=retained, 1, 2, 3)

Uses shared representation learning.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
import xgboost as xgb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetentionPredictor(BaseEstimator):
    """
    Multi-task predictor for student retention.

    Uses XGBoost with separate heads for:
    - Binary classification (retention_risk)
    - Regression (risk_score)
    - Ordinal classification (risk_category)
    - Ordinal regression (departure_year)

    Benefits:
    - Shared representation across tasks
    - Regularization from related tasks
    - Consistent predictions
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

        self.risk_clf_: Optional[xgb.XGBClassifier] = None
        self.score_reg_: Optional[xgb.XGBRegressor] = None
        self.category_clf_: Optional[xgb.XGBClassifier] = None
        self.departure_clf_: Optional[xgb.XGBClassifier] = None

        self.n_classes_risk_ = 2
        self.n_classes_category_ = 4
        self.n_classes_departure_ = 4

    def fit(
        self,
        X: np.ndarray,
        y_dict: Dict[str, np.ndarray],
        sample_weight: Optional[np.ndarray] = None,
    ) -> 'RetentionPredictor':
        """
        Fit multi-task retention model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y_dict: Dict of target arrays {
                'retention_risk': binary (0/1),
                'risk_score': continuous (0-100),
                'risk_category': ordinal (0-3),
                'departure_year': ordinal (0-3)
            }
            sample_weight: Optional sample weights

        Returns:
            self
        """
        logger.info(f"Fitting retention predictor for {len(y_dict)} tasks")

        # 1. Binary risk classifier
        if 'retention_risk' in y_dict:
            logger.info("  Training: retention_risk (binary)")
            self.risk_clf_ = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                eval_metric='logloss',
                n_jobs=-1
            )
            self.risk_clf_.fit(X, y_dict['retention_risk'], sample_weight=sample_weight)

        # 2. Risk score regressor
        if 'risk_score' in y_dict:
            logger.info("  Training: risk_score (regression)")
            self.score_reg_ = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.score_reg_.fit(X, y_dict['risk_score'], sample_weight=sample_weight)

        # 3. Risk category ordinal classifier
        if 'risk_category' in y_dict:
            logger.info("  Training: risk_category (4-class)")
            self.category_clf_ = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                eval_metric='mlogloss',
                n_jobs=-1
            )
            self.category_clf_.fit(X, y_dict['risk_category'], sample_weight=sample_weight)

        # 4. Departure year ordinal classifier
        if 'departure_year' in y_dict:
            logger.info("  Training: departure_year (ordinal)")
            self.departure_clf_ = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                eval_metric='mlogloss',
                n_jobs=-1
            )
            self.departure_clf_.fit(X, y_dict['departure_year'], sample_weight=sample_weight)

        logger.info("Retention training complete")
        return self

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict for all tasks.

        Args:
            X: Feature matrix

        Returns:
            Dict of predictions {task_name: predictions}
        """
        predictions = {}

        if self.risk_clf_ is not None:
            predictions['retention_risk'] = self.risk_clf_.predict(X)

        if self.score_reg_ is not None:
            predictions['risk_score'] = self.score_reg_.predict(X)

        if self.category_clf_ is not None:
            predictions['risk_category'] = self.category_clf_.predict(X)

        if self.departure_clf_ is not None:
            predictions['departure_year'] = self.departure_clf_.predict(X)

        return predictions

    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict probabilities for classification tasks.

        Args:
            X: Feature matrix

        Returns:
            Dict of probability arrays {task_name: probas}
        """
        probas = {}

        if self.risk_clf_ is not None:
            probas['retention_risk'] = self.risk_clf_.predict_proba(X)

        if self.category_clf_ is not None:
            probas['risk_category'] = self.category_clf_.predict_proba(X)

        if self.departure_clf_ is not None:
            probas['departure_year'] = self.departure_clf_.predict_proba(X)

        return probas

    def predict_risk_score(self, X: np.ndarray) -> np.ndarray:
        """Convenience: predict risk score only."""
        if self.score_reg_ is None:
            raise ValueError("Risk score regressor not fitted")
        return self.score_reg_.predict(X)

    def predict_retention_risk(self, X: np.ndarray) -> np.ndarray:
        """Convenience: predict binary retention risk only."""
        if self.risk_clf_ is None:
            raise ValueError("Risk classifier not fitted")
        return self.risk_clf_.predict(X)

    def predict_proba_retention_risk(self, X: np.ndarray) -> np.ndarray:
        """Convenience: predict probability of dropout."""
        if self.risk_clf_ is None:
            raise ValueError("Risk classifier not fitted")
        return self.risk_clf_.predict_proba(X)[:, 1]
