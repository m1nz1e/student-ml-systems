"""
Multi-task NSS Prediction Model.

Predicts:
1. Overall satisfaction (binary classification)
2. Theme scores (7 x regression)
3. NPS (ordinal/regression)
"""

from typing import Dict, Optional, Any, List
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NSSMultiTaskModel(BaseEstimator, ClassifierMixin, RegressorMixin):
    """
    Multi-task model for NSS prediction.
    
    Uses XGBoost for both classification (satisfaction) and regression (themes, NPS).
    
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
        
        self.satisfaction_model = None
        self.nps_model = None
        self.theme_models = {}
        
    def fit(
        self,
        X: np.ndarray,
        y_dict: Dict[str, np.ndarray],
        sample_weight: Optional[np.ndarray] = None,
    ) -> 'NSSMultiTaskModel':
        """
        Fit multi-task model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y_dict: Dict of targets {
                'satisfied': binary array (n_samples,),
                'nps': array (n_samples,),
                'themes': array (n_samples, 7)
            }
            sample_weight: Optional sample weights
            
        Returns:
            self
        """
        logger.info("Fitting NSS multi-task model")
        
        # Satisfaction classifier (binary)
        logger.info("  Training satisfaction classifier")
        self.satisfaction_model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='auc',
            n_jobs=-1
        )
        self.satisfaction_model.fit(X, y_dict['satisfied'], sample_weight=sample_weight)
        
        # NPS regressor (ordinal)
        logger.info("  Training NPS regressor")
        self.nps_model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.nps_model.fit(X, y_dict['nps'], sample_weight=sample_weight)
        
        # Theme regressors (7 themes)
        theme_names = [
            'teaching', 'assessment', 'feedback',
            'support', 'organisation', 'learning_resources', 'student_voice'
        ]
        for i, theme in enumerate(theme_names):
            logger.info(f"  Training {theme} regressor")
            self.theme_models[theme] = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.theme_models[theme].fit(X, y_dict['themes'][:, i], sample_weight=sample_weight)
        
        logger.info("NSS training complete")
        return self
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict for all tasks.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dict of predictions {
                'satisfied': binary predictions,
                'nps': NPS predictions,
                'themes': theme predictions (n_samples, 7)
            }
        """
        predictions = {}
        
        predictions['satisfied'] = self.satisfaction_model.predict(X)
        predictions['nps'] = self.nps_model.predict(X)
        
        themes = []
        for theme in ['teaching', 'assessment', 'feedback',
                      'support', 'organisation', 'learning_resources', 'student_voice']:
            themes.append(self.theme_models[theme].predict(X))
        predictions['themes'] = np.column_stack(themes)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict probabilities for satisfaction task.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dict with satisfaction probabilities
        """
        return {
            'satisfied_proba': self.satisfaction_model.predict_proba(X)
        }
    
    def score(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]) -> float:
        """
        Average accuracy across classification tasks.
        """
        from sklearn.metrics import accuracy_score, r2_score
        
        predictions = self.predict(X)
        
        # Satisfaction accuracy
        sat_acc = accuracy_score(y_dict['satisfied'], predictions['satisfied'])
        
        # NPS R²
        nps_r2 = r2_score(y_dict['nps'], predictions['nps'])
        
        # Average R² for themes
        theme_r2s = []
        for i in range(7):
            r2 = r2_score(y_dict['themes'][:, i], predictions['themes'][:, i])
            theme_r2s.append(r2)
        
        avg_theme_r2 = np.mean(theme_r2s)
        
        # Combined score (normalized)
        combined = (sat_acc + max(0, nps_r2) + max(0, avg_theme_r2)) / 3
        return combined
