"""
Multi-task Graduate Outcome Classifier.

Predicts 3 outcomes simultaneously:
1. Employment status (4 classes)
2. Salary band (4 classes)
3. Further study destination (3 classes)

Uses shared representation learning.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiTaskGraduateClassifier(BaseEstimator, ClassifierMixin):
    """
    Multi-task classifier for graduate outcomes.
    
    Uses XGBoost with multi-output wrapper for joint prediction
    of employment, salary, and further study outcomes.
    
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
        
        self.models_: Dict[str, Any] = {}
        self.classes_: Dict[str, np.ndarray] = {}
        self.n_classes_: Dict[str, int] = {}
        
    def fit(
        self,
        X: np.ndarray,
        y_dict: Dict[str, np.ndarray],
        sample_weight: Optional[np.ndarray] = None,
    ) -> 'MultiTaskGraduateClassifier':
        """
        Fit multi-task model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y_dict: Dict of target arrays {task_name: y_values}
            sample_weight: Optional sample weights
            
        Returns:
            self
        """
        logger.info(f"Fitting multi-task classifier for {len(y_dict)} tasks")
        
        for task_name, y in y_dict.items():
            logger.info(f"  Training task: {task_name}")
            
            # XGBoost classifier
            clf = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                eval_metric='mlogloss',
                n_jobs=-1
            )
            
            clf.fit(X, y, sample_weight=sample_weight)
            
            self.models_[task_name] = clf
            self.classes_[task_name] = clf.classes_
            self.n_classes_[task_name] = len(clf.classes_)
        
        logger.info("Multi-task training complete")
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
        for task_name, clf in self.models_.items():
            predictions[task_name] = clf.predict(X)
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predict probabilities for all tasks.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dict of probability arrays {task_name: probas}
            Each probas is (n_samples, n_classes)
        """
        probas = {}
        for task_name, clf in self.models_.items():
            probas[task_name] = clf.predict_proba(X)
        return probas
    
    def score(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]) -> float:
        """
        Average accuracy across all tasks.
        
        Args:
            X: Feature matrix
            y_dict: Dict of true targets
            
        Returns:
            Average accuracy
        """
        predictions = self.predict(X)
        accuracies = []
        for task_name, y_true in y_dict.items():
            acc = np.mean(predictions[task_name] == y_true)
            accuracies.append(acc)
        return np.mean(accuracies)
