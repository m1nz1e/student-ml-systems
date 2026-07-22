"""
Ordinal Classifier for Degree Outcome Prediction.

Uses threshold-based ordinal approach:
1. Train binary classifiers for each threshold
2. Combine with logical constraints
3. Enforce monotonicity
4. Calibrate probabilities

Classes: 0=Fail, 1=Third, 2=2:2, 3=2:1, 4=First
"""

from typing import Optional, List, Dict, Any
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DegreeOutcomeClassifier:
    """
    Ordinal classifier for degree outcomes.

    Uses threshold-based approach:
    - Train K-1 binary classifiers (one per threshold)
    - P(Y<=k) = P(Y<=k-1) * P(binary_k=1)
    - Enforce monotonicity (P(Y<=k) >= P(Y<=k+1))
    - Calibrate probabilities

    Classes: 0=Fail, 1=Third, 2=2:2, 3=2:1, 4=First
    """

    def __init__(
        self,
        base_model: str = "xgboost",
        n_thresholds: int = 4,
        calibration_method: str = "isotonic",
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        """
        Initialize ordinal classifier.

        Args:
            base_model: Base model type ('xgboost' or 'sklearn')
            n_thresholds: Number of thresholds (K-1 for K classes)
            calibration_method: Calibration method ('isotonic', 'platt', or None)
            max_depth: Max tree depth for base learners
            learning_rate: Learning rate
            n_estimators: Number of boosting rounds
            random_state: Random seed
        """
        self.base_model = base_model
        self.n_thresholds = n_thresholds
        self.calibration_method = calibration_method
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.random_state = random_state

        self.binary_classifiers: List[Any] = []
        self.calibrators: List[Any] = []
        self.n_classes: int = n_thresholds + 1
        self.is_fitted: bool = False

    def _create_binary_classifier(self) -> Any:
        """Create a new binary classifier instance."""
        if self.base_model == "xgboost":
            try:
                import xgboost as xgb
                return xgb.XGBClassifier(
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    n_estimators=self.n_estimators,
                    use_label_encoder=False,
                    eval_metric="logloss",
                    random_state=self.random_state,
                    verbosity=0,
                )
            except ImportError:
                logger.warning("xgboost not available, falling back to sklearn")
                self.base_model = "sklearn"

        # Fallback to sklearn
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )

    def _create_calibrator(self) -> Any:
        """Create a calibrator wrapper."""
        if self.calibration_method is None:
            return None

        from sklearn.calibration import CalibratedClassifierCV

        base_clf = self._create_binary_classifier()

        if self.calibration_method == "isotonic":
            return CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
        elif self.calibration_method == "platt":
            return CalibratedClassifierCV(base_clf, method="sigmoid", cv=3)
        else:
            return None

    def _create_binary_labels(self, y: np.ndarray, threshold: int) -> np.ndarray:
        """
        Create binary labels for a specific threshold.

        Args:
            y: Ordinal labels (0 to K-1)
            threshold: Threshold index (0 to K-2)

        Returns:
            Binary labels where 1 = y <= threshold
        """
        return (y <= threshold).astype(int)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_calib: Optional[np.ndarray] = None,
        y_calib: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        """
        Fit ordinal classifier.

        Trains K-1 binary classifiers, one per threshold.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,) - values 0 to K
            X_calib: Optional calibration features
            y_calib: Optional calibration labels
            verbose: Print training progress
        """
        logger.info(f"Fitting ordinal classifier with {self.n_thresholds} thresholds...")

        y_train = np.asarray(y_train).flatten()
        self.n_classes = len(np.unique(y_train))
        self.n_thresholds = self.n_classes - 1

        if verbose:
            logger.info(f"Number of classes: {self.n_classes}")
            logger.info(f"Class distribution: {np.bincount(y_train, minlength=self.n_classes)}")

        # Create calibration split if not provided
        if X_calib is None and self.calibration_method is not None:
            from sklearn.model_selection import train_test_split

            X_train, X_calib, y_train, y_calib = train_test_split(
                X_train, y_train,
                test_size=0.2,
                random_state=self.random_state,
                stratify=y_train,
            )
            if verbose:
                logger.info(f"Split off calibration set: {len(X_calib)} samples")

        self.binary_classifiers = []
        self.calibrators = []

        # Train binary classifier for each threshold
        for k in range(self.n_thresholds):
            if verbose:
                logger.info(f"Training classifier for threshold {k} (P(Y <= {k}))...")

            # Create binary labels: 1 if y <= k, 0 otherwise
            y_binary = self._create_binary_labels(y_train, k)

            if verbose:
                pos_rate = y_binary.mean()
                logger.info(f"  Positive class rate: {pos_rate:.3f}")

            # Train binary classifier
            clf = self._create_binary_classifier()
            clf.fit(X_train, y_binary)

            self.binary_classifiers.append(clf)

            # Calibrate if we have calibration data
            if self.calibration_method is not None and X_calib is not None and y_calib is not None:
                y_binary_calib = self._create_binary_labels(y_calib, k)
                calibrator = self._create_calibrator()
                if calibrator is not None:
                    calibrator.fit(X_calib, y_binary_calib)
                    self.calibrators.append(calibrator)
                else:
                    self.calibrators.append(None)
            else:
                self.calibrators.append(None)

        self.is_fitted = True
        logger.info("Ordinal classifier training complete")
        return self

    def _predict_cumulative_prob_raw(self, X: np.ndarray) -> np.ndarray:
        """
        Predict raw cumulative probabilities P(Y <= k) for each threshold.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Cumulative probabilities (n_samples, n_thresholds)
        """
        n_samples = X.shape[0]
        cumulative_probs = np.zeros((n_samples, self.n_thresholds))

        prev_cumulative = np.ones(n_samples)  # P(Y <= -1) = 1

        for k in range(self.n_thresholds):
            # Get probability from binary classifier
            if self.calibrators[k] is not None:
                # Use calibrated classifier
                prob_k = self.calibrators[k].predict_proba(X)[:, 1]
            else:
                prob_k = self.binary_classifiers[k].predict_proba(X)[:, 1]

            # P(Y <= k) = P(Y <= k-1) * P(Y <= k | Y <= k-1)
            cumulative_probs[:, k] = prev_cumulative * prob_k
            prev_cumulative = cumulative_probs[:, k]

        return cumulative_probs

    def predict_cumulative_prob(self, X: np.ndarray) -> np.ndarray:
        """
        Predict P(Y <= k) for each threshold k.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Cumulative probabilities (n_samples, n_thresholds)
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before predict_cumulative_prob()")

        cumulative_probs = self._predict_cumulative_prob_raw(X)

        # Enforce monotonicity: P(Y <= k) >= P(Y <= k+1)
        for k in range(1, self.n_thresholds):
            cumulative_probs[:, k] = np.minimum(
                cumulative_probs[:, k],
                cumulative_probs[:, k - 1]
            )

        return cumulative_probs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before predict_proba()")

        cumulative_probs = self.predict_cumulative_prob(X)
        n_samples = X.shape[0]

        # Convert cumulative probabilities to class probabilities
        # P(Y = k) = P(Y <= k) - P(Y <= k-1)
        probs = np.zeros((n_samples, self.n_classes))

        probs[:, 0] = cumulative_probs[:, 0]  # P(Y = 0) = P(Y <= 0)
        for k in range(1, self.n_thresholds):
            probs[:, k] = cumulative_probs[:, k] - cumulative_probs[:, k - 1]
        probs[:, self.n_thresholds] = 1 - cumulative_probs[:, self.n_thresholds - 1]  # P(Y = K-1)

        # Ensure probabilities sum to 1 (clip and renormalize)
        probs = np.clip(probs, 0, 1)
        probs = probs / probs.sum(axis=1, keepdims=True)

        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict ordinal class labels.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted labels (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before predict()")

        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
