"""
Enrollment Yield Classifier using XGBoost.

Implements:
1. XGBoost binary classifier
2. Probability calibration (Platt scaling, isotonic)
3. Class imbalance handling
4. Hyperparameter tuning with Optuna
5. Model evaluation (ROC-AUC, PR-AUC, calibration)
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseEnrollmentClassifier(ABC):
    """Abstract base class for enrollment classifiers."""

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fit the classifier."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        pass


class XGBoostEnrollmentClassifier(BaseEnrollmentClassifier):
    """
    XGBoost classifier for enrollment yield prediction.

    Features:
    - Handles class imbalance (scale_pos_weight)
    - Probability calibration (Platt, isotonic)
    - Feature importance extraction
    - Hyperparameter tuning with Optuna
    """

    def __init__(
        self,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        n_estimators: int = 200,
        min_child_weight: int = 1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
        calibration_method: Optional[str] = "isotonic",
        calibration_size: float = 0.2,
    ):
        """
        Initialize XGBoost classifier.

        Args:
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            n_estimators: Number of boosting rounds
            min_child_weight: Minimum child weight
            subsample: Subsample ratio
            colsample_bytree: Column subsample ratio
            scale_pos_weight: Class imbalance weight (auto-calculated if None)
            random_state: Random seed
            calibration_method: 'platt', 'isotonic', or None
            calibration_size: Fraction of data for calibration
        """
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.calibration_method = calibration_method
        self.calibration_size = calibration_size

        self.model = None
        self.calibrator = None
        self.feature_names: List[str] = []
        self.calibration_indices: Optional[np.ndarray] = None

    def _calculate_scale_pos_weight(self, y: np.ndarray) -> float:
        """Calculate scale_pos_weight for class imbalance."""
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        return n_neg / n_pos if n_pos > 0 else 1.0

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        X_calib: Optional[np.ndarray] = None,
        y_calib: Optional[np.ndarray] = None,
        verbose: bool = True,
    ):
        """
        Fit XGBoost model with optional calibration.

        Args:
            X_train: Training features
            y_train: Training labels
            feature_names: Feature names for importance
            X_calib: Calibration features (if None, split from train)
            y_calib: Calibration labels
            verbose: Print training progress
        """
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("xgboost not installed. Install with: pip install xgboost")
            raise

        logger.info("Fitting XGBoost enrollment classifier...")

        # Auto-calculate scale_pos_weight if not provided
        if self.scale_pos_weight is None:
            self.scale_pos_weight = self._calculate_scale_pos_weight(y_train)
            logger.info(f"Auto-calculated scale_pos_weight: {self.scale_pos_weight:.2f}")

        # Store feature names
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]

        # Split off calibration set if not provided
        if X_calib is None and self.calibration_method is not None:
            from sklearn.model_selection import train_test_split

            X_train, X_calib, y_train, y_calib = train_test_split(
                X_train, y_train, test_size=self.calibration_size, random_state=self.random_state, stratify=y_train
            )
            logger.info(f"Split calibration set: {len(X_calib)} samples")

        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)

        # Set parameters
        params = {
            "objective": "binary:logistic",
            "eval_metric": ["auc", "logloss"],
            "max_depth": self.max_depth,
            "eta": self.learning_rate,
            "n_estimators": self.n_estimators,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "scale_pos_weight": self.scale_pos_weight,
            "seed": self.random_state,
            "tree_method": "hist",  # Fast histogram-based algorithm
        }

        # Train model
        logger.info(f"Training XGBoost with {self.n_estimators} rounds...")
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=[(dtrain, "train")],
            verbose_eval=10 if verbose else False,
        )

        logger.info("XGBoost training complete")

        # Calibrate probabilities
        if self.calibration_method is not None and X_calib is not None and y_calib is not None:
            logger.info(f"Calibrating probabilities ({self.calibration_method})...")
            self._calibrate(X_calib, y_calib)

        return self

    def _calibrate(self, X_calib: np.ndarray, y_calib: np.ndarray):
        """Calibrate probability outputs."""
        from sklearn.calibration import CalibratedClassifierCV
        import xgboost as xgb

        # Wrap XGBoost in sklearn interface for calibration
        xgb_sklearn = xgb.XGBClassifier(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
        )

        # Copy trained model parameters
        xgb_sklearn._Booster = self.model

        # Calibrate
        # Note: cv="prefit" was removed in sklearn 1.3+, use cv=int or cv=KFold
        if self.calibration_method == "platt":
            self.calibrator = CalibratedClassifierCV(
                xgb_sklearn, method="sigmoid", cv=3
            )
        elif self.calibration_method == "isotonic":
            self.calibrator = CalibratedClassifierCV(
                xgb_sklearn, method="isotonic", cv=3
            )

        self.calibrator.fit(X_calib, y_calib)
        logger.info("Probability calibration complete")

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix
            threshold: Classification threshold

        Returns:
            Binary predictions
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Probability array (n_samples, 2)
        """
        import xgboost as xgb

        if self.model is None:
            raise ValueError("Must call fit() before predict_proba()")

        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        proba_positive = self.model.predict(dmatrix)

        # Apply calibration if available
        if self.calibrator is not None:
            proba_positive = self.calibrator.predict_proba(X)[:, 1]

        # Return both class probabilities
        return np.column_stack([1 - proba_positive, proba_positive])

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance scores.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Must call fit() before getting feature importance")

        importance_dict = self.model.get_score(importance_type="gain")
        importance_df = pd.DataFrame(
            list(importance_dict.items()),
            columns=["feature", "gain"]
        )
        importance_df = importance_df.sort_values("gain", ascending=False)

        return importance_df.head(top_n)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        from sklearn.metrics import (
            roc_auc_score,
            average_precision_score,
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            brier_score_loss,
            calibration_curve,
        )

        logger.info("Evaluating model performance...")

        # Predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "pr_auc": average_precision_score(y_test, y_pred_proba),
            "brier_score": brier_score_loss(y_test, y_pred_proba),
        }

        # Calibration error
        prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
        metrics["calibration_error"] = np.mean(np.abs(prob_true - prob_pred))

        # Log metrics
        logger.info("\n" + "=" * 60)
        logger.info("MODEL EVALUATION METRICS")
        logger.info("=" * 60)
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        logger.info("=" * 60)

        return metrics


class EnrollmentYieldTuner:
    """
    Hyperparameter tuner for enrollment yield classifier using Optuna.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 50,
        timeout: int = 3600,
        random_state: int = 42,
    ):
        """
        Initialize tuner.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            random_state: Random seed
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state
        self.best_params = None
        self.best_score = 0.0

    def objective(self, trial):
        """Optuna objective function."""
        import xgboost as xgb
        from sklearn.metrics import average_precision_score

        # Hyperparameter search space
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
        }

        # Train model
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dval = xgb.DMatrix(self.X_val, label=self.y_val)

        xgb_params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",  # PR-AUC for imbalanced data
            **params,
            "seed": self.random_state,
            "tree_method": "hist",
        }

        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=params["n_estimators"],
            evals=[(dval, "val")],
            verbose_eval=False,
        )

        # Predict
        y_pred_proba = model.predict(dval)

        # PR-AUC (better for imbalanced data)
        score = average_precision_score(self.y_val, y_pred_proba)

        return score

    def optimize(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            verbose: Print progress

        Returns:
            Best parameters dictionary
        """
        try:
            import optuna
        except ImportError:
            logger.error("optuna not installed. Install with: pip install optuna")
            raise

        logger.info(f"Starting hyperparameter optimization ({self.n_trials} trials)...")

        # Create study
        study = optuna.create_study(
            study_name="enrollment_yield_tuning",
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )

        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=verbose,
        )

        self.best_params = study.best_params
        self.best_score = study.best_value

        logger.info(f"\nOptimization complete!")
        logger.info(f"Best PR-AUC: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

        return self.best_params


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    tune_hyperparameters: bool = False,
) -> Tuple[XGBoostEnrollmentClassifier, Dict[str, float]]:
    """
    Train and evaluate enrollment yield classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        feature_names: Feature names
        tune_hyperparameters: Whether to run hyperparameter tuning

    Returns:
        Tuple of (trained_model, evaluation_metrics)
    """
    logger.info("Training enrollment yield classifier...")

    # Hyperparameter tuning
    if tune_hyperparameters:
        logger.info("Running hyperparameter tuning...")
        tuner = EnrollmentYieldTuner(X_train, y_train, X_test, y_test, n_trials=30)
        best_params = tuner.optimize()

        # Create model with best parameters
        model = XGBoostEnrollmentClassifier(
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            n_estimators=best_params["n_estimators"],
            min_child_weight=best_params["min_child_weight"],
            subsample=best_params["subsample"],
            colsample_bytree=best_params["colsample_bytree"],
            scale_pos_weight=best_params["scale_pos_weight"],
            calibration_method="isotonic",
        )
    else:
        # Default parameters
        model = XGBoostEnrollmentClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1.0,
            calibration_method="isotonic",
        )

    # Train model
    model.fit(X_train, y_train, feature_names=feature_names, verbose=True)

    # Evaluate
    metrics = model.evaluate(X_test, y_test)

    # Feature importance
    importance_df = model.get_feature_importance(top_n=10)
    logger.info("\nTop 10 Feature Importance:")
    for _, row in importance_df.iterrows():
        logger.info(f"  {row['feature']}: {row['gain']:.4f}")

    return model, metrics


# Example usage
if __name__ == "__main__":
    from src.data.synthetic import SITSSyntheticGenerator
    from src.models.enrollment.data_prep import EnrollmentYieldFeatureEngineer

    # Generate synthetic data
    print("Generating synthetic data...")
    generator = SITSSyntheticGenerator(n_students=2000, n_courses=50, seed=42)
    datasets = generator.generate_all_datasets()

    # Engineer features
    print("\nEngineering features...")
    engineer = EnrollmentYieldFeatureEngineer(target_col="accepted_offer", test_size=0.2)
    df, X, y = engineer.engineer_features(
        students_df=datasets["students"],
        qualifications_df=datasets["qualifications"],
        courses_df=datasets["courses"],
        enrollments_df=datasets["enrollments"],
    )

    # Train/test split
    X_train, X_test, y_train, y_test = engineer.create_train_test_split(X, y, stratified=True)

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    print(f"Positive rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")

    # Train and evaluate
    print("\n" + "=" * 60)
    print("TRAINING ENROLLMENT YIELD CLASSIFIER")
    print("=" * 60)

    model, metrics = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        feature_names=engineer.feature_names,
        tune_hyperparameters=False,  # Set True for tuning
    )

    print("\n✓ Training complete!")
    print(f"\nFinal Metrics:")
    print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  Calibration Error: {metrics['calibration_error']:.4f}")
