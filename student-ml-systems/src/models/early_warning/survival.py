"""
Survival Analysis models for time-to-event prediction in student dropout.

Supports:
    - Cox Proportional Hazards (CoxPH)
    - Random Survival Forests (RSF)
    - C-index evaluation
    - Survival function prediction
    - Feature importance extraction
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Attempt to import scikit-survival
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv

    SKSURV_AVAILABLE = True
except ImportError:
    SKSURV_AVAILABLE = False
    logger.warning("scikit-survival not installed. Run: pip install scikit-survival")


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

# y_surv format: structured array with dtype [('event', '?'), ('time', 'f8')]
#   event: True if the event (dropout) occurred, False if censored
#   time:  time to event or censoring


def make_surv_array(event: np.ndarray, time: np.ndarray) -> Any:
    """
    Create a structured survival array for scikit-survival.

    Args:
        event: Boolean array (True=event occurred)
        time:  Array of times

    Returns:
        Structured array with dtype [('event', '?'), ('time', 'f8')]
    """
    if not SKSURV_AVAILABLE:
        raise ImportError("scikit-survival is required. Install with: pip install scikit-survival")
    return Surv.from_arrays(event=event, time=time)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────


class SurvivalAnalyzer:
    """
    Wrapper for survival analysis models (CoxPH or RSF).

    Supports time-to-event prediction for student dropout with
    risk scoring, survival function estimation, and feature importance.
    """

    def __init__(
        self,
        model_type: str = "cox",
        n_estimators: int = 100,
        min_samples_split: int = 6,
        min_samples_leaf: int = 3,
        max_depth: Optional[int] = None,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize the survival analyzer.

        Args:
            model_type: 'cox' for Cox Proportional Hazards, 'rsf' for Random Survival Forest
            n_estimators: Number of trees (for RSF)
            alpha: L1/L2 regularization strength (for Cox)
            min_samples_split: Min samples to split a node (RSF)
            min_samples_leaf: Min samples in leaf (RSF)
            max_depth: Maximum tree depth (RSF)
            random_state: Random seed
            **kwargs: Additional model-specific arguments
        """
        if not SKSURV_AVAILABLE:
            raise ImportError("scikit-survival is required. Install with: pip install scikit-survival")

        self.model_type = model_type.lower()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model: Any = None
        self._feature_names: Optional[List[str]] = None

        if self.model_type == "cox":
            # Cox with elastic-net regularization
            self._model = CoxnetSurvivalAnalysis(
                n_alphas=20,
                alpha_min_ratio="auto",
                l1_ratio=0.5,
                max_iter=100000,
                tol=1e-6,
                verbose=False,
            )
        elif self.model_type == "rsf":
            self._model = RandomSurvivalForest(
                n_estimators=n_estimators,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_depth=max_depth if max_depth else None,
                n_jobs=-1,
                random_state=random_state,
            )
        else:
            raise ValueError(f"model_type must be 'cox' or 'rsf', got '{model_type}'")

        logger.info(f"SurvivalAnalyzer initialized: type={self.model_type}, n_estimators={n_estimators}")

    def fit(
        self,
        X: np.ndarray,
        y_surv: Any,
        feature_names: Optional[List[str]] = None,
    ) -> "SurvivalAnalyzer":
        """
        Fit the survival model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y_surv: Structured survival array from make_surv_array()
            feature_names: Optional list of feature names

        Returns:
            self (for chaining)
        """
        if feature_names is not None:
            self._feature_names = feature_names

        logger.info(f"Fitting {self.model_type.upper()} model on {X.shape[0]} samples, {X.shape[1]} features")
        self._model.fit(X, y_surv)
        logger.info(f"{self.model_type.upper()} model fitted successfully")

        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """
        Predict risk scores (higher = more likely to experience event sooner).

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Array of risk scores (n_samples,)
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        risk = self._model.predict(X)

        # For Coxnet, predict returns sparse — convert to dense
        if hasattr(risk, "toarray"):
            risk = risk.toarray().ravel()

        return np.asarray(risk).ravel()

    def predict_survival_function(
        self,
        X: np.ndarray,
        return_times: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict survival function S(t) for each sample.

        Args:
            X: Feature matrix (n_samples, n_features)
            return_times: If True, also return the time points

        Returns:
            Survival probabilities (n_samples, n_time_points) or
            (survival_probs, time_points) if return_times=True
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Get survival curves for each sample
        if hasattr(self._model, "predict_survival_function"):
            surv_funcs = self._model.predict_survival_function(X)

            # Collect all unique time points
            all_times: List[float] = []
            for sf in surv_funcs:
                all_times.extend(sf.x)
            all_times = sorted(set(all_times))

            # Evaluate each survival function at all time points
            surv_matrix = np.zeros((len(surv_funcs), len(all_times)))
            for i, sf in enumerate(surv_funcs):
                surv_matrix[i, :] = sf(all_times)

            if return_times:
                return surv_matrix, np.array(all_times)
            return surv_matrix
        else:
            raise NotImplementedError(
                f"Model type '{self.model_type}' does not support survival function prediction"
            )

    def get_c_index(
        self,
        X: np.ndarray,
        y_surv: Any,
    ) -> float:
        """
        Compute concordance index (C-index) for model evaluation.

        C-index measures how well the model ranks patients by risk.
        0.5 = random, 1.0 = perfect, 0.0 = perfectly wrong.

        Args:
            X: Feature matrix
            y_surv: Structured survival array

        Returns:
            C-index score
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        risk_scores = self.predict_risk(X)
        event = y_surv["event"]
        time = y_surv["time"]

        c_index, _, _, _, _ = concordance_index_censored(
            event_indicator=event,
            event_time=time,
            estimate=risk_scores,
        )

        logger.info(f"C-index: {c_index:.4f}")
        return float(c_index)

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get feature importance scores.

        For RSF: uses permutation-based importance.
        For Cox: uses absolute coefficient magnitudes.

        Args:
            top_n: Return only top N features

        Returns:
            Dictionary of feature_name → importance score
        """
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self.model_type == "rsf":
            if not hasattr(self._model, "feature_importances_"):
                raise AttributeError("RSF model does not have feature_importances_")
            importances = self._model.feature_importances_
        elif self.model_type == "cox":
            # Coxnet stores coefficients per alpha
            coefs = self._model.coef_
            if np.ndim(coefs) > 1:
                # Average across alphas for summary importance
                importances = np.abs(coefs).mean(axis=1) if coefs.ndim > 1 else np.abs(coefs).ravel()
            else:
                importances = np.abs(coefs).ravel()
        else:
            raise NotImplementedError(f"Feature importance not supported for {self.model_type}")

        # Handle sparse output
        if hasattr(importances, "toarray"):
            importances = importances.toarray().ravel()
        importances = np.asarray(importances).ravel()

        if self._feature_names is not None and len(self._feature_names) == len(importances):
            named_importances = dict(zip(self._feature_names, importances.tolist()))
        else:
            named_importances = {f"feature_{i}": float(v) for i, v in enumerate(importances)}

        # Sort and return top N
        sorted_importance = sorted(named_importances.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_importance[:top_n])

    def get_risk_percentile(
        self,
        X: np.ndarray,
        risk_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Convert risk scores to percentile ranks (0-100).

        Args:
            X: Feature matrix
            risk_scores: Pre-computed risk scores (optional)

        Returns:
            Percentile rank for each sample (0 = lowest risk, 100 = highest)
        """
        if risk_scores is None:
            risk_scores = self.predict_risk(X)

        from scipy.stats import rankdata
        percentile = (rankdata(risk_scores) - 1) / len(risk_scores) * 100
        return percentile

    def risk_stratification(
        self,
        X: np.ndarray,
        thresholds: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stratify students into risk groups based on risk scores.

        Args:
            X: Feature matrix
            thresholds: Custom percentile thresholds (default: [25, 50, 75])

        Returns:
            Tuple of (risk_scores, risk_labels)
        """
        risk_scores = self.predict_risk(X)
        percentiles = self.get_risk_percentile(X, risk_scores)

        if thresholds is None:
            thresholds = [25, 50, 75]

        labels = np.empty(len(risk_scores), dtype=object)
        prev = 0
        for i, t in enumerate(thresholds):
            labels[(percentiles >= prev) & (percentiles < t)] = ["low", "medium", "high", "very_high"][i]
            prev = t
        labels[percentiles >= prev] = ["low", "medium", "high", "very_high"][len(thresholds)]

        return risk_scores, labels

    def summary(self) -> str:
        """Return a summary string of the model."""
        model_type_str = self.model_type.upper()
        if self.model_type == "rsf":
            info = f"n_estimators={self.n_estimators}"
        else:
            info = "elastic-net regularization"

        return (
            f"SurvivalAnalyzer(\n"
            f"  model_type={model_type_str},\n"
            f"  {info}\n"
            f"  fitted={self._model is not None}\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: quick fit and evaluate
# ─────────────────────────────────────────────────────────────────────────────


def fit_and_evaluate(
    X: np.ndarray,
    event: np.ndarray,
    time: np.ndarray,
    model_type: str = "cox",
    test_size: float = 0.2,
    random_state: int = 42,
    **kwargs,
) -> Tuple[SurvivalAnalyzer, float, Dict[str, float]]:
    """
    Fit a survival model and evaluate with C-index on a test set.

    Args:
        X: Feature matrix (n_samples, n_features)
        event: Event indicator (bool)
        time: Time to event or censoring
        model_type: 'cox' or 'rsf'
        test_size: Fraction for test split
        random_state: Random seed
        **kwargs: Passed to SurvivalAnalyzer

    Returns:
        Tuple of (fitted model, c_index, feature_importance dict)
    """
    from sklearn.model_selection import train_test_split

    X_train, X_test, event_train, event_test, time_train, time_test = train_test_split(
        X, event, time, test_size=test_size, random_state=random_state, stratify=event
    )

    y_train = make_surv_array(event_train, time_train)
    y_test = make_surv_array(event_test, time_test)

    model = SurvivalAnalyzer(model_type=model_type, random_state=random_state, **kwargs)
    model.fit(X_train, y_train)

    c_idx = model.get_c_index(X_test, y_test)
    importance = model.get_feature_importance(top_n=20)

    return model, c_idx, importance


# ─────────────────────────────────────────────────────────────────────────────
# Module exports
# ─────────────────────────────────────────────────────────────────────────────
__all__ = [
    "SurvivalAnalyzer",
    "make_surv_array",
    "fit_and_evaluate",
]
