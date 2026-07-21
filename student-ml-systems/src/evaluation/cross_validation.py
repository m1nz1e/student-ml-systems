"""
Cross-validation strategies for student ML systems.

Implements three CV strategies:
1. Stratified K-Fold — for Course Recommender (preserves enrollment distribution)
2. TimeSeriesSplit — for Enrollment Yield (temporal validity)
3. Grouped K-Fold — for Early Warning (no student leakage)
"""

from typing import Tuple, Generator, Optional, List
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    TimeSeriesSplit,
    GroupKFold,
    cross_val_score,
)
from sklearn.base import BaseEstimator


class CrossValidationStrategies:
    """Factory class for cross-validation strategies."""

    @staticmethod
    def get_stratified_cv(
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> StratifiedKFold:
        """
        Stratified K-Fold for Course Recommender.

        Preserves class distribution across folds.

        Args:
            n_splits: Number of folds (default: 5)
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility

        Returns:
            StratifiedKFold object
        """
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

    @staticmethod
    def get_timeseries_cv(
        n_splits: int = 5,
        gap: int = 0,
        test_size: Optional[int] = None,
    ) -> TimeSeriesSplit:
        """
        TimeSeriesSplit for Enrollment Yield.

        Ensures temporal validity (train on past, test on future).

        Args:
            n_splits: Number of splits
            gap: Minimum gap between train and test (in samples)
            test_size: Size of test sets (None for auto)

        Returns:
            TimeSeriesSplit object
        """
        return TimeSeriesSplit(
            n_splits=n_splits,
            gap=gap,
            test_size=test_size,
        )

    @staticmethod
    def get_grouped_cv(n_splits: int = 5) -> GroupKFold:
        """
        Grouped K-Fold for Early Warning.

        Prevents same student appearing in both train and test.

        Args:
            n_splits: Number of folds

        Returns:
            GroupKFold object
        """
        return GroupKFold(n_splits=n_splits)


def cross_validate(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv_strategy: str = "stratified",
    n_splits: int = 5,
    groups: Optional[np.ndarray] = None,
    scoring: str = "accuracy",
    random_state: int = 42,
) -> Tuple[np.ndarray, float, float]:
    """
    Perform cross-validation with specified strategy.

    Args:
        model: Scikit-learn compatible model
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        cv_strategy: One of ['stratified', 'timeseries', 'grouped']
        n_splits: Number of folds
        groups: Group labels for GroupKFold (student IDs)
        scoring: Scoring metric (e.g., 'accuracy', 'f1', 'roc_auc')
        random_state: Random seed

    Returns:
        Tuple of (fold_scores, mean_score, std_score)

    Raises:
        ValueError: If invalid cv_strategy or missing groups for grouped CV
    """
    cv_factory = CrossValidationStrategies()

    if cv_strategy == "stratified":
        cv = cv_factory.get_stratified_cv(n_splits=n_splits, random_state=random_state)
    elif cv_strategy == "timeseries":
        cv = cv_factory.get_timeseries_cv(n_splits=n_splits)
    elif cv_strategy == "grouped":
        if groups is None:
            raise ValueError("Groups required for grouped cross-validation")
        cv = cv_factory.get_grouped_cv(n_splits=n_splits)
    else:
        raise ValueError(
            f"Invalid cv_strategy: {cv_strategy}. "
            "Must be one of ['stratified', 'timeseries', 'grouped']"
        )

    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, groups=groups)

    return scores, scores.mean(), scores.std()


def cross_validate_with_splits(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv_strategy: str = "stratified",
    n_splits: int = 5,
    groups: Optional[np.ndarray] = None,
    scoring: str = "accuracy",
    random_state: int = 42,
) -> Generator[Tuple[int, np.ndarray, np.ndarray, float], None, None]:
    """
    Perform cross-validation and yield per-fold results.

    Yields:
        Tuple of (fold_number, train_indices, val_indices, fold_score)
    """
    cv_factory = CrossValidationStrategies()

    if cv_strategy == "stratified":
        cv = cv_factory.get_stratified_cv(n_splits=n_splits, random_state=random_state)
        splits = cv.split(X, y)
    elif cv_strategy == "timeseries":
        cv = cv_factory.get_timeseries_cv(n_splits=n_splits)
        splits = cv.split(X)
    elif cv_strategy == "grouped":
        if groups is None:
            raise ValueError("Groups required for grouped cross-validation")
        cv = cv_factory.get_grouped_cv(n_splits=n_splits)
        splits = cv.split(X, y, groups)
    else:
        raise ValueError(f"Invalid cv_strategy: {cv_strategy}")

    for fold, (train_idx, val_idx) in enumerate(splits, 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_train, y_train)

        # Evaluate
        y_pred = model_clone.predict(X_val)
        if scoring == "accuracy":
            score = (y_pred == y_val).mean()
        elif scoring == "f1":
            from sklearn.metrics import f1_score

            score = f1_score(y_val, y_pred, average="weighted")
        elif scoring == "roc_auc":
            from sklearn.metrics import roc_auc_score

            y_pred_proba = model_clone.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred_proba)
        else:
            score = (y_pred == y_val).mean()

        yield fold, train_idx, val_idx, score


# Example usage
if __name__ == "__main__":
    # Example: Stratified K-Fold for Course Recommender
    from sklearn.ensemble import RandomForestClassifier

    # Synthetic data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 10)
    y = np.random.randint(0, 3, n_samples)  # 3 classes

    # Cross-validation
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    scores, mean_score, std_score = cross_validate(
        model, X, y, cv_strategy="stratified", n_splits=5, scoring="accuracy"
    )

    print(f"Stratified 5-Fold CV Results:")
    print(f"  Fold scores: {scores}")
    print(f"  Mean accuracy: {mean_score:.4f} (+/- {std_score:.4f})")

    # Example: TimeSeriesSplit for Enrollment Yield
    y_binary = np.random.randint(0, 2, n_samples)  # Binary classification
    scores_ts, mean_ts, std_ts = cross_validate(
        model, X, y_binary, cv_strategy="timeseries", n_splits=5, scoring="roc_auc"
    )

    print(f"\nTimeSeries 5-Split CV Results:")
    print(f"  Fold scores: {scores_ts}")
    print(f"  Mean ROC-AUC: {mean_ts:.4f} (+/- {std_ts:.4f})")

    # Example: Grouped K-Fold for Early Warning
    groups = np.repeat(np.arange(n_samples // 10), 10)  # 100 students, 10 samples each
    scores_grp, mean_grp, std_grp = cross_validate(
        model,
        X,
        y_binary,
        cv_strategy="grouped",
        n_splits=5,
        groups=groups,
        scoring="f1",
    )

    print(f"\nGrouped 5-Fold CV Results:")
    print(f"  Fold scores: {scores_grp}")
    print(f"  Mean F1: {mean_grp:.4f} (+/- {std_grp:.4f}")
