"""
Evaluation Metrics for Student Retention.

Evaluates multi-task retention predictions:
- Binary classification metrics (retention_risk)
- Regression metrics (risk_score)
- Ordinal classification metrics (risk_category, departure_year)
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_qwk(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """
    Quadratic Weighted Kappa for ordinal classification.

    Measures agreement between predicted and true ordinal values,
    with higher weight for larger disagreements.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    num_ratings = n_classes
    weight = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            weight[i, j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)

    hist_true = np.sum(cm, axis=1)
    hist_pred = np.sum(cm, axis=0)
    expected = np.outer(hist_true, hist_pred)
    if expected.sum() == 0:
        return 0.0

    numerator = np.sum(weight * cm)
    denominator = np.sum(weight * expected)

    if denominator == 0:
        return 0.0

    return 1.0 - (numerator / denominator)


def calculate_ordinal_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Strict ordinal accuracy (all levels must match exactly).
    """
    return np.mean(y_true == y_pred)


def calculate_ordinal_accuracy_within_n(
    y_true: np.ndarray, y_pred: np.ndarray, n: int = 1
) -> float:
    """
    Relaxed ordinal accuracy — prediction within n levels of true.
    """
    return np.mean(np.abs(y_true - y_pred) <= n)


def evaluate_retention(
    y_true_dict: Dict[str, np.ndarray],
    y_pred_dict: Dict[str, np.ndarray],
    y_proba_dict: Dict[str, np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of retention predictions.

    Args:
        y_true_dict: True labels {'retention_risk', 'risk_score', 'risk_category', 'departure_year'}
        y_pred_dict: Predicted labels
        y_proba_dict: Predicted probabilities (for binary/ordinal classifiers)

    Returns:
        Dict of all metrics
    """
    results = {}

    # --- Binary: retention_risk ---
    if 'retention_risk' in y_true_dict and 'retention_risk' in y_pred_dict:
        y_t = y_true_dict['retention_risk']
        y_p = y_pred_dict['retention_risk']

        results['risk_accuracy'] = accuracy_score(y_t, y_p)
        results['risk_precision'] = precision_score(y_t, y_p, zero_division=0)
        results['risk_recall'] = recall_score(y_t, y_p, zero_division=0)
        results['risk_f1'] = f1_score(y_t, y_p, zero_division=0)

        # ROC-AUC if probabilities available
        if 'retention_risk' in y_proba_dict and len(np.unique(y_t)) > 1:
            proba = y_proba_dict['retention_risk']
            # Handle binary (n_classes=2) vs already binary
            if proba.shape[1] == 2:
                results['risk_roc_auc'] = roc_auc_score(y_t, proba[:, 1])
            else:
                results['risk_roc_auc'] = roc_auc_score(y_t, proba)

    # --- Regression: risk_score ---
    if 'risk_score' in y_true_dict and 'risk_score' in y_pred_dict:
        y_t = y_true_dict['risk_score']
        y_p = y_pred_dict['risk_score']

        results['score_mae'] = mean_absolute_error(y_t, y_p)
        results['score_rmse'] = np.sqrt(mean_squared_error(y_t, y_p))
        # Clip predictions to [0, 100] before computing MAE
        y_p_clipped = np.clip(y_p, 0, 100)
        results['score_mae_clipped'] = mean_absolute_error(y_t, y_p_clipped)

    # --- Ordinal: risk_category (4 classes: Low=0, Medium=1, High=2, Critical=3) ---
    if 'risk_category' in y_true_dict and 'risk_category' in y_pred_dict:
        y_t = y_true_dict['risk_category']
        y_p = y_pred_dict['risk_category']
        n_classes = 4

        results['category_accuracy'] = accuracy_score(y_t, y_p)
        results['category_qwk'] = calculate_qwk(y_t, y_p, n_classes)
        results['category_exact'] = calculate_ordinal_accuracy(y_t, y_p)
        results['category_within_1'] = calculate_ordinal_accuracy_within_n(y_t, y_p, n=1)

    # --- Ordinal: departure_year (0-3) ---
    if 'departure_year' in y_true_dict and 'departure_year' in y_pred_dict:
        y_t = y_true_dict['departure_year']
        y_p = y_pred_dict['departure_year']
        n_classes = 4

        results['departure_accuracy'] = accuracy_score(y_t, y_p)
        results['departure_qwk'] = calculate_qwk(y_t, y_p, n_classes)
        results['departure_exact'] = calculate_ordinal_accuracy(y_t, y_p)
        results['departure_within_1'] = calculate_ordinal_accuracy_within_n(y_t, y_p, n=1)

    # --- Composite scores ---
    accs = [v for k, v in results.items() if 'accuracy' in k and 'composite' not in k]
    results['composite_accuracy'] = np.mean(accs) if accs else 0.0

    aucs = [v for k, v in results.items() if 'roc_auc' in k]
    results['composite_roc_auc'] = np.mean(aucs) if aucs else 0.0

    return results
