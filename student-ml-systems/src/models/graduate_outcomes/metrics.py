"""
Evaluation Metrics for Graduate Outcomes.

Evaluates multi-task predictions:
- Per-task metrics (ROC-AUC, accuracy)
- Composite metrics
- QWK for ordinal targets (salary)
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_qwk(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """
    Quadratic Weighted Kappa for ordinal classification.
    
    Measures agreement between predicted and true ordinal values,
    with higher weight for larger disagreements.
    
    Args:
        y_true: True ordinal labels
        y_pred: Predicted ordinal labels
        n_classes: Number of classes
        
    Returns:
        QWK score (-1 to 1, 1 = perfect agreement)
    """
    # Build confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    
    # Weights
    num_ratings = n_classes
    weight = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            weight[i, j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)
    
    # Expected matrix
    hist_true = np.sum(cm, axis=1)
    hist_pred = np.sum(cm, axis=0)
    expected = np.outer(hist_true, hist_pred)
    expected = expected / expected.sum() * cm.sum()
    
    # Normalize
    if expected.sum() == 0:
        return 0.0
    
    numerator = np.sum(weight * cm)
    denominator = np.sum(weight * expected)
    
    if denominator == 0:
        return 0.0
    
    return 1.0 - (numerator / denominator)


def calculate_roc_auc_macro(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_classes: int
) -> float:
    """
    Macro-average ROC-AUC for multi-class.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities (n_samples, n_classes)
        n_classes: Number of classes
        
    Returns:
        Macro-average ROC-AUC
    """
    try:
        # One-vs-rest ROC-AUC for each class
        y_true_bin = np.zeros((len(y_true), n_classes))
        for i, val in enumerate(y_true):
            if 0 <= val < n_classes:
                y_true_bin[i, val] = 1
        
        aucs = []
        for i in range(n_classes):
            if len(np.unique(y_true_bin[:, i])) > 1:
                auc = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
                aucs.append(auc)
        
        return np.mean(aucs) if aucs else 0.0
    except Exception:
        return 0.0


def per_task_metrics(
    task_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_classes: int,
) -> Dict[str, float]:
    """
    Calculate metrics for a single task.
    
    Args:
        task_name: Name of the task
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        n_classes: Number of classes
        
    Returns:
        Dict of metrics
    """
    metrics = {
        f'{task_name}_accuracy': accuracy_score(y_true, y_pred),
        f'{task_name}_roc_auc': calculate_roc_auc_macro(y_true, y_proba, n_classes),
    }
    
    # QWK for salary (ordinal)
    if task_name == 'salary':
        metrics[f'{task_name}_qwk'] = calculate_qwk(y_true, y_pred, n_classes)
    
    return metrics


def evaluate_graduate_outcomes(
    y_true_dict: Dict[str, np.ndarray],
    y_pred_dict: Dict[str, np.ndarray],
    y_proba_dict: Dict[str, np.ndarray],
    class_counts: Dict[str, int],
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of graduate outcome predictions.
    
    Args:
        y_true_dict: True labels for each task
        y_pred_dict: Predicted labels for each task
        y_proba_dict: Predicted probabilities for each task
        class_counts: Number of classes for each task
        
    Returns:
        Dict of all metrics
    """
    results = {}
    
    # Per-task metrics
    for task_name in y_true_dict.keys():
        n_classes = class_counts.get(task_name, len(np.unique(y_true_dict[task_name])))
        task_metrics = per_task_metrics(
            task_name,
            y_true_dict[task_name],
            y_pred_dict[task_name],
            y_proba_dict[task_name],
            n_classes
        )
        results.update(task_metrics)
    
    # Composite metrics
    roc_aucs = [v for k, v in results.items() if 'roc_auc' in k]
    results['composite_roc_auc'] = np.mean(roc_aucs) if roc_aucs else 0.0
    
    accuracies = [v for k, v in results.items() if 'accuracy' in k and 'composite' not in k]
    results['composite_accuracy'] = np.mean(accuracies) if accuracies else 0.0
    
    return results
