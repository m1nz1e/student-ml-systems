"""
Evaluation Metrics for Degree Outcome Prediction.

Implements ordinal-aware evaluation metrics:
1. Mean Absolute Error (ordinal)
2. Exact Match Accuracy
3. Within-One Accuracy
4. Calibration Metrics
5. Comprehensive Evaluation

Classes: 0=Fail, 1=Third, 2=2:2, 3=2:1, 4=First
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mean_absolute_error_ordinal(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    MAE treating ordinal values as continuous.

    Measures the average absolute distance between predicted and true
    ordinal classes, treating them as evenly-spaced numeric values.

    Args:
        y_true: True ordinal labels (n_samples,)
        y_pred: Predicted ordinal labels (n_samples,)

    Returns:
        Mean absolute error (0 to K-1)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mae = np.mean(np.abs(y_true - y_pred))

    return float(mae)


def exact_match_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Exact match rate.

    Measures the fraction of predictions that exactly match the true class.

    Args:
        y_true: True ordinal labels (n_samples,)
        y_pred: Predicted ordinal labels (n_samples,)

    Returns:
        Exact match accuracy (0 to 1)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    accuracy = np.mean(y_true == y_pred)

    return float(accuracy)


def within_one_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Within-one accuracy (predict within 1 class).

    Measures the fraction of predictions that are within 1 class
    of the true value. This is more lenient than exact match.

    Args:
        y_true: True ordinal labels (n_samples,)
        y_pred: Predicted ordinal labels (n_samples,)

    Returns:
        Within-one accuracy (0 to 1)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    within_one = np.mean(np.abs(y_true - y_pred) <= 1)

    return float(within_one)


def within_two_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Within-two accuracy (predict within 2 classes).

    Measures the fraction of predictions that are within 2 classes
    of the true value.

    Args:
        y_true: True ordinal labels (n_samples,)
        y_pred: Predicted ordinal labels (n_samples,)

    Returns:
        Within-two accuracy (0 to 1)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    within_two = np.mean(np.abs(y_true - y_pred) <= 2)

    return float(within_two)


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Quadratic Weighted Kappa (QWK).

    Measures agreement between two ratings, giving more weight to
    larger disagreements.

    Args:
        y_true: True ordinal labels (n_samples,)
        y_pred: Predicted ordinal labels (n_samples,)

    Returns:
        QWK score (-1 to 1, where 1 is perfect agreement)
    """
    from sklearn.metrics import cohen_kappa_score

    y_true = np.asarray(y_true).flatten().astype(int)
    y_pred = np.asarray(y_pred).flatten().astype(int)

    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


def calibration_plot(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 5,
) -> Dict[str, Any]:
    """
    Calculate calibration metrics per class.

    Measures how well predicted probabilities match actual outcomes
    for each class.

    Args:
        y_true: True ordinal labels (n_samples,)
        y_proba: Predicted probabilities (n_samples, n_classes)
        n_bins: Number of bins for calibration

    Returns:
        Dictionary with calibration metrics per class
    """
    from sklearn.calibration import calibration_curve

    y_true = np.asarray(y_true).flatten()
    n_classes = y_proba.shape[1]

    class_metrics = {}

    for c in range(n_classes):
        y_true_binary = (y_true == c).astype(int)
        y_proba_c = y_proba[:, c]

        # Skip if class not present
        if y_true_binary.sum() == 0:
            class_metrics[c] = {
                "n_samples": 0,
                "calibration_error": None,
                "mean_predicted_prob": None,
                "actual_rate": None,
            }
            continue

        # Calibration curve
        try:
            prob_true, prob_pred = calibration_curve(
                y_true_binary, y_proba_c, n_bins=n_bins, strategy="uniform"
            )

            # Calibration error (ECE - Expected Calibration Error)
            if len(prob_true) > 0:
                ece = np.mean(np.abs(prob_true - prob_pred))
            else:
                ece = None
        except Exception:
            ece = None
            prob_pred = []
            prob_true = []

        class_metrics[c] = {
            "n_samples": int(y_true_binary.sum()),
            "calibration_error": float(ece) if ece is not None else None,
            "mean_predicted_prob": float(y_proba_c.mean()),
            "actual_rate": float(y_true_binary.mean()),
            "calibration_curve": {
                "prob_pred": [float(p) for p in prob_pred],
                "prob_true": [float(p) for p in prob_true],
            },
        }

    # Overall ECE (weighted by class frequency)
    total_samples = sum(m["n_samples"] for m in class_metrics.values())
    if total_samples > 0 and all(m["calibration_error"] is not None for m in class_metrics.values()):
        ece_weighted = sum(
            m["n_samples"] * m["calibration_error"]
            for m in class_metrics.values()
        ) / total_samples
    else:
        ece_weighted = None

    return {
        "class_metrics": class_metrics,
        "expected_calibration_error": float(ece_weighted) if ece_weighted is not None else None,
        "n_bins": n_bins,
    }


def confusion_matrix_ordinal(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 5,
) -> np.ndarray:
    """
    Compute ordinal confusion matrix.

    Args:
        y_true: True ordinal labels
        y_pred: Predicted ordinal labels
        n_classes: Number of classes

    Returns:
        Confusion matrix (n_classes, n_classes)
    """
    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    return cm


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class precision, recall, F1.

    Args:
        y_true: True ordinal labels
        y_pred: Predicted ordinal labels
        class_names: Names of classes

    Returns:
        Dictionary with per-class metrics
    """
    from sklearn.metrics import precision_recall_fscore_support

    y_true = np.asarray(y_true).flatten().astype(int)
    y_pred = np.asarray(y_pred).flatten().astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    metrics = {}
    for c in range(len(class_names)):
        metrics[class_names[c]] = {
            "precision": float(precision[c]),
            "recall": float(recall[c]),
            "f1": float(f1[c]),
            "support": int(support[c]),
        }

    return metrics


def evaluate_degree_outcome(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Comprehensive evaluation for degree outcome prediction.

    Args:
        y_true: True ordinal labels (n_samples,)
        y_pred: Predicted ordinal labels (n_samples,)
        y_proba: Predicted probabilities (n_samples, n_classes)
        class_names: Names of classes

    Returns:
        Comprehensive evaluation dictionary
    """
    logger.info("Evaluating degree outcome predictions...")

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Ordinal metrics
    mae = mean_absolute_error_ordinal(y_true, y_pred)
    exact_acc = exact_match_accuracy(y_true, y_pred)
    within_one = within_one_accuracy(y_true, y_pred)
    within_two = within_two_accuracy(y_true, y_pred)
    qwk = quadratic_weighted_kappa(y_true, y_pred)

    # Per-class metrics
    per_class = per_class_metrics(y_true, y_pred, class_names)

    # Calibration
    calibration = calibration_plot(y_true, y_proba)

    # Confusion matrix
    cm = confusion_matrix_ordinal(y_true, y_pred, n_classes=len(class_names))

    # Overall accuracy
    from sklearn.metrics import accuracy_score
    overall_accuracy = accuracy_score(y_true, y_pred)

    results = {
        "ordinal_metrics": {
            "mae": mae,
            "exact_match_accuracy": exact_acc,
            "within_one_accuracy": within_one,
            "within_two_accuracy": within_two,
            "quadratic_weighted_kappa": qwk,
        },
        "overall_accuracy": float(overall_accuracy),
        "per_class": per_class,
        "calibration": calibration,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }

    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("DEGREE OUTCOME EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"\nOrdinal Metrics:")
    logger.info(f"  MAE: {mae:.3f}")
    logger.info(f"  Exact Match Accuracy: {exact_acc:.3f} ({exact_acc*100:.1f}%)")
    logger.info(f"  Within-One Accuracy: {within_one:.3f} ({within_one*100:.1f}%)")
    logger.info(f"  Within-Two Accuracy: {within_two:.3f} ({within_two*100:.1f}%)")
    logger.info(f"  Quadratic Weighted Kappa: {qwk:.3f}")
    logger.info(f"\nOverall Accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")

    logger.info(f"\nPer-Class Metrics:")
    for c_name, c_metrics in per_class.items():
        logger.info(
            f"  {c_name}: P={c_metrics['precision']:.3f} "
            f"R={c_metrics['recall']:.3f} F1={c_metrics['f1']:.3f} "
            f"(n={c_metrics['support']})"
        )

    if calibration["expected_calibration_error"] is not None:
        logger.info(
            f"\nCalibration ECE: {calibration['expected_calibration_error']:.3f}"
        )

    logger.info("\n" + "=" * 60)

    return results
