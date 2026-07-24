"""Evaluation Metrics for Exam Performance Prediction."""
from typing import Dict
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, mean_absolute_error,
    r2_score, precision_score, recall_score,
    brier_score_loss, mean_squared_error,
    cohen_kappa_score,
)


def evaluate_exam_performance(
    y_true_dict: Dict[str, np.ndarray],
    y_pred_dict: Dict[str, np.ndarray],
    y_proba_dict: Dict[str, np.ndarray] = None,
) -> Dict[str, float]:
    """
    Comprehensive evaluation of exam predictions.
    
    Args:
        y_true_dict: {exam_mark, pass_fail, grade_class}
        y_pred_dict: predictions
        y_proba_dict: probabilities (optional)
        
    Returns:
        Dict of all metrics
    """
    results = {}
    
    # --- Exam mark (regression) ---
    results['mark_mae'] = mean_absolute_error(y_true_dict['exam_mark'], y_pred_dict['exam_mark'])
    results['mark_rmse'] = np.sqrt(mean_squared_error(y_true_dict['exam_mark'], y_pred_dict['exam_mark']))
    results['mark_r2'] = r2_score(y_true_dict['exam_mark'], y_pred_dict['exam_mark'])
    results['mark_corr'] = np.corrcoef(y_true_dict['exam_mark'], y_pred_dict['exam_mark'])[0, 1]
    
    # --- Pass/Fail (binary) ---
    if len(np.unique(y_true_dict['pass_fail'])) > 1:
        results['pass_accuracy'] = accuracy_score(y_true_dict['pass_fail'], y_pred_dict['pass_fail'])
        results['pass_precision'] = precision_score(y_true_dict['pass_fail'], y_pred_dict['pass_fail'], zero_division=0)
        results['pass_recall'] = recall_score(y_true_dict['pass_fail'], y_pred_dict['pass_fail'], zero_division=0)
        
        if y_proba_dict and 'pass_proba' in y_proba_dict:
            proba = y_proba_dict['pass_proba']
            if proba.shape[1] == 2:
                results['pass_roc_auc'] = roc_auc_score(y_true_dict['pass_fail'], proba[:, 1])
                results['brier_score'] = brier_score_loss(y_true_dict['pass_fail'], proba[:, 1])
            else:
                results['pass_roc_auc'] = roc_auc_score(y_true_dict['pass_fail'], proba)
    
    # --- Grade class (ordinal) ---
    results['grade_accuracy'] = accuracy_score(y_true_dict['grade_class'], y_pred_dict['grade_class'])
    results['grade_qwk'] = cohen_kappa_score(y_true_dict['grade_class'], y_pred_dict['grade_class'], weights='quadratic')
    
    # Within-one accuracy (off by 1 grade still counts)
    within_one = np.abs(y_true_dict['grade_class'] - y_pred_dict['grade_class']) <= 1
    results['grade_within_1'] = float(within_one.mean())
    
    # --- Overall score ---
    results['overall_score'] = (
        max(0, results.get('mark_r2', 0)) * 0.30 +
        results.get('pass_roc_auc', 0.5) * 0.35 +
        results.get('grade_qwk', 0) * 0.20 +
        max(0, 1 - results.get('mark_mae', 10) / 10) * 0.15
    )
    
    return results
