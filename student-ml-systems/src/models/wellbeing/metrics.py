"""Evaluation Metrics for Student Wellbeing Prediction."""
from typing import Dict
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, mean_absolute_error,
    r2_score, precision_score, recall_score,
    brier_score_loss, mean_squared_error,
    cohen_kappa_score,
)


def evaluate_wellbeing(
    y_true_dict: Dict[str, np.ndarray],
    y_pred_dict: Dict[str, np.ndarray],
    y_proba_dict: Dict[str, np.ndarray] = None,
) -> Dict[str, float]:
    """
    Comprehensive evaluation of wellbeing predictions.
    
    Args:
        y_true_dict: {wellbeing_score, at_risk, risk_level, support_need}
        y_pred_dict: predictions
        y_proba_dict: probabilities (optional)
        
    Returns:
        Dict of all metrics
    """
    results = {}
    
    # --- Wellbeing score (regression) ---
    results['score_mae'] = mean_absolute_error(y_true_dict['wellbeing_score'], y_pred_dict['wellbeing_score'])
    results['score_rmse'] = np.sqrt(mean_squared_error(y_true_dict['wellbeing_score'], y_pred_dict['wellbeing_score']))
    results['score_r2'] = r2_score(y_true_dict['wellbeing_score'], y_pred_dict['wellbeing_score'])
    results['score_corr'] = np.corrcoef(y_true_dict['wellbeing_score'], y_pred_dict['wellbeing_score'])[0, 1]
    
    # --- At-risk (binary) ---
    if len(np.unique(y_true_dict['at_risk'])) > 1:
        results['at_risk_accuracy'] = accuracy_score(y_true_dict['at_risk'], y_pred_dict['at_risk'])
        results['at_risk_precision'] = precision_score(y_true_dict['at_risk'], y_pred_dict['at_risk'], zero_division=0)
        results['at_risk_recall'] = recall_score(y_true_dict['at_risk'], y_pred_dict['at_risk'], zero_division=0)
        
        if y_proba_dict and 'risk_proba' in y_proba_dict:
            proba = y_proba_dict['risk_proba']
            if proba.shape[1] == 2:
                results['at_risk_roc_auc'] = roc_auc_score(y_true_dict['at_risk'], proba[:, 1])
                results['brier_score'] = brier_score_loss(y_true_dict['at_risk'], proba[:, 1])
    
    # --- Support need (ordinal) ---
    results['support_accuracy'] = accuracy_score(y_true_dict['support_need'], y_pred_dict['support_need'])
    results['support_qwk'] = cohen_kappa_score(y_true_dict['support_need'], y_pred_dict['support_need'], weights='quadratic')
    
    within_one = np.abs(y_true_dict['support_need'] - y_pred_dict['support_need']) <= 1
    results['support_within_1'] = float(within_one.mean())
    
    # --- Overall score ---
    results['overall_score'] = (
        max(0, results.get('score_r2', 0)) * 0.30 +
        results.get('at_risk_roc_auc', 0.5) * 0.40 +
        results.get('support_qwk', 0) * 0.15 +
        max(0, 1 - results.get('score_mae', 20) / 20) * 0.15
    )
    
    return results
