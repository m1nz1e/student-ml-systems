"""
Evaluation Metrics for NSS Prediction.
"""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
)


def evaluate_nss_predictions(
    y_true_dict: Dict[str, np.ndarray],
    y_pred_dict: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Comprehensive evaluation of NSS predictions.
    
    Args:
        y_true_dict: True targets {
            'satisfied': binary (n_samples,),
            'nps': (n_samples,),
            'themes': (n_samples, 7)
        }
        y_pred_dict: Predicted targets (same structure)
        
    Returns:
        Dict of all metrics
    """
    results = {}
    
    # Satisfaction metrics
    if len(np.unique(y_true_dict['satisfied'])) > 1:
        results['satisfaction_accuracy'] = accuracy_score(
            y_true_dict['satisfied'], y_pred_dict['satisfied']
        )
        results['satisfaction_roc_auc'] = roc_auc_score(
            y_true_dict['satisfied'], y_pred_dict['satisfied']
        )
    
    # NPS metrics
    results['nps_mae'] = mean_absolute_error(y_true_dict['nps'], y_pred_dict['nps'])
    results['nps_r2'] = r2_score(y_true_dict['nps'], y_pred_dict['nps'])
    results['nps_corr'] = np.corrcoef(y_true_dict['nps'], y_pred_dict['nps'])[0, 1]
    
    # Theme metrics
    theme_names = [
        'teaching', 'assessment', 'feedback',
        'support', 'organisation', 'learning_resources', 'student_voice'
    ]
    
    theme_maes = []
    theme_r2s = []
    
    for i, theme in enumerate(theme_names):
        mae = mean_absolute_error(y_true_dict['themes'][:, i], y_pred_dict['themes'][:, i])
        r2 = r2_score(y_true_dict['themes'][:, i], y_pred_dict['themes'][:, i])
        results[f'{theme}_mae'] = mae
        results[f'{theme}_r2'] = r2
        theme_maes.append(mae)
        theme_r2s.append(r2)
    
    results['avg_theme_mae'] = np.mean(theme_maes)
    results['avg_theme_r2'] = np.mean(theme_r2s)
    
    # Overall score
    satisfaction_score = results.get('satisfaction_accuracy', 0.5) * 0.3
    nps_score = max(0, results['nps_r2']) * 0.2
    theme_score = max(0, results['avg_theme_r2']) * 0.5
    
    results['overall_score'] = satisfaction_score + nps_score + theme_score
    
    return results
