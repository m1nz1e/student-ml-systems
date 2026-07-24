"""Evaluation Metrics for Module Demand Forecasting."""
from typing import Dict
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

def evaluate_module_demand(y_true_dict: Dict[str, np.ndarray], y_pred_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
    results = {}
    results['enrollment_mae'] = mean_absolute_error(y_true_dict['enrollment_count'], y_pred_dict['enrollment_count'])
    results['enrollment_rmse'] = np.sqrt(mean_squared_error(y_true_dict['enrollment_count'], y_pred_dict['enrollment_count']))
    results['enrollment_r2'] = r2_score(y_true_dict['enrollment_count'], y_pred_dict['enrollment_count'])
    results['fill_rate_mae'] = mean_absolute_error(y_true_dict['fill_rate'], y_pred_dict['fill_rate'])
    results['fill_rate_r2'] = r2_score(y_true_dict['fill_rate'], y_pred_dict['fill_rate'])
    results['demand_category_accuracy'] = accuracy_score(y_true_dict['demand_category'], y_pred_dict['demand_category'])
    results['overall_score'] = (
        min(1.0, 20 / max(results['enrollment_mae'], 1)) * 0.25 +
        max(0, results['enrollment_r2']) * 0.25 +
        max(0, results['fill_rate_r2']) * 0.25 +
        results['demand_category_accuracy'] * 0.25
    )
    return results
