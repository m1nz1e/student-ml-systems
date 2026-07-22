"""
Fairness Auditing for Graduate Outcomes.

Checks for bias in employment and salary predictions:
- Demographic parity
- Error rate parity
- disparate impact
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraduateOutcomesFairnessAuditor:
    """
    Audits graduate outcome predictions for fairness.
    
    Checks:
    - Employment rate parity by demographic group
    - Salary gap by demographic group
    - Error rate parity (FPR/FNR)
    """
    
    def __init__(self):
        self.results = {}
    
    def employment_parity(
        self,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame,
        protected_groups: List[str],
    ) -> Dict[str, float]:
        """
        Calculate employment rate disparity across groups.
        
        Args:
            y_pred: Predicted employment status (0=Unemployed, 3=Employed)
            sensitive_features: DataFrame with demographic columns
            protected_groups: List of column names to check
            
        Returns:
            Dict of disparities per group
        """
        disparities = {}
        
        # Binary employment (Employed vs Not)
        employed = (y_pred >= 3).astype(int)
        
        for group in protected_groups:
            if group in sensitive_features.columns:
                rates = sensitive_features.groupby(group).apply(
                    lambda x: employed[x.index].mean(),
                    include_groups=False
                )
                max_diff = rates.max() - rates.min()
                disparities[f'{group}_employment_gap'] = max_diff
        
        return disparities
    
    def salary_gap(
        self,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame,
        protected_groups: List[str],
    ) -> Dict[str, float]:
        """
        Calculate salary band disparity across groups.
        
        Args:
            y_pred: Predicted salary band (0-3, higher = more £)
            sensitive_features: DataFrame with demographic columns
            protected_groups: List of column names to check
            
        Returns:
            Dict of salary gaps per group
        """
        gaps = {}
        
        for group in protected_groups:
            if group in sensitive_features.columns:
                mean_salary = sensitive_features.groupby(group).apply(
                    lambda x: y_pred[x.index].mean(),
                    include_groups=False
                )
                max_gap = mean_salary.max() - mean_salary.min()
                gaps[f'{group}_salary_gap'] = max_gap
        
        return gaps
    
    def error_rate_parity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame,
        protected_groups: List[str],
    ) -> Dict[str, float]:
        """
        Calculate error rate parity (FPR/FNR) across groups.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features: DataFrame with demographic columns
            protected_groups: List of column names to check
            
        Returns:
            Dict of error rate disparities per group
        """
        errors = {}
        
        for group in protected_groups:
            if group in sensitive_features.columns:
                fprs, fnrs = [], []
                for val in sensitive_features[group].unique():
                    mask = sensitive_features[group] == val
                    if mask.sum() > 0:
                        y_t = y_true[mask]
                        y_p = y_pred[mask]
                        fpr = np.mean(y_p[y_t == 0] == 1) if (y_t == 0).sum() > 0 else 0
                        fnr = np.mean(y_p[y_t == 1] == 0) if (y_t == 1).sum() > 0 else 0
                        fprs.append(fpr)
                        fnrs.append(fnr)
                
                if fprs and fnrs:
                    errors[f'{group}_fpr_gap'] = max(fprs) - min(fprs)
                    errors[f'{group}_fnr_gap'] = max(fnrs) - min(fnrs)
        
        return errors
    
    def audit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame,
        protected_groups: List[str],
    ) -> Dict[str, Any]:
        """
        Run full fairness audit.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features: DataFrame with demographic columns
            protected_groups: List of protected group columns
            
        Returns:
            Dict with all fairness metrics
        """
        audit_results = {}
        
        # Employment parity
        emp_parity = self.employment_parity(y_pred, sensitive_features, protected_groups)
        audit_results.update(emp_parity)
        
        # Salary gap
        sal_gap = self.salary_gap(y_pred, sensitive_features, protected_groups)
        audit_results.update(sal_gap)
        
        # Error rate parity
        err_parity = self.error_rate_parity(y_true, y_pred, sensitive_features, protected_groups)
        audit_results.update(err_parity)
        
        # Overall score (1 = perfectly fair)
        thresholds = {
            'employment_gap': 0.05,  # 5% max gap
            'salary_gap': 0.1,       # 10% max gap
            'fpr_gap': 0.05,
            'fnr_gap': 0.05,
        }
        
        violations = 0
        for key, threshold in thresholds.items():
            matching = [v for k, v in audit_results.items() if key in k]
            violations += sum(1 for v in matching if v > threshold)
        
        audit_results['fairness_score'] = 1.0 - (violations / max(len(thresholds) * len(protected_groups), 1))
        audit_results['fairness_status'] = 'PASS' if audit_results['fairness_score'] > 0.8 else 'FAIL'
        
        return audit_results
