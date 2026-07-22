"""
Fairness Audit Module for Degree Outcome Prediction.

Implements fairness metrics adapted for ordinal classification:
1. Demographic parity (prediction rates across groups)
2. Error rate parity (error rates across groups)
3. Ordinal disparity (ordinal distance bias)

Classes: 0=Fail, 1=Third, 2=2:2, 3=2:1, 4=First
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DegreeOutcomeFairnessAuditor:
    """
    Fairness auditor for degree outcome models.

    Evaluates fairness across protected groups for ordinal predictions.
    """

    CLASS_NAMES = ["Fail", "Third", "2:2", "2:1", "First"]

    def __init__(self):
        """Initialize fairness auditor."""
        self.audit_results: Dict[str, Any] = {}

    def demographic_parity(
        self,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame,
        protected_attrs: List[str],
    ) -> pd.DataFrame:
        """
        Check prediction rates across groups.

        For ordinal outcomes, checks the distribution of predicted classes
        and mean predicted ordinal value per group.

        Args:
            y_pred: Predicted ordinal labels (n_samples,)
            sensitive_features: DataFrame with protected attributes
            protected_attrs: List of protected attribute column names

        Returns:
            DataFrame with demographic parity metrics per attribute
        """
        results = []

        for attr in protected_attrs:
            if attr not in sensitive_features.columns:
                logger.warning(f"Protected attribute '{attr}' not found")
                continue

            groups = sensitive_features[attr].unique()
            group_metrics = {}

            for group in groups:
                mask = sensitive_features[attr] == group
                if mask.sum() == 0:
                    continue

                y_group = y_pred[mask]

                # Mean ordinal prediction
                mean_pred = y_group.mean()

                # Distribution of predictions
                class_counts = np.bincount(y_group, minlength=5) / len(y_group)

                # Rate of favorable outcomes (2:1 or First)
                favorable_mask = y_group >= 3
                favorable_rate = favorable_mask.mean()

                group_metrics[str(group)] = {
                    "mean_predicted_class": float(mean_pred),
                    "favorable_rate": float(favorable_rate),
                    "class_distribution": class_counts.tolist(),
                }

            # Calculate disparity
            mean_preds = [v["mean_predicted_class"] for v in group_metrics.values()]
            if len(mean_preds) >= 2:
                disparity = max(mean_preds) - min(mean_preds)
            else:
                disparity = 0.0

            results.append(
                {
                    "protected_attribute": attr,
                    "disparity": float(disparity),
                    "group_metrics": group_metrics,
                }
            )

        return pd.DataFrame(results)

    def error_rate_parity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame,
        protected_attrs: List[str],
    ) -> pd.DataFrame:
        """
        Check error rates across groups.

        Computes per-group error rates (absolute deviation from true class).

        Args:
            y_true: True ordinal labels (n_samples,)
            y_pred: Predicted ordinal labels (n_samples,)
            sensitive_features: DataFrame with protected attributes
            protected_attrs: List of protected attribute column names

        Returns:
            DataFrame with error rate metrics per attribute
        """
        results = []

        for attr in protected_attrs:
            if attr not in sensitive_features.columns:
                continue

            groups = sensitive_features[attr].unique()
            group_metrics = {}

            for group in groups:
                mask = sensitive_features[attr] == group
                if mask.sum() == 0:
                    continue

                y_true_group = y_true[mask]
                y_pred_group = y_pred[mask]

                # Exact match error rate
                exact_error = (y_true_group != y_pred_group).mean()

                # Mean absolute error (ordinal)
                mae = np.abs(y_true_group - y_pred_group).mean()

                # Within-one accuracy
                within_one = (np.abs(y_true_group - y_pred_group) <= 1).mean()

                # Per-class error rates
                per_class_error = {}
                for c in range(5):
                    c_mask = y_true_group == c
                    if c_mask.sum() > 0:
                        per_class_error[int(c)] = float(
                            (y_true_group[c_mask] != y_pred_group[c_mask]).mean()
                        )

                group_metrics[str(group)] = {
                    "exact_error_rate": float(exact_error),
                    "mae": float(mae),
                    "within_one_accuracy": float(within_one),
                    "per_class_error": per_class_error,
                }

            # Disparity in MAE
            maes = [v["mae"] for v in group_metrics.values()]
            if len(maes) >= 2:
                mae_disparity = max(maes) - min(maes)
            else:
                mae_disparity = 0.0

            results.append(
                {
                    "protected_attribute": attr,
                    "mae_disparity": float(mae_disparity),
                    "group_metrics": group_metrics,
                }
            )

        return pd.DataFrame(results)

    def ordinal_disparity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame,
        protected_attrs: List[str],
    ) -> pd.DataFrame:
        """
        Check ordinal distance bias.

        Measures whether certain groups systematically receive
        predictions that are higher or lower than their true outcomes.

        Args:
            y_true: True ordinal labels (n_samples,)
            y_pred: Predicted ordinal labels (n_samples,)
            sensitive_features: DataFrame with protected attributes
            protected_attrs: List of protected attribute column names

        Returns:
            DataFrame with ordinal disparity metrics per attribute
        """
        results = []

        for attr in protected_attrs:
            if attr not in sensitive_features.columns:
                continue

            groups = sensitive_features[attr].unique()
            group_metrics = {}

            for group in groups:
                mask = sensitive_features[attr] == group
                if mask.sum() == 0:
                    continue

                y_true_group = y_true[mask]
                y_pred_group = y_pred[mask]

                # Mean bias (positive = overestimation)
                bias = (y_pred_group - y_true_group).mean()

                # Signed error distribution
                signed_errors = y_pred_group - y_true_group

                # Rate of overestimation
                overest_rate = (signed_errors > 0).mean()

                # Rate of underestimation
                underest_rate = (signed_errors < 0).mean()

                # Normalized bias (relative to scale)
                normalized_bias = bias / 4.0  # 4 is the max ordinal distance

                group_metrics[str(group)] = {
                    "mean_bias": float(bias),
                    "normalized_bias": float(normalized_bias),
                    "overestimation_rate": float(overest_rate),
                    "underestimation_rate": float(underest_rate),
                    "signed_error_std": float(signed_errors.std()),
                }

            # Disparity in bias
            biases = [v["mean_bias"] for v in group_metrics.values()]
            if len(biases) >= 2:
                bias_disparity = max(biases) - min(biases)
            else:
                bias_disparity = 0.0

            results.append(
                {
                    "protected_attribute": attr,
                    "bias_disparity": float(bias_disparity),
                    "group_metrics": group_metrics,
                }
            )

        return pd.DataFrame(results)

    def audit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: pd.DataFrame,
        protected_attrs: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run comprehensive fairness audit.

        Args:
            y_true: True ordinal labels
            y_pred: Predicted ordinal labels
            sensitive_features: DataFrame with protected attributes
            protected_attrs: List of protected attributes (if None, use all columns)

        Returns:
            Comprehensive audit results dictionary
        """
        if protected_attrs is None:
            protected_attrs = sensitive_features.columns.tolist()

        logger.info(f"Running fairness audit on {len(protected_attrs)} protected attributes...")

        # Demographic parity
        dp_results = self.demographic_parity(y_pred, sensitive_features, protected_attrs)

        # Error rate parity
        er_results = self.error_rate_parity(y_true, y_pred, sensitive_features, protected_attrs)

        # Ordinal disparity
        od_results = self.ordinal_disparity(y_true, y_pred, sensitive_features, protected_attrs)

        audit_results = {
            "demographic_parity": dp_results.to_dict(orient="records"),
            "error_rate_parity": er_results.to_dict(orient="records"),
            "ordinal_disparity": od_results.to_dict(orient="records"),
        }

        # Compute overall fairness score
        all_disparities = []

        for _, row in dp_results.iterrows():
            all_disparities.append(row["disparity"])

        for _, row in er_results.iterrows():
            all_disparities.append(row["mae_disparity"])

        for _, row in od_results.iterrows():
            all_disparities.append(row["bias_disparity"])

        if all_disparities:
            # Lower disparity = better fairness
            # Score of 1.0 = no disparity, 0.0 = max disparity
            max_disparity = 2.0  # Max possible MAE disparity
            avg_disparity = np.mean(all_disparities)
            overall_score = max(0, 1 - avg_disparity / max_disparity)
        else:
            overall_score = 1.0

        audit_results["overall_fairness_score"] = float(overall_score)
        audit_results["pass_fail"] = "PASS" if overall_score >= 0.7 else "FAIL"

        self.audit_results = audit_results
        self._log_results(audit_results)

        return audit_results

    def _log_results(self, results: Dict[str, Any]):
        """Log audit results."""
        logger.info("\n" + "=" * 60)
        logger.info("DEGREE OUTCOME FAIRNESS AUDIT RESULTS")
        logger.info("=" * 60)

        logger.info(f"\nOverall Fairness Score: {results['overall_fairness_score']:.3f}")
        logger.info(f"Status: {results['pass_fail']}")

        for attr_metrics in results["demographic_parity"]:
            attr = attr_metrics["protected_attribute"]
            logger.info(f"\n--- {attr.upper()} ---")
            logger.info(f"  Demographic Disparity: {attr_metrics['disparity']:.3f}")

        for attr_metrics in results["error_rate_parity"]:
            attr = attr_metrics["protected_attribute"]
            logger.info(f"\n--- {attr.upper()} ---")
            logger.info(f"  MAE Disparity: {attr_metrics['mae_disparity']:.3f}")

        for attr_metrics in results["ordinal_disparity"]:
            attr = attr_metrics["protected_attribute"]
            logger.info(f"\n--- {attr.upper()} ---")
            logger.info(f"  Bias Disparity: {attr_metrics['bias_disparity']:.3f}")

        logger.info("\n" + "=" * 60)
