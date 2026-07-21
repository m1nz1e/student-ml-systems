"""
Fairness Audit Module for Enrollment Yield Prediction.

Implements fairness metrics:
1. Demographic Parity Difference
2. Equalized Odds Difference
3. Disparate Impact Ratio (4/5ths rule)
4. Calibration by Group
5. Overall Fairness Score

Use Cases:
- Pre-deployment audit
- Continuous monitoring
- Bias detection and mitigation
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FairnessAuditor:
    """
    Comprehensive fairness auditing for enrollment yield models.

    Evaluates model fairness across protected groups:
    - Gender
    - Ethnicity
    - Socioeconomic status (IMD, POLAR)
    - Disability status
    """

    def __init__(
        self,
        protected_attributes: List[str],
        favorable_outcome: int = 1,
    ):
        """
        Initialize fairness auditor.

        Args:
            protected_attributes: List of protected attribute column names
            favorable_outcome: Value representing favorable outcome
        """
        self.protected_attributes = protected_attributes
        self.favorable_outcome = favorable_outcome
        self.audit_results: Dict[str, Any] = {}

    def demographic_parity_difference(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attribute: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate Demographic Parity Difference.

        DP Difference = max(P(Ŷ=1|A=a)) - min(P(Ŷ=1|A=a))

        Should be close to 0 (all groups have equal positive rates).

        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_attribute: Protected attribute values

        Returns:
            Tuple of (dp_difference, group_rates)
        """
        unique_groups = np.unique(protected_attribute)
        group_rates = {}

        for group in unique_groups:
            mask = protected_attribute == group
            if mask.sum() > 0:
                group_rates[str(group)] = y_pred[mask].mean()

        if len(group_rates) < 2:
            return 0.0, group_rates

        rates = list(group_rates.values())
        dp_diff = max(rates) - min(rates)

        return dp_diff, group_rates

    def equalized_odds_difference(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attribute: np.ndarray,
    ) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
        """
        Calculate Equalized Odds Difference.

        EO Difference = max(TPR difference, FPR difference) across groups.

        Should be close to 0 (all groups have equal TPR and FPR).

        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_attribute: Protected attribute values

        Returns:
            Tuple of (eo_diff, tpr_diff, fpr_diff, group_metrics)
        """
        unique_groups = np.unique(protected_attribute)
        group_metrics = {}

        tpr_by_group = {}
        fpr_by_group = {}

        for group in unique_groups:
            mask = protected_attribute == group
            if mask.sum() == 0:
                continue

            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]

            # True Positive Rate (Recall for positive class)
            pos_mask = y_true_group == 1
            if pos_mask.sum() > 0:
                tpr = y_pred_group[pos_mask].mean()
            else:
                tpr = 0.0

            # False Positive Rate
            neg_mask = y_true_group == 0
            if neg_mask.sum() > 0:
                fpr = y_pred_group[neg_mask].mean()
            else:
                fpr = 0.0

            tpr_by_group[str(group)] = tpr
            fpr_by_group[str(group)] = fpr

            group_metrics[str(group)] = {"tpr": tpr, "fpr": fpr}

        if len(tpr_by_group) < 2:
            return 0.0, 0.0, group_metrics

        tpr_diff = max(tpr_by_group.values()) - min(tpr_by_group.values())
        fpr_diff = max(fpr_by_group.values()) - min(fpr_by_group.values())

        eo_diff = max(tpr_diff, fpr_diff)

        return eo_diff, tpr_diff, fpr_diff, group_metrics

    def disparate_impact_ratio(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attribute: np.ndarray,
    ) -> Tuple[float, str, Dict[str, float]]:
        """
        Calculate Disparate Impact Ratio (4/5ths rule).

        DI Ratio = min(P(Ŷ=1|A=a)) / max(P(Ŷ=1|A=a))

        Should be >= 0.8 to pass 4/5ths rule.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_attribute: Protected attribute values

        Returns:
            Tuple of (di_ratio, pass_fail, group_rates)
        """
        unique_groups = np.unique(protected_attribute)
        group_rates = {}

        for group in unique_groups:
            mask = protected_attribute == group
            if mask.sum() > 0:
                group_rates[str(group)] = y_pred[mask].mean()

        if len(group_rates) < 2:
            return 1.0, "PASS", group_rates

        rates = list(group_rates.values())
        min_rate = min(rates)
        max_rate = max(rates)

        if max_rate == 0:
            di_ratio = 1.0
        else:
            di_ratio = min_rate / max_rate

        pass_fail = "PASS" if di_ratio >= 0.8 else "FAIL"

        return di_ratio, pass_fail, group_rates

    def calibration_by_group(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        protected_attribute: np.ndarray,
        n_bins: int = 10,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate calibration error by group.

        Measures how well predicted probabilities match actual outcomes
        across different groups.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            protected_attribute: Protected attribute values
            n_bins: Number of bins for calibration

        Returns:
            Tuple of (max_calibration_error, group_errors)
        """
        from sklearn.calibration import calibration_curve

        unique_groups = np.unique(protected_attribute)
        group_errors = {}
        max_error = 0.0

        for group in unique_groups:
            mask = protected_attribute == group
            if mask.sum() == 0:
                continue

            y_true_group = y_true[mask]
            y_proba_group = y_pred_proba[mask]

            # Calibration curve
            prob_true, prob_pred = calibration_curve(
                y_true_group, y_proba_group, n_bins=n_bins
            )

            # Calibration error (mean absolute difference)
            if len(prob_true) > 0:
                error = np.mean(np.abs(prob_true - prob_pred))
            else:
                error = 0.0

            group_errors[str(group)] = error
            max_error = max(max_error, error)

        return max_error, group_errors

    def overall_fairness_score(
        self,
        dp_diff: float,
        eo_diff: float,
        di_ratio: float,
        calibration_error: float,
    ) -> float:
        """
        Calculate overall fairness score (0-1, higher is better).

        Combines multiple fairness metrics into single score.

        Args:
            dp_diff: Demographic parity difference
            eo_diff: Equalized odds difference
            di_ratio: Disparate impact ratio
            calibration_error: Max calibration error

        Returns:
            Overall fairness score (0-1)
        """
        # Normalize each metric (0 = unfair, 1 = fair)
        dp_score = max(0, 1 - dp_diff / 0.2)  # 0.2 threshold
        eo_score = max(0, 1 - eo_diff / 0.2)
        di_score = min(1, di_ratio / 0.8)  # 0.8 is passing threshold
        calibration_score = max(0, 1 - calibration_error / 0.2)

        # Weighted average
        weights = {
            "dp": 0.25,
            "eo": 0.30,
            "di": 0.30,
            "calibration": 0.15,
        }

        overall_score = (
            weights["dp"] * dp_score +
            weights["eo"] * eo_score +
            weights["di"] * di_score +
            weights["calibration"] * calibration_score
        )

        return overall_score

    def audit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        protected_attributes_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Run comprehensive fairness audit.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            protected_attributes_df: DataFrame with protected attributes

        Returns:
            Comprehensive audit results dictionary
        """
        logger.info("Running fairness audit...")

        audit_results = {
            "protected_attributes": [],
            "metrics": {},
            "overall_score": 0.0,
            "pass_fail": "PENDING",
        }

        overall_scores = []

        for attr_name in self.protected_attributes:
            if attr_name not in protected_attributes_df.columns:
                logger.warning(f"Protected attribute '{attr_name}' not found")
                continue

            protected_attr = protected_attributes_df[attr_name].values

            # Demographic Parity
            dp_diff, dp_rates = self.demographic_parity_difference(
                y_true, y_pred, protected_attr
            )

            # Equalized Odds
            eo_diff, tpr_diff, fpr_diff, eo_metrics = self.equalized_odds_difference(
                y_true, y_pred, protected_attr
            )

            # Disparate Impact
            di_ratio, di_pass, di_rates = self.disparate_impact_ratio(
                y_true, y_pred, protected_attr
            )

            # Calibration
            calib_error, calib_errors = self.calibration_by_group(
                y_true, y_pred_proba, protected_attr
            )

            # Overall score for this attribute
            attr_score = self.overall_fairness_score(
                dp_diff, eo_diff, di_ratio, calib_error
            )

            overall_scores.append(attr_score)

            # Store results
            audit_results["protected_attributes"].append(attr_name)
            audit_results["metrics"][attr_name] = {
                "demographic_parity": {
                    "difference": dp_diff,
                    "group_rates": dp_rates,
                },
                "equalized_odds": {
                    "difference": eo_diff,
                    "tpr_difference": tpr_diff,
                    "fpr_difference": fpr_diff,
                    "group_metrics": eo_metrics,
                },
                "disparate_impact": {
                    "ratio": di_ratio,
                    "pass_fail": di_pass,
                    "group_rates": di_rates,
                },
                "calibration": {
                    "max_error": calib_error,
                    "group_errors": calib_errors,
                },
                "overall_score": attr_score,
            }

        # Overall fairness score (average across attributes)
        if overall_scores:
            audit_results["overall_score"] = np.mean(overall_scores)
            audit_results["pass_fail"] = (
                "PASS" if audit_results["overall_score"] >= 0.7 else "FAIL"
            )

        # Log results
        self._log_audit_results(audit_results)
        self.audit_results = audit_results

        return audit_results

    def _log_audit_results(self, results: Dict[str, Any]):
        """Log audit results in readable format."""
        logger.info("\n" + "=" * 60)
        logger.info("FAIRNESS AUDIT RESULTS")
        logger.info("=" * 60)

        logger.info(f"\nOverall Fairness Score: {results['overall_score']:.3f}")
        logger.info(f"Status: {results['pass_fail']}")

        for attr_name in results["protected_attributes"]:
            metrics = results["metrics"][attr_name]
            logger.info(f"\n--- {attr_name.upper()} ---")
            logger.info(f"  Demographic Parity Diff: {metrics['demographic_parity']['difference']:.3f}")
            logger.info(f"  Equalized Odds Diff: {metrics['equalized_odds']['difference']:.3f}")
            logger.info(f"  Disparate Impact Ratio: {metrics['disparate_impact']['ratio']:.3f} ({metrics['disparate_impact']['pass_fail']})")
            logger.info(f"  Calibration Error: {metrics['calibration']['max_error']:.3f}")
            logger.info(f"  Overall Score: {metrics['overall_score']:.3f}")

        logger.info("\n" + "=" * 60)

    def generate_report(self, output_path: str = "fairness_audit_report.md"):
        """
        Generate markdown fairness audit report.

        Args:
            output_path: Path to save report
        """
        if not self.audit_results:
            logger.warning("No audit results to report")
            return

        report = []
        report.append("# Fairness Audit Report\n")
        report.append(f"**Overall Fairness Score:** {self.audit_results['overall_score']:.3f}\n")
        report.append(f"**Status:** {'✅ PASS' if self.audit_results['pass_fail'] == 'PASS' else '❌ FAIL'}\n")
        report.append("\n---\n")

        for attr_name in self.audit_results["protected_attributes"]:
            metrics = self.audit_results["metrics"][attr_name]

            report.append(f"## Protected Attribute: {attr_name}\n")
            report.append(f"**Overall Score:** {metrics['overall_score']:.3f}\n\n")

            # Demographic Parity
            report.append("### Demographic Parity\n")
            report.append(f"- Difference: {metrics['demographic_parity']['difference']:.3f}\n")
            report.append("**Group Rates:**\n")
            for group, rate in metrics['demographic_parity']['group_rates'].items():
                report.append(f"  - {group}: {rate:.3f}\n")
            report.append("\n")

            # Disparate Impact
            report.append("### Disparate Impact (4/5ths Rule)\n")
            report.append(f"- Ratio: {metrics['disparate_impact']['ratio']:.3f}\n")
            report.append(f"- Status: {'✅ PASS' if metrics['disparate_impact']['pass_fail'] == 'PASS' else '❌ FAIL'}\n\n")

            # Equalized Odds
            report.append("### Equalized Odds\n")
            report.append(f"- Difference: {metrics['equalized_odds']['difference']:.3f}\n")
            report.append(f"- TPR Difference: {metrics['equalized_odds']['tpr_difference']:.3f}\n")
            report.append(f"- FPR Difference: {metrics['equalized_odds']['fpr_difference']:.3f}\n\n")

            # Calibration
            report.append("### Calibration\n")
            report.append(f"- Max Error: {metrics['calibration']['max_error']:.3f}\n\n")

            report.append("---\n\n")

        # Save report
        report_text = "".join(report)
        with open(output_path, "w") as f:
            f.write(report_text)

        logger.info(f"Fairness audit report saved to {output_path}")
        return report_text


# Example usage
if __name__ == "__main__":
    # Synthetic test
    np.random.seed(42)
    n_samples = 1000

    # Generate synthetic predictions
    y_true = np.random.binomial(1, 0.7, n_samples)
    y_pred_proba = np.random.uniform(0, 1, n_samples)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Protected attributes
    protected_df = pd.DataFrame({
        "gender": np.random.choice(["Male", "Female"], n_samples),
        "ethnicity": np.random.choice(["White", "Asian", "Black", "Mixed"], n_samples, p=[0.65, 0.20, 0.10, 0.05]),
    })

    # Run audit
    auditor = FairnessAuditor(protected_attributes=["gender", "ethnicity"])
    results = auditor.audit(y_true, y_pred, y_pred_proba, protected_df)

    # Generate report
    auditor.generate_report("fairness_audit_report.md")

    print("\n✓ Fairness audit complete!")
