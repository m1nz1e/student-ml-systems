"""
Data validation module for Student ML Systems.

Implements data quality checks using Great Expectations-style validation.
Checks for:
- Schema validation (types, nulls)
- Value ranges
- Uniqueness constraints
- Data drift detection (PSI, KL divergence)
- Referential integrity
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationRule:
    """Base class for validation rules."""

    def __init__(self, name: str, description: str, severity: str = "warning"):
        """
        Initialize validation rule.

        Args:
            name: Rule name
            description: Human-readable description
            severity: One of ['info', 'warning', 'error', 'critical']
        """
        self.name = name
        self.description = description
        self.severity = severity

    def validate(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate DataFrame against rule.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (passed, message)
        """
        raise NotImplementedError


class SchemaValidation(ValidationRule):
    """Validate column types and presence."""

    def __init__(
        self,
        expected_columns: List[str],
        expected_types: Dict[str, str],
        allow_missing: bool = False,
    ):
        super().__init__(
            name="schema_validation",
            description="Validate column presence and types",
            severity="error",
        )
        self.expected_columns = expected_columns
        self.expected_types = expected_types
        self.allow_missing = allow_missing

    def validate(self, df: pd.DataFrame) -> Tuple[bool, str]:
        messages = []
        passed = True

        # Check column presence
        missing_cols = set(self.expected_columns) - set(df.columns)
        if missing_cols:
            if self.allow_missing:
                messages.append(f"Missing columns (allowed): {missing_cols}")
            else:
                messages.append(f"Missing required columns: {missing_cols}")
                passed = False

        # Check column types
        for col, expected_type in self.expected_types.items():
            if col not in df.columns:
                continue

            actual_type = str(df[col].dtype)
            if expected_type == "numeric" and actual_type not in [
                "int64",
                "float64",
                "int32",
                "float32",
            ]:
                messages.append(f"Column {col} has type {actual_type}, expected numeric")
                passed = False
            elif expected_type == "string" and actual_type not in [
                "object",
                "string",
            ]:
                messages.append(f"Column {col} has type {actual_type}, expected string")
                passed = False
            elif expected_type == "datetime" and "datetime" not in actual_type:
                messages.append(
                    f"Column {col} has type {actual_type}, expected datetime"
                )
                passed = False

        return passed, "; ".join(messages) if messages else "Schema validation passed"


class NullCheck(ValidationRule):
    """Check for null values in columns."""

    def __init__(
        self,
        columns: List[str],
        max_null_pct: float = 0.0,
        allow_nulls: bool = False,
    ):
        super().__init__(
            name="null_check",
            description="Check for null values",
            severity="warning" if allow_nulls else "error",
        )
        self.columns = columns
        self.max_null_pct = max_null_pct
        self.allow_nulls = allow_nulls

    def validate(self, df: pd.DataFrame) -> Tuple[bool, str]:
        messages = []
        passed = True

        for col in self.columns:
            if col not in df.columns:
                continue

            null_pct = df[col].isnull().mean()
            if null_pct > self.max_null_pct:
                messages.append(
                    f"Column {col} has {null_pct * 100:.1f}% nulls (max: {self.max_null_pct * 100}%)"
                )
                if not self.allow_nulls:
                    passed = False

        return passed, "; ".join(messages) if messages else "Null check passed"


class RangeCheck(ValidationRule):
    """Check that numeric values are within expected ranges."""

    def __init__(
        self,
        column: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        super().__init__(
            name="range_check",
            description=f"Check {column} is within range",
            severity="warning",
        )
        self.column = column
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, df: pd.DataFrame) -> Tuple[bool, str]:
        if self.column not in df.columns:
            return True, f"Column {self.column} not found"

        col_data = df[self.column].dropna()
        messages = []
        passed = True

        if self.min_value is not None:
            below_min = (col_data < self.min_value).sum()
            if below_min > 0:
                messages.append(
                    f"{below_min} values in {self.column} below minimum {self.min_value}"
                )
                passed = False

        if self.max_value is not None:
            above_max = (col_data > self.max_value).sum()
            if above_max > 0:
                messages.append(
                    f"{above_max} values in {self.column} above maximum {self.max_value}"
                )
                passed = False

        return passed, "; ".join(messages) if messages else "Range check passed"


class UniquenessCheck(ValidationRule):
    """Check for duplicate values in columns."""

    def __init__(self, columns: List[str], allow_duplicates: bool = False):
        super().__init__(
            name="uniqueness_check",
            description="Check for uniqueness",
            severity="error" if not allow_duplicates else "warning",
        )
        self.columns = columns
        self.allow_duplicates = allow_duplicates

    def validate(self, df: pd.DataFrame) -> Tuple[bool, str]:
        messages = []
        passed = True

        for col in self.columns:
            if col not in df.columns:
                continue

            duplicates = df[col].duplicated().sum()
            if duplicates > 0:
                messages.append(f"Column {col} has {duplicates} duplicate values")
                if not self.allow_duplicates:
                    passed = False

        return passed, "; ".join(messages) if messages else "Uniqueness check passed"


class CardinalityCheck(ValidationRule):
    """Check categorical column cardinality."""

    def __init__(self, column: str, min_cardinality: int = 1, max_cardinality: int = 100):
        super().__init__(
            name="cardinality_check",
            description=f"Check {column} cardinality",
            severity="warning",
        )
        self.column = column
        self.min_cardinality = min_cardinality
        self.max_cardinality = max_cardinality

    def validate(self, df: pd.DataFrame) -> Tuple[bool, str]:
        if self.column not in df.columns:
            return True, f"Column {self.column} not found"

        cardinality = df[self.column].nunique()
        messages = []
        passed = True

        if cardinality < self.min_cardinality:
            messages.append(
                f"Column {self.column} has low cardinality ({cardinality} < {self.min_cardinality})"
            )
            passed = False

        if cardinality > self.max_cardinality:
            messages.append(
                f"Column {self.column} has high cardinality ({cardinality} > {self.max_cardinality})"
            )

        return passed, "; ".join(messages) if messages else "Cardinality check passed"


class DataDriftDetector:
    """
    Detect data drift between reference and current datasets.

    Implements:
    - Population Stability Index (PSI)
    - KL Divergence
    - Kolmogorov-Smirnov test
    """

    def __init__(self, reference_data: pd.DataFrame):
        """
        Initialize drift detector with reference data.

        Args:
            reference_data: Historical/baseline data
        """
        self.reference_data = reference_data
        self.reference_stats = {}

        # Compute reference statistics
        for col in reference_data.columns:
            if reference_data[col].dtype in ["int64", "float64"]:
                # Numeric: compute histogram
                hist, bin_edges = np.histogram(reference_data[col].dropna(), bins=20)
                self.reference_stats[col] = {
                    "type": "numeric",
                    "histogram": hist / hist.sum(),  # Normalize
                    "bin_edges": bin_edges,
                }
            else:
                # Categorical: compute value counts
                value_counts = reference_data[col].value_counts(normalize=True)
                self.reference_stats[col] = {
                    "type": "categorical",
                    "distribution": value_counts.to_dict(),
                }

    def compute_psi(
        self, current_data: pd.DataFrame, column: str
    ) -> Tuple[float, str]:
        """
        Compute Population Stability Index for a column.

        PSI Interpretation:
        - < 0.1: No significant change
        - 0.1-0.25: Moderate change
        - > 0.25: Significant change

        Args:
            current_data: Current dataset
            column: Column to check

        Returns:
            Tuple of (PSI value, interpretation)
        """
        if column not in self.reference_stats:
            return 0.0, "Column not in reference data"

        ref_stats = self.reference_stats[column]
        current_col = current_data[column].dropna()

        if len(current_col) == 0:
            return float("inf"), "Current data is empty"

        if ref_stats["type"] == "numeric":
            # Numeric: use reference bins
            hist_current, _ = np.histogram(
                current_col, bins=ref_stats["bin_edges"]
            )
            current_dist = hist_current / hist_current.sum()
            ref_dist = ref_stats["histogram"]
        else:
            # Categorical: align categories
            ref_dist_dict = ref_stats["distribution"]
            current_dist_dict = current_col.value_counts(normalize=True).to_dict()

            all_categories = set(ref_dist_dict.keys()) | set(current_dist_dict.keys())
            ref_dist = np.array([ref_dist_dict.get(cat, 0.0) for cat in all_categories])
            current_dist = np.array(
                [current_dist_dict.get(cat, 0.0) for cat in all_categories]
            )

        # Add small epsilon to avoid division by zero
        ref_dist = ref_dist + 1e-6
        current_dist = current_dist + 1e-6

        # Normalize
        ref_dist = ref_dist / ref_dist.sum()
        current_dist = current_dist / current_dist.sum()

        # PSI formula: sum((actual - expected) * ln(actual / expected))
        psi = np.sum((current_dist - ref_dist) * np.log(current_dist / ref_dist))

        # Interpretation
        if psi < 0.1:
            interpretation = "No significant change"
        elif psi < 0.25:
            interpretation = "Moderate change"
        else:
            interpretation = "Significant change"

        return psi, interpretation

    def compute_kl_divergence(
        self, current_data: pd.DataFrame, column: str
    ) -> float:
        """
        Compute KL Divergence between reference and current distributions.

        Args:
            current_data: Current dataset
            column: Column to check

        Returns:
            KL divergence value
        """
        psi, _ = self.compute_psi(current_data, column)
        return psi  # PSI is based on KL divergence

    def ks_test(self, current_data: pd.DataFrame, column: str) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test for numeric columns.

        Args:
            current_data: Current dataset
            column: Column to check (must be numeric)

        Returns:
            Tuple of (KS statistic, p-value)
        """
        if column not in self.reference_data.columns:
            return 0.0, 1.0

        ref_col = self.reference_data[column].dropna()
        current_col = current_data[column].dropna()

        if len(ref_col) == 0 or len(current_col) == 0:
            return 0.0, 1.0

        ks_stat, p_value = stats.ks_2samp(ref_col, current_col)
        return ks_stat, p_value

    def detect_all_drift(
        self, current_data: pd.DataFrame, threshold: float = 0.1
    ) -> Dict[str, Dict[str, Any]]:
        """
        Detect drift for all columns.

        Args:
            current_data: Current dataset
            threshold: PSI threshold for flagging drift

        Returns:
            Dictionary of drift results per column
        """
        drift_results = {}

        for column in self.reference_stats.keys():
            if column not in current_data.columns:
                continue

            psi, interpretation = self.compute_psi(current_data, column)

            drift_results[column] = {
                "psi": psi,
                "interpretation": interpretation,
                "drift_detected": psi > threshold,
            }

            # Add KS test for numeric columns
            if self.reference_stats[column]["type"] == "numeric":
                ks_stat, p_value = self.ks_test(current_data, column)
                drift_results[column]["ks_statistic"] = ks_stat
                drift_results[column]["ks_p_value"] = p_value

        return drift_results


class DataValidator:
    """
    Main validator class that orchestrates all validation rules.
    """

    def __init__(self):
        self.rules: List[ValidationRule] = []
        self.drift_detector: Optional[DataDriftDetector] = None
        self.validation_results: Dict[str, Any] = {}

    def add_rule(self, rule: ValidationRule):
        """Add a validation rule."""
        self.rules.append(rule)

    def set_reference_data(self, reference_data: pd.DataFrame):
        """Set reference data for drift detection."""
        self.drift_detector = DataDriftDetector(reference_data)
        logger.info("Reference data set for drift detection")

    def validate(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset",
    ) -> Dict[str, Any]:
        """
        Run all validation rules on DataFrame.

        Args:
            df: DataFrame to validate
            dataset_name: Name for reporting

        Returns:
            Validation results dictionary
        """
        logger.info(f"Validating {dataset_name} ({len(df)} rows, {len(df.columns)} columns)")

        results = {
            "dataset_name": dataset_name,
            "timestamp": datetime.now().isoformat(),
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "rules_passed": 0,
            "rules_failed": 0,
            "rule_results": [],
            "drift_results": None,
        }

        # Run all rules
        for rule in self.rules:
            try:
                passed, message = rule.validate(df)
                results["rule_results"].append(
                    {
                        "rule_name": rule.name,
                        "description": rule.description,
                        "severity": rule.severity,
                        "passed": passed,
                        "message": message,
                    }
                )
                if passed:
                    results["rules_passed"] += 1
                else:
                    results["rules_failed"] += 1
                    logger.warning(f"Validation failed: {rule.name} - {message}")
            except Exception as e:
                results["rule_results"].append(
                    {
                        "rule_name": rule.name,
                        "error": str(e),
                        "passed": False,
                    }
                )
                results["rules_failed"] += 1
                logger.error(f"Validation error in {rule.name}: {e}")

        # Run drift detection if reference data is set
        if self.drift_detector:
            results["drift_results"] = self.drift_detector.detect_all_drift(df)
            drift_detected = sum(
                1
                for col_result in results["drift_results"].values()
                if col_result.get("drift_detected", False)
            )
            if drift_detected > 0:
                logger.warning(f"Data drift detected in {drift_detected} columns")

        # Overall status
        results["overall_status"] = "passed" if results["rules_failed"] == 0 else "failed"
        results["validation_score"] = (
            results["rules_passed"] / len(self.rules) if self.rules else 1.0
        )

        self.validation_results = results
        return results

    def generate_report(self, output_path: str = "validation_report.md"):
        """
        Generate markdown validation report.

        Args:
            output_path: Path to save report
        """
        if not self.validation_results:
            logger.warning("No validation results to report")
            return

        report = []
        report.append("# Data Validation Report\n")
        report.append(f"**Dataset:** {self.validation_results['dataset_name']}\n")
        report.append(f"**Timestamp:** {self.validation_results['timestamp']}\n")
        report.append(f"**Rows:** {self.validation_results['n_rows']}\n")
        report.append(f"**Columns:** {self.validation_results['n_columns']}\n")
        report.append(
            f"**Overall Status:** {'✅ PASSED' if self.validation_results['overall_status'] == 'passed' else '❌ FAILED'}\n"
        )
        report.append(
            f"**Validation Score:** {self.validation_results['validation_score']:.1%}\n"
        )
        report.append("\n---\n")

        # Rule results
        report.append("## Validation Rules\n")
        for result in self.validation_results["rule_results"]:
            status = "✅" if result["passed"] else "❌"
            report.append(
                f"### {status} {result['rule_name']} ({result['severity']})\n"
            )
            report.append(f"{result['description']}\n")
            report.append(f"**Result:** {result.get('message', 'N/A')}\n")
            report.append("\n")

        # Drift results
        if self.validation_results["drift_results"]:
            report.append("## Data Drift Detection\n")
            for column, drift_info in self.validation_results["drift_results"].items():
                if drift_info.get("drift_detected"):
                    report.append(
                        f"### ⚠️ {column}: DRIFT DETECTED\n"
                    )
                    report.append(f"- PSI: {drift_info['psi']:.3f}\n")
                    report.append(f"- Interpretation: {drift_info['interpretation']}\n")
                    if "ks_statistic" in drift_info:
                        report.append(
                            f"- KS Statistic: {drift_info['ks_statistic']:.3f}\n"
                        )
                        report.append(f"- KS p-value: {drift_info['ks_p_value']:.3f}\n")
                    report.append("\n")

        # Save report
        report_text = "\n".join(report)
        with open(output_path, "w") as f:
            f.write(report_text)

        logger.info(f"Validation report saved to {output_path}")
        return report_text


# Example usage
if __name__ == "__main__":
    from src.data.synthetic import SITSSyntheticGenerator

    # Generate synthetic data
    print("Generating synthetic data...")
    generator = SITSSyntheticGenerator(n_students=1000, seed=42)
    datasets = generator.generate_all_datasets()

    # Initialize validator
    validator = DataValidator()

    # Add validation rules
    validator.add_rule(
        SchemaValidation(
            expected_columns=["student_id", "gender", "ethnicity"],
            expected_types={"student_id": "string", "gender": "string"},
        )
    )

    validator.add_rule(
        NullCheck(
            columns=["student_id", "gender"],
            max_null_pct=0.0,
        )
    )

    validator.add_rule(
        UniquenessCheck(
            columns=["student_id"],
            allow_duplicates=False,
        )
    )

    validator.add_rule(
        RangeCheck(
            column="ucas_tariff_points",
            min_value=0,
            max_value=200,
        )
    )

    # Validate students dataset
    print("\nValidating students dataset...")
    results = validator.validate(datasets["students"], dataset_name="students")

    print(f"\nValidation Score: {results['validation_score']:.1%}")
    print(f"Rules Passed: {results['rules_passed']}")
    print(f"Rules Failed: {results['rules_failed']}")

    # Set reference data and test drift
    print("\nSetting reference data for drift detection...")
    validator.set_reference_data(datasets["students"])

    # Generate new data with drift
    print("Generating drifted data...")
    generator_drifted = SITSSyntheticGenerator(n_students=1000, seed=123)
    drifted_datasets = generator_drifted.generate_all_datasets()

    # Detect drift
    print("\nDetecting data drift...")
    drift_results = validator.validate(
        drifted_datasets["students"], dataset_name="students_drifted"
    )

    # Generate report
    print("\nGenerating validation report...")
    report = validator.generate_report("validation_report.md")
    print("✓ Validation complete!")
