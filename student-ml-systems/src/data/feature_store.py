"""
Feature Store for Student ML Systems.

Implements feature engineering, transformations, and train/test splitting
for all three ML systems (Course Recommender, Enrollment Yield, Early Warning).

Feature Groups:
- Student Demographics
- Prior Qualifications
- Course Characteristics
- Engagement Metrics (VLE, Attendance)
- Performance Metrics (Assessments)
- Contextual Indicators (IMD, POLAR)
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    TimeSeriesSplit,
    GroupKFold,
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureRegistry:
    """
    Registry for tracking feature metadata.

    Stores feature names, types, descriptions, and statistics.
    """

    def __init__(self):
        self.features: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        dtype: str,
        description: str,
        group: str,
        transformation: Optional[str] = None,
        cardinality: Optional[int] = None,
    ):
        """
        Register a feature in the registry.

        Args:
            name: Feature name
            dtype: Data type (numeric, categorical, binary, datetime)
            description: Human-readable description
            group: Feature group (demographics, qualifications, etc.)
            transformation: Applied transformation (standardize, encode, etc.)
            cardinality: Number of unique values (for categorical)
        """
        self.features[name] = {
            "dtype": dtype,
            "description": description,
            "group": group,
            "transformation": transformation,
            "cardinality": cardinality,
        }

    def get_features_by_group(self, group: str) -> List[str]:
        """Get all feature names in a group."""
        return [
            name for name, meta in self.features.items() if meta["group"] == group
        ]

    def get_all_features(self) -> List[str]:
        """Get all feature names."""
        return list(self.features.keys())

    def to_dataframe(self) -> pd.DataFrame:
        """Export registry as DataFrame."""
        records = []
        for name, meta in self.features.items():
            record = {"feature_name": name, **meta}
            records.append(record)
        return pd.DataFrame(records)

    def save(self, path: str):
        """Save registry to CSV."""
        self.to_dataframe().to_csv(path, index=False)
        logger.info(f"Feature registry saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FeatureRegistry":
        """Load registry from CSV."""
        df = pd.read_csv(path)
        registry = cls()
        for _, row in df.iterrows():
            registry.register(
                name=row["feature_name"],
                dtype=row["dtype"],
                description=row["description"],
                group=row["group"],
                transformation=row.get("transformation"),
                cardinality=row.get("cardinality"),
            )
        return registry


class FeatureStore:
    """
    Central feature store for student ML systems.

    Handles:
    - Feature engineering
    - Transformations (scaling, encoding)
    - Train/test splitting
    - Feature persistence
    """

    def __init__(self, registry: Optional[FeatureRegistry] = None):
        """
        Initialize feature store.

        Args:
            registry: Optional FeatureRegistry instance
        """
        self.registry = registry or FeatureRegistry()
        self.encoders: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_stats: Dict[str, Dict[str, Any]] = {}

    def engineer_student_features(
        self,
        students_df: pd.DataFrame,
        qualifications_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Engineer student-level features.

        Args:
            students_df: Student demographics DataFrame
            qualifications_df: Prior qualifications DataFrame

        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering student features...")

        # Merge dataframes
        df = students_df.merge(qualifications_df, on="student_id", how="left")

        # === Demographics ===
        # Gender encoding
        df["gender_encoded"] = (df["gender"] == "Female").astype(int)
        self.registry.register(
            "gender_encoded", "binary", "Gender (1=Female, 0=Other)", "demographics"
        )

        # Ethnicity (one-hot encoded later)
        ethnicity_dummies = pd.get_dummies(df["ethnicity"], prefix="ethnicity")
        df = pd.concat([df, ethnicity_dummies], axis=1)
        for col in ethnicity_dummies.columns:
            self.registry.register(
                col, "categorical", f"Ethnicity: {col}", "demographics", cardinality=2
            )

        # === Socioeconomic Indicators ===
        # IMD decile (inverse - higher = more deprived)
        df["imd_inverse"] = 11 - df["imd_decile"]
        self.registry.register(
            "imd_inverse",
            "numeric",
            "IMD decile (inverse: higher=more deprived)",
            "socioeconomic",
            transformation="inverse",
        )

        # POLAR4 (quintile 1 = lowest participation)
        df["low_participation_area"] = (df["polar_quintile"] == 1).astype(int)
        self.registry.register(
            "low_participation_area",
            "binary",
            "From low participation area (POLAR Q1)",
            "socioeconomic",
        )

        # Contextual indicators
        df["contextual_score"] = (
            df["care_leaver"].astype(int) * 3
            + (df["first_generation_uni"].astype(int)) * 2
            + (df["low_participation_area"].astype(int)) * 1
        )
        self.registry.register(
            "contextual_score",
            "numeric",
            "Composite contextual indicators score",
            "socioeconomic",
            transformation="weighted_sum",
        )

        # === Qualifications ===
        # Qualification type (one-hot)
        qual_dummies = pd.get_dummies(df["qualification_type"], prefix="qual")
        df = pd.concat([df, qual_dummies], axis=1)
        for col in qual_dummies.columns:
            self.registry.register(
                col, "categorical", f"Qualification: {col}", "qualifications", cardinality=2
            )

        # UCAS tariff points (already numeric)
        self.registry.register(
            "ucas_tariff_points",
            "numeric",
            "UCAS tariff points from qualifications",
            "qualifications",
        )

        # Predicted grade flag
        df["predicted_grade"] = df["predicted_grade"].astype(int)
        self.registry.register(
            "predicted_grade",
            "binary",
            "Has predicted grade (vs achieved)",
            "qualifications",
        )

        # Grade bands
        df["grade_band"] = pd.cut(
            df["ucas_tariff_points"],
            bins=[0, 80, 112, 128, 144, 168, 200],
            labels=["C", "BC", "B", "AB", "A", "A*"],
        )
        grade_dummies = pd.get_dummies(df["grade_band"], prefix="grade")
        df = pd.concat([df, grade_dummies], axis=1)

        return df

    def engineer_course_features(self, courses_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer course-level features.

        Args:
            courses_df: Course database DataFrame

        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering course features...")

        df = courses_df.copy()

        # === Course Characteristics ===
        # Department (one-hot)
        dept_dummies = pd.get_dummies(df["department"], prefix="dept")
        df = pd.concat([df, dept_dummies], axis=1)
        for col in dept_dummies.columns:
            self.registry.register(
                col, "categorical", f"Department: {col}", "course", cardinality=2
            )

        # Entry tariff (numeric)
        self.registry.register(
            "entry_tariff", "numeric", "UCAS tariff entry requirement", "course"
        )

        # Entry tariff bands
        df["tariff_band"] = pd.cut(
            df["entry_tariff"],
            bins=[0, 96, 120, 144, 168, 200],
            labels=["Low", "Medium", "High", "Very High"],
        )
        tariff_dummies = pd.get_dummies(df["tariff_band"], prefix="tariff")
        df = pd.concat([df, tariff_dummies], axis=1)

        # Course length
        df["sandwich_year"] = (df["course_length_years"] == 4).astype(int)
        self.registry.register(
            "sandwich_year", "binary", "Has sandwich/placement year", "course"
        )

        # Assessment type
        df["coursework_heavy"] = (df["coursework_weight_pct"] > 60).astype(int)
        self.registry.register(
            "coursework_heavy",
            "binary",
            "Coursework-heavy assessment (>60%)",
            "course",
        )

        # Outcomes
        self.registry.register(
            "employment_rate_15m",
            "numeric",
            "Employment rate 15 months after graduation",
            "course",
        )

        self.registry.register(
            "satisfaction_score",
            "numeric",
            "Student satisfaction score (NSS-style)",
            "course",
        )

        df["accredited"] = df["accredited"].astype(int)
        self.registry.register(
            "accredited", "binary", "Professionally accredited", "course"
        )

        return df

    def engineer_engagement_features(
        self,
        attendance_df: pd.DataFrame,
        vle_df: pd.DataFrame,
        agg_level: str = "student",
    ) -> pd.DataFrame:
        """
        Engineer engagement features from time-series data.

        Args:
            attendance_df: Attendance records DataFrame
            vle_df: VLE engagement DataFrame
            agg_level: Aggregation level ('student' or 'student-week')

        Returns:
            DataFrame with aggregated engagement features
        """
        logger.info("Engineering engagement features...")

        # === Attendance Features ===
        attendance_agg = attendance_df.groupby("student_id").agg(
            attendance_rate=("attended", "mean"),
            attendance_std=("attended", "std"),
            total_weeks=("week", "count"),
            weeks_present=("attended", "sum"),
            authorised_absences=("status", lambda x: (x == "Authorised Absence").sum()),
            medical_absences=("status", lambda x: (x == "Medical").sum()),
        ).reset_index()

        # Fill NaN std with 0
        attendance_agg["attendance_std"] = attendance_agg["attendance_std"].fillna(0)

        # Attendance trend (first half vs second half)
        attendance_df_copy = attendance_df.copy()
        attendance_df_copy["first_half"] = (attendance_df_copy["week"] <= 15).astype(int)
        first_half = (
            attendance_df_copy[attendance_df_copy["first_half"] == 1]
            .groupby("student_id")["attended"]
            .mean()
        )
        second_half = (
            attendance_df_copy[attendance_df_copy["first_half"] == 0]
            .groupby("student_id")["attended"]
            .mean()
        )
        attendance_agg["attendance_trend"] = (
            second_half - first_half
        ).fillna(0).values

        for col in attendance_agg.columns:
            if col != "student_id":
                self.registry.register(
                    col, "numeric", f"Attendance: {col}", "engagement"
                )

        # === VLE Engagement Features ===
        vle_agg = vle_df.groupby("student_id").agg(
            total_logins=("logins", "sum"),
            total_resources=("resources_accessed", "sum"),
            total_forum_posts=("forum_posts", "sum"),
            total_quiz_attempts=("quiz_attempts", "sum"),
            total_video_views=("video_views", "sum"),
            total_actions=("total_actions", "sum"),
            avg_logins_per_week=("logins", "mean"),
            avg_resources_per_week=("resources_accessed", "mean"),
        ).reset_index()

        # VLE trend (first half vs second half)
        vle_df_copy = vle_df.copy()
        vle_df_copy["first_half"] = (vle_df_copy["week"] <= 15).astype(int)
        vle_first = (
            vle_df_copy[vle_df_copy["first_half"] == 1]
            .groupby("student_id")["total_actions"]
            .mean()
        )
        vle_second = (
            vle_df_copy[vle_df_copy["first_half"] == 0]
            .groupby("student_id")["total_actions"]
            .mean()
        )
        vle_agg["vle_engagement_trend"] = (
            vle_second - vle_first
        ).fillna(0).values

        # Engagement consistency (std dev of weekly actions)
        vle_consistency = vle_df.groupby("student_id")["total_actions"].std().reset_index()
        vle_consistency.columns = ["student_id", "vle_consistency"]
        vle_consistency["vle_consistency"] = vle_consistency["vle_consistency"].fillna(0)
        vle_agg = vle_agg.merge(vle_consistency, on="student_id", how="left")

        for col in vle_agg.columns:
            if col != "student_id":
                self.registry.register(
                    col, "numeric", f"VLE Engagement: {col}", "engagement"
                )

        # Merge attendance and VLE
        engagement_df = attendance_agg.merge(vle_agg, on="student_id", how="outer")

        return engagement_df

    def engineer_performance_features(
        self,
        assessments_df: pd.DataFrame,
        modules_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Engineer performance features from assessment data.

        Args:
            assessments_df: Assessment records DataFrame
            modules_df: Module database DataFrame

        Returns:
            DataFrame with aggregated performance features
        """
        logger.info("Engineering performance features...")

        # Merge with module info
        assessments_with_modules = assessments_df.merge(
            modules_df[["module_id", "course_id", "year", "credits"]],
            on="module_id",
            how="left",
        )

        # Aggregate by student
        perf_agg = assessments_with_modules.groupby("student_id").agg(
            avg_mark=("mark", "mean"),
            std_mark=("mark", "std"),
            min_mark=("mark", "min"),
            max_mark=("mark", "max"),
            total_assessments=("mark", "count"),
            submitted_count=("submitted", "sum"),
            late_submission_count=("late_submission", "sum"),
            resit_count=("attempt", lambda x: (x > 1).sum()),
        ).reset_index()

        # Fill NaN std with 0
        perf_agg["std_mark"] = perf_agg["std_mark"].fillna(0)

        # Submission rate
        perf_agg["submission_rate"] = (
            perf_agg["submitted_count"] / perf_agg["total_assessments"]
        )

        # Late submission rate
        perf_agg["late_submission_rate"] = (
            perf_agg["late_submission_count"] / perf_agg["total_assessments"]
        )

        # Resit rate
        perf_agg["resit_rate"] = perf_agg["resit_count"] / perf_agg["total_assessments"]

        # Pass rate (mark >= 40%)
        assessments_with_modules["passed"] = (
            assessments_with_modules["mark"] >= 40
        ).astype(int)
        pass_agg = (
            assessments_with_modules.groupby("student_id")["passed"]
            .mean()
            .reset_index()
        )
        pass_agg.columns = ["student_id", "pass_rate"]
        perf_agg = perf_agg.merge(pass_agg, on="student_id", how="left")

        # Performance by year
        for year in [1, 2, 3]:
            year_data = assessments_with_modules[
                assessments_with_modules["year"] == year
            ]
            if len(year_data) > 0:
                year_avg = year_data.groupby("student_id")["mark"].mean().reset_index()
                year_avg.columns = ["student_id", f"year{year}_avg_mark"]
                perf_agg = perf_agg.merge(year_avg, on="student_id", how="left")

        # Performance trend (improvement over time)
        if "year1_avg_mark" in perf_agg.columns and "year3_avg_mark" in perf_agg.columns:
            perf_agg["performance_trend"] = (
                perf_agg["year3_avg_mark"] - perf_agg["year1_avg_mark"]
            ).fillna(0)

        for col in perf_agg.columns:
            if col != "student_id":
                self.registry.register(
                    col, "numeric", f"Performance: {col}", "performance"
                )

        return perf_agg

    def create_target_variables(
        self,
        enrollments_df: pd.DataFrame,
        students_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create target variables for ML models.

        Args:
            enrollments_df: Enrollment records
            students_df: Student demographics

        Returns:
            DataFrame with target variables
        """
        logger.info("Creating target variables...")

        df = enrollments_df.copy()

        # === Course Recommender Target ===
        # Satisfaction proxy (retained + completed with good classification)
        df["high_satisfaction"] = (
            (df["retained_year2"] == 1)
            & (
                df["final_classification"].isin(["First", "2:1", "Completed"])
            )
        ).astype(int)
        self.registry.register(
            "high_satisfaction",
            "binary",
            "High satisfaction proxy (retained + good outcome)",
            "targets",
        )

        # === Enrollment Yield Target ===
        # Already enrolled = accepted offer
        df["accepted_offer"] = 1  # All records are enrolled students
        # For real data, would need offer-holder data
        self.registry.register(
            "accepted_offer", "binary", "Accepted offer (enrolled)", "targets"
        )

        # === Early Warning Target ===
        # At-risk: withdrawn or not retained
        df["at_risk"] = (
            (df["enrollment_status"] == "Withdrawn") | (df["retained_year2"] == 0)
        ).astype(int)
        self.registry.register(
            "at_risk", "binary", "At-risk student (withdrawn or not retained)", "targets"
        )

        # === Degree Classification Target ===
        classification_map = {
            "First": 4,
            "2:1": 3,
            "2:2": 2,
            "Third": 1,
            "Pass": 0,
            None: -1,
        }
        df["classification_numeric"] = df["final_classification"].map(classification_map)
        self.registry.register(
            "classification_numeric",
            "ordinal",
            "Degree classification (numeric)",
            "targets",
        )

        return df

    def apply_transformations(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
        numeric_cols: List[str],
        fit: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply scaling and encoding transformations.

        Args:
            df: Input DataFrame
            categorical_cols: List of categorical column names
            numeric_cols: List of numeric column names
            fit: Whether to fit transformers (True for train, False for test)

        Returns:
            Tuple of (transformed DataFrame, metadata dict)
        """
        logger.info(f"Applying transformations (fit={fit})...")

        df_transformed = df.copy()
        metadata = {"encoders": {}, "scalers": {}}

        # === Encode Categorical Variables ===
        for col in categorical_cols:
            if fit:
                encoder = LabelEncoder()
                df_transformed[col + "_encoded"] = encoder.fit_transform(
                    df_transformed[col].astype(str)
                )
                self.encoders[col] = encoder
                metadata["encoders"][col] = encoder
            else:
                if col in self.encoders:
                    # Handle unseen categories
                    df_transformed[col + "_encoded"] = df_transformed[col].apply(
                        lambda x: self.encoders[col].transform([x])[0]
                        if x in self.encoders[col].classes_
                        else -1
                    )

        # === Scale Numeric Variables ===
        for col in numeric_cols:
            if fit:
                scaler = StandardScaler()
                df_transformed[col + "_scaled"] = scaler.fit_transform(
                    df_transformed[[col]]
                )
                self.scalers[col] = scaler
                metadata["scalers"][col] = scaler
            else:
                if col in self.scalers:
                    df_transformed[col + "_scaled"] = self.scalers[col].transform(
                        df_transformed[[col]]
                    )

        return df_transformed, metadata

    def create_train_test_splits(
        self,
        df: pd.DataFrame,
        target_col: str,
        split_strategy: str = "stratified",
        test_size: float = 0.2,
        n_splits: int = 5,
        groups_col: Optional[str] = None,
        random_state: int = 42,
    ) -> Union[
        Tuple[pd.DataFrame, pd.DataFrame],
        List[Tuple[pd.DataFrame, pd.DataFrame]],
    ]:
        """
        Create train/test splits with specified strategy.

        Args:
            df: Input DataFrame
            target_col: Target column name
            split_strategy: One of ['random', 'stratified', 'timeseries', 'grouped']
            test_size: Test set size (for random/stratified)
            n_splits: Number of splits (for CV strategies)
            groups_col: Group column (for grouped split)
            random_state: Random seed

        Returns:
            Train/test DataFrames or list of fold pairs

        Raises:
            ValueError: If invalid split_strategy
        """
        logger.info(f"Creating {split_strategy} train/test splits...")

        if split_strategy == "random":
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=random_state
            )
            return train_df, test_df

        elif split_strategy == "stratified":
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                stratify=df[target_col],
                random_state=random_state,
            )
            return train_df, test_df

        elif split_strategy == "timeseries":
            # Sort by time (assume 'date' or 'year' column exists)
            time_col = "enrollment_date" if "enrollment_date" in df.columns else df.columns[0]
            df_sorted = df.sort_values(time_col)
            split_idx = int(len(df_sorted) * (1 - test_size))
            train_df = df_sorted.iloc[:split_idx]
            test_df = df_sorted.iloc[split_idx:]
            return train_df, test_df

        elif split_strategy == "grouped":
            if groups_col is None:
                raise ValueError("groups_col required for grouped split")

            gkf = GroupKFold(n_splits=n_splits)
            splits = []
            for train_idx, test_idx in gkf.split(df, df[target_col], df[groups_col]):
                train_df = df.iloc[train_idx]
                test_df = df.iloc[test_idx]
                splits.append((train_df, test_df))
            return splits

        else:
            raise ValueError(
                f"Invalid split_strategy: {split_strategy}. "
                "Must be one of ['random', 'stratified', 'timeseries', 'grouped']"
            )

    def compute_feature_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Compute statistics for all features.

        Args:
            df: DataFrame with features

        Returns:
            Dictionary of feature statistics
        """
        logger.info("Computing feature statistics...")

        stats = {}

        for col in df.columns:
            if df[col].dtype in ["int64", "float64"]:
                stats[col] = {
                    "dtype": str(df[col].dtype),
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "null_count": int(df[col].isnull().sum()),
                    "null_pct": float(df[col].isnull().mean() * 100),
                }
            else:
                stats[col] = {
                    "dtype": str(df[col].dtype),
                    "unique_count": int(df[col].nunique()),
                    "null_count": int(df[col].isnull().sum()),
                    "null_pct": float(df[col].isnull().mean() * 100),
                }

        self.feature_stats = stats
        return stats

    def save_feature_stats(self, path: str):
        """Save feature statistics to JSON."""
        import json

        with open(path, "w") as f:
            json.dump(self.feature_stats, f, indent=2, default=str)
        logger.info(f"Feature statistics saved to {path}")


# Example usage
if __name__ == "__main__":
    from src.data.synthetic import SITSSyntheticGenerator

    # Generate synthetic data
    print("Generating synthetic data...")
    generator = SITSSyntheticGenerator(n_students=1000, n_courses=50, seed=42)
    datasets = generator.generate_all_datasets()

    # Initialize feature store
    feature_store = FeatureStore()

    # Engineer features
    print("\nEngineering student features...")
    student_features = feature_store.engineer_student_features(
        datasets["students"], datasets["qualifications"]
    )
    print(f"  Shape: {student_features.shape}")
    print(f"  Features: {len(feature_store.registry.get_all_features())}")

    print("\nEngineering course features...")
    course_features = feature_store.engineer_course_features(datasets["courses"])
    print(f"  Shape: {course_features.shape}")

    print("\nEngineering engagement features...")
    engagement_features = feature_store.engineer_engagement_features(
        datasets["attendance"], datasets["vle_engagement"]
    )
    print(f"  Shape: {engagement_features.shape}")

    print("\nEngineering performance features...")
    performance_features = feature_store.engineer_performance_features(
        datasets["assessments"], datasets["modules"]
    )
    print(f"  Shape: {performance_features.shape}")

    print("\nCreating target variables...")
    targets = feature_store.create_target_variables(
        datasets["enrollments"], datasets["students"]
    )
    print(f"  Shape: {targets.shape}")

    # Export feature registry
    print("\nExporting feature registry...")
    feature_store.registry.save("feature_registry.csv")
    print(f"  Total features registered: {len(feature_store.registry.features)}")

    # Compute statistics
    print("\nComputing feature statistics...")
    stats = feature_store.compute_feature_statistics(student_features)
    feature_store.save_feature_stats("feature_statistics.json")

    print("\n✓ Feature engineering complete!")
    print(f"  Student features: {student_features.shape}")
    print(f"  Course features: {course_features.shape}")
    print(f"  Engagement features: {engagement_features.shape}")
    print(f"  Performance features: {performance_features.shape}")
    print(f"  Targets: {targets.shape}")
