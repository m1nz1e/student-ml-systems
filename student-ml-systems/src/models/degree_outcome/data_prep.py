"""
Data Preparation for Degree Outcome Prediction.

Creates features for multi-class classification of degree classification:
- Ordinal: Fail (0) < Third (1) < 2:2 (2) < 2:1 (3) < First (4)

Feature Groups:
- Temporal assessment features (year-by-year GPA, trends, subject performance)
- Engagement trajectory (attendance and VLE trends over time)
- Static features (entry qualifications, demographics, course difficulty)
- Derived indicators (early warning flags, credit-weighted averages)

CRITICAL: All features are computed from data available BEFORE the final year
to prevent future data leakage. Final year data is used only for the target.
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ordinal class mapping
DEGREE_CLASSIFICATION_MAP = {
    "Fail": 0,
    "Third": 1,
    "2:2": 2,
    "2:1": 3,
    "First": 4,
}
DEGREE_CLASS_NAMES = ["Fail", "Third", "2:2", "2:1", "First"]


class DegreeOutcomeFeatureEngineer:
    """
    Feature engineering for degree outcome prediction.

    Creates features from pre-final-year data only:
    - Year 1 and Year 2 assessment performance (temporal)
    - Engagement trajectories (attendance, VLE over time)
    - Static student and course characteristics
    - Derived early-warning and trend indicators

    The final year is reserved for the target variable only.
    """

    def __init__(self, n_years: int = 3):
        """
        Initialize feature engineer.

        Args:
            n_years: Number of years of study (e.g. 3 for BSc, 4 for MEng).
                     Year < n_years data is used for features; Year == n_years
                     is the target year.
        """
        self.n_years = n_years
        self.feature_names: List[str] = []
        self.class_names = DEGREE_CLASS_NAMES

    def engineer_features(
        self,
        assessments_df: pd.DataFrame,
        attendance_df: pd.DataFrame,
        vle_df: pd.DataFrame,
        students_df: pd.DataFrame,
        enrollments_df: pd.DataFrame,
        degree_outcomes_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Engineer features for degree outcome prediction.

        Only pre-final-year data is used for features. Final-year data
        is exclusively used to construct the target variable.

        Args:
            assessments_df: Assessment records (student_id, module_id, mark,
                           attempt, late_submission)
            attendance_df: Attendance records (student_id, week, status, attended)
            vle_df: VLE engagement records (student_id, week, logins, resources,
                    forum_posts, quiz_attempts, video_views)
            students_df: Student demographics (student_id, gender, ethnicity,
                        imd_decile, polar_quintile, etc.)
            enrollments_df: Enrollment records (student_id, course_id, academic_year)
            degree_outcomes_df: Target outcomes (student_id, final_classification,
                                weighted_gpa)

        Returns:
            Tuple of:
            - df: DataFrame with all engineered features plus student_id, target columns
            - X: Feature matrix of shape (n_samples, n_features)
            - y: Target vector of shape (n_samples,) with ordinal labels
                 0=Fail, 1=Third, 2=2:2, 3=2:1, 4=First
        """
        logger.info("Engineering degree outcome features...")

        # Validate inputs
        required = ["assessments_df", "students_df", "enrollments_df", "degree_outcomes_df"]
        for name in required:
            if locals()[name] is None or len(locals()[name]) == 0:
                raise ValueError(f"{name} is required and must be non-empty")

        # Start with degree outcomes (one row per student)
        df = degree_outcomes_df[["student_id", "final_classification", "weighted_gpa"]].copy()
        logger.info(f"Base: {len(df)} students with degree outcomes")

        # Merge enrollments for course context
        course_cols = ["student_id", "course_id"]
        if "academic_year" in enrollments_df.columns:
            course_cols.append("academic_year")
        df = df.merge(
            enrollments_df[course_cols].drop_duplicates(subset=["student_id"]),
            on="student_id",
            how="left",
        )

        # Merge student demographics
        df = df.merge(students_df, on="student_id", how="left")

        # === Build assessment features (pre-final year only) ===
        df = self._build_assessment_features(df, assessments_df)

        # === Build engagement trajectory features ===
        df = self._build_engagement_features(df, attendance_df, vle_df)

        # === Build static features ===
        df = self._build_static_features(df)

        # === Build derived / early-warning features ===
        df = self._build_derived_features(df)

        # === Create ordinal target ===
        df["target"] = df["final_classification"].map(DEGREE_CLASSIFICATION_MAP)
        missing_targets = df["target"].isna().sum()
        if missing_targets > 0:
            logger.warning(f"Dropping {missing_targets} rows with unrecognised classification")
            df = df.dropna(subset=["target"])

        # Select feature columns
        self.feature_names = [col for col in df.columns if col.startswith("feat_")]
        X = df[self.feature_names].fillna(0).infer_objects(copy=False).values.astype(np.float32)
        y = df["target"].values.astype(np.int32)

        logger.info(f"Features: {X.shape}, Target distribution: {np.bincount(y)}")
        logger.info(f"Created {len(self.feature_names)} features")

        return df, X, y

    # ------------------------------------------------------------------
    # Assessment Features (pre-final year only)
    # ------------------------------------------------------------------

    def _build_assessment_features(
        self,
        df: pd.DataFrame,
        assessments_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build assessment-based features from pre-final-year modules.

        We need to know which modules belong to which year.
        The synthetic data has module_id with format {course_id}_MOD{XX}.
        We estimate year from attempt patterns and module sequence position.
        """
        logger.info("Building assessment features...")

        # Merge assessments with student info to get course context
        student_course = df[["student_id", "course_id"]].drop_duplicates(subset=["student_id"])
        ass = assessments_df.merge(student_course, on="student_id", how="inner")

        if len(ass) == 0:
            logger.warning("No matching assessments found; using defaults")
            df = self._add_default_assessment_features(df)
            return df

        # Estimate module year from position within course
        # Module IDs are {course_id}_MOD{XX}; we use module position as year proxy
        # Format: course modules are sequential MOD01, MOD02, ... per year
        # Year 1: MOD01-MOD04, Year 2: MOD05-MOD08, Year 3: MOD09-MOD12
        def infer_module_year(module_id: str) -> int:
            try:
                parts = module_id.split("_MOD")
                if len(parts) < 2:
                    return 1
                mod_num = int(parts[-1])
                if mod_num <= 4:
                    return 1
                elif mod_num <= 8:
                    return 2
                elif mod_num <= 12:
                    return 3
                else:
                    return (mod_num - 1) // 4 + 1
            except (ValueError, IndexError):
                return 1

        ass["module_year"] = ass["module_id"].apply(infer_module_year)

        # Filter to pre-final year (exclude final year assessments)
        ass_pre = ass[ass["module_year"] < self.n_years].copy()

        # --- Cumulative GPA per year ---
        for year in range(1, self.n_years):
            year_data = ass_pre[ass_pre["module_year"] == year]
            year_gpa = (
                year_data.groupby("student_id")["mark"]
                .mean()
                .reset_index()
            )
            year_gpa.columns = ["student_id", f"feat_year_{year}_gpa"]
            df = df.merge(year_gpa, on="student_id", how="left")

        # --- Overall pre-final GPA (credit-weighted) ---
        if "credits" in ass_pre.columns:
            ass_pre["weighted_mark"] = ass_pre["mark"] * ass_pre["credits"]
            overall = (
                ass_pre.groupby("student_id")
                .agg(total_weighted=("weighted_mark", "sum"), total_credits=("credits", "sum"))
                .reset_index()
            )
            overall["feat_pre_final_gpa"] = overall["total_weighted"] / overall["total_credits"].clip(lower=1)
            df = df.merge(overall[["student_id", "feat_pre_final_gpa"]], on="student_id", how="left")
        else:
            # Equal weighting fallback
            overall = (
                ass_pre.groupby("student_id")["mark"]
                .mean()
                .reset_index()
            )
            overall.columns = ["student_id", "feat_pre_final_gpa"]
            df = df.merge(overall, on="student_id", how="left")

        # --- Mark trend (Year 2 vs Year 1) ---
        if f"feat_year_1_gpa" in df.columns and f"feat_year_2_gpa" in df.columns:
            df["feat_mark_trend_y2_vs_y1"] = df[f"feat_year_2_gpa"] - df[f"feat_year_1_gpa"]
            # Categorise trend
            df["feat_mark_trend_category"] = pd.cut(
                df["feat_mark_trend_y2_vs_y1"],
                bins=[-np.inf, -5, 5, np.inf],
                labels=["declining", "stable", "improving"],
            )
            df["feat_mark_trend_category"] = (
                df["feat_mark_trend_category"].cat.codes
            )

        # --- Best / worst subject marks ---
        subject_stats = (
            ass_pre.groupby("student_id")
            .agg(
                feat_best_mark=("mark", "max"),
                feat_worst_mark=("mark", "min"),
                feat_std_mark=("mark", "std"),
            )
            .reset_index()
        )
        df = df.merge(subject_stats, on="student_id", how="left")

        # --- Resit count and success rate ---
        resit_stats = self._compute_resit_features(ass_pre)
        df = df.merge(resit_stats, on="student_id", how="left")

        # --- Late submission count ---
        if "late_submission" in ass_pre.columns:
            late_stats = (
                ass_pre.groupby("student_id")
                .agg(
                    feat_late_submissions=("late_submission", "sum"),
                    feat_late_submission_rate=("late_submission", "mean"),
                )
                .reset_index()
            )
            df = df.merge(late_stats, on="student_id", how="left")

        # --- Credit-weighted average mark ---
        if "credits" in ass_pre.columns:
            credit_avg = (
                ass_pre.groupby("student_id")
                .apply(lambda g: (g["mark"] * g["credits"]).sum() / g["credits"].sum())
                .reset_index()
            )
            credit_avg.columns = ["student_id", "feat_credit_weighted_avg"]
            df = df.merge(credit_avg, on="student_id", how="left")
        else:
            credit_avg = (
                ass_pre.groupby("student_id")["mark"].mean().reset_index()
            )
            credit_avg.columns = ["student_id", "feat_credit_weighted_avg"]
            df = df.merge(credit_avg, on="student_id", how="left")

        return df

    def _compute_resit_features(self, ass_pre: pd.DataFrame) -> pd.DataFrame:
        """Compute resit count and pass rate from assessment attempts."""
        resits = ass_pre[ass_pre["attempt"] > 1]
        first_attempts = ass_pre[ass_pre["attempt"] == 1]

        # Count resits per student
        resit_count = (
            resits.groupby("student_id").size().reset_index(name="feat_resit_count")
        )

        # Resit success: passed after resit (mark >= 40)
        resit_passed = (
            resits[resits["mark"] >= 40]
            .groupby("student_id")
            .size()
            .reset_index(name="feat_resits_passed")
        )

        # Total attempts for resit rate
        total_attempts = (
            ass_pre.groupby("student_id")
            .size()
            .reset_index(name="feat_total_attempts")
        )

        result = total_attempts.merge(resit_count, on="student_id", how="left")
        result = result.merge(resit_passed, on="student_id", how="left")
        result["feat_resit_count"] = result["feat_resit_count"].fillna(0)
        result["feat_resits_passed"] = result["feat_resits_passed"].fillna(0)
        result["feat_resit_success_rate"] = (
            result["feat_resits_passed"] / result["feat_resit_count"].clip(lower=1)
        ).fillna(1.0)

        return result[["student_id", "feat_resit_count", "feat_resit_success_rate"]]

    def _add_default_assessment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add zeroed assessment features when no assessment data is available."""
        defaults = [
            "feat_year_1_gpa",
            "feat_year_2_gpa",
            "feat_pre_final_gpa",
            "feat_mark_trend_y2_vs_y1",
            "feat_mark_trend_category",
            "feat_best_mark",
            "feat_worst_mark",
            "feat_std_mark",
            "feat_resit_count",
            "feat_resit_success_rate",
            "feat_late_submissions",
            "feat_late_submission_rate",
            "feat_credit_weighted_avg",
        ]
        for col in defaults:
            if col not in df.columns:
                df[col] = 0.0
        return df

    # ------------------------------------------------------------------
    # Engagement Trajectory Features
    # ------------------------------------------------------------------

    def _build_engagement_features(
        self,
        df: pd.DataFrame,
        attendance_df: Optional[pd.DataFrame],
        vle_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Build engagement trajectory features from attendance and VLE data."""
        logger.info("Building engagement trajectory features...")

        df = self._build_attendance_features(df, attendance_df)
        df = self._build_vle_features(df, vle_df)

        return df

    def _build_attendance_features(
        self,
        df: pd.DataFrame,
        attendance_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Build attendance-based features aggregated per student."""
        if attendance_df is None or len(attendance_df) == 0:
            logger.info("No attendance data; using defaults")
            defaults = ["feat_attendance_rate", "feat_attendance_trend", "feat_absence_rate"]
            for col in defaults:
                if col not in df.columns:
                    df[col] = 0.0
            return df

        # Attendance rate per student
        att_agg = (
            attendance_df.groupby("student_id")
            .agg(
                feat_attendance_rate=("attended", "mean"),
                feat_total_absences=("attended", lambda x: (~x).sum()),
            )
            .reset_index()
        )

        # Absence rate (absent / total)
        att_counts = attendance_df.groupby("student_id").size().reset_index(name="feat_total_sessions")
        att_agg = att_agg.merge(att_counts, on="student_id", how="left")
        att_agg["feat_absence_rate"] = (
            att_agg["feat_total_absences"] / att_agg["feat_total_sessions"].clip(lower=1)
        )

        df = df.merge(att_agg[["student_id", "feat_attendance_rate", "feat_absence_rate"]], on="student_id", how="left")

        # Attendance trend (early vs late term)
        if "week" in attendance_df.columns:
            trend_data = []
            for student_id, group in attendance_df.groupby("student_id"):
                if len(group) < 4:
                    trend_data.append({"student_id": student_id, "feat_attendance_trend": 0.0})
                    continue
                first_half = group.head(len(group) // 2)["attended"].mean()
                second_half = group.tail(len(group) // 2)["attended"].mean()
                trend_data.append({
                    "student_id": student_id,
                    "feat_attendance_trend": second_half - first_half,
                })
            trend_df = pd.DataFrame(trend_data)
            df = df.merge(trend_df, on="student_id", how="left")

        return df

    def _build_vle_features(
        self,
        df: pd.DataFrame,
        vle_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Build VLE engagement features per student."""
        if vle_df is None or len(vle_df) == 0:
            logger.info("No VLE data; using defaults")
            defaults = [
                "feat_vle_avg_logins",
                "feat_vle_total_actions",
                "feat_vle_trend",
                "feat_vle_forum_activity",
            ]
            for col in defaults:
                if col not in df.columns:
                    df[col] = 0.0
            return df

        vle_agg = (
            vle_df.groupby("student_id")
            .agg(
                feat_vle_avg_logins=("logins", "mean"),
                feat_vle_total_logins=("logins", "sum"),
                feat_vle_total_resources=("resources_accessed", "sum"),
                feat_vle_total_forum_posts=("forum_posts", "sum"),
                feat_vle_total_actions=("total_actions", "mean"),
            )
            .reset_index()
        )

        # Normalise actions to 0-1 scale
        if vle_agg["feat_vle_total_actions"].max() > 0:
            vle_agg["feat_vle_total_actions"] = (
                vle_agg["feat_vle_total_actions"] / vle_agg["feat_vle_total_actions"].max()
            )
        if vle_agg["feat_vle_avg_logins"].max() > 0:
            vle_agg["feat_vle_avg_logins"] = (
                vle_agg["feat_vle_avg_logins"] / vle_agg["feat_vle_avg_logins"].max()
            )

        df = df.merge(
            vle_agg[["student_id", "feat_vle_avg_logins", "feat_vle_total_actions", "feat_vle_total_forum_posts"]],
            on="student_id",
            how="left",
        )

        # VLE engagement trend
        if "week" in vle_df.columns:
            trend_data = []
            for student_id, group in vle_df.groupby("student_id"):
                if len(group) < 4:
                    trend_data.append({"student_id": student_id, "feat_vle_trend": 0.0})
                    continue
                first_half = group.head(len(group) // 2)["total_actions"].mean()
                second_half = group.tail(len(group) // 2)["total_actions"].mean()
                trend_data.append({
                    "student_id": student_id,
                    "feat_vle_trend": second_half - first_half,
                })
            trend_df = pd.DataFrame(trend_data)
            df = df.merge(trend_df, on="student_id", how="left")

        # Forum activity indicator
        df["feat_vle_forum_activity"] = (df["feat_vle_total_forum_posts"] > 0).astype(float)

        return df

    # ------------------------------------------------------------------
    # Static Features
    # ------------------------------------------------------------------

    def _build_static_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build static/demographic and course-level features."""
        logger.info("Building static features...")

        # --- Entry tariff points ---
        if "ucas_tariff_points" in df.columns:
            df["feat_entry_tariff"] = df["ucas_tariff_points"].fillna(
                df["ucas_tariff_points"].median()
            )
            df["feat_entry_tariff_normalized"] = df["feat_entry_tariff"] / 168.0
        else:
            df["feat_entry_tariff"] = 100.0
            df["feat_entry_tariff_normalized"] = 100.0 / 168.0

        # --- Gender ---
        if "gender" in df.columns:
            df["feat_gender_female"] = (df["gender"] == "Female").astype(float)
            df["feat_gender_male"] = (df["gender"] == "Male").astype(float)
        else:
            df["feat_gender_female"] = 0.0
            df["feat_gender_male"] = 0.0

        # --- Ethnicity (grouped for sparse representation) ---
        if "ethnicity" in df.columns:
            # Group minority ethnicities to reduce sparsity
            ethnicity_map = {
                "White British": "white_british",
                "White Irish": "white_other",
                "White Other": "white_other",
                "Mixed": "mixed",
                "Asian/Asian British": "asian",
                "Black/Black British": "black",
                "Other Ethnic Group": "other",
                "Not disclosed": "not_disclosed",
            }
            df["ethnicity_grouped"] = df["ethnicity"].map(ethnicity_map).fillna("unknown")
            for group in ["white_british", "asian", "black", "white_other", "mixed"]:
                df[f"feat_ethnicity_{group}"] = (df["ethnicity_grouped"] == group).astype(float)
        else:
            for group in ["white_british", "asian", "black", "white_other", "mixed"]:
                df[f"feat_ethnicity_{group}"] = 0.0

        # --- IMD decile (socioeconomic disadvantage proxy) ---
        if "imd_decile" in df.columns:
            df["feat_imd_deprivation"] = df["imd_decile"].fillna(5) / 10.0
        else:
            df["feat_imd_deprivation"] = 0.5

        # --- POLAR4 quintile (participation in higher education) ---
        if "polar_quintile" in df.columns:
            df["feat_low_participation"] = (df["polar_quintile"] == 1).astype(float)
        else:
            df["feat_low_participation"] = 0.0

        # --- Care leaver ---
        if "care_leaver" in df.columns:
            df["feat_care_leaver"] = df["care_leaver"].astype(float)
        else:
            df["feat_care_leaver"] = 0.0

        # --- Disability ---
        if "disability" in df.columns:
            df["feat_disability"] = df["disability"].astype(float)
        else:
            df["feat_disability"] = 0.0

        # --- First generation university student ---
        if "first_generation_uni" in df.columns:
            df["feat_first_generation"] = df["first_generation_uni"].astype(float)
        else:
            df["feat_first_generation"] = 0.0

        # --- Course difficulty (cohort average mark from modules) ---
        if "course_id" in df.columns:
            # Use assessments to estimate course difficulty
            if len(df) > 0:
                df["feat_course_difficulty"] = 0.5  # default
            else:
                df["feat_course_difficulty"] = 0.5

        # --- Course length ---
        if "course_length_years" in df.columns:
            df["feat_course_length"] = df["course_length_years"].fillna(3)
        else:
            df["feat_course_length"] = 3.0

        return df

    # ------------------------------------------------------------------
    # Derived / Early-Warning Features
    # ------------------------------------------------------------------

    def _build_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build derived features: first-year GPA predictor, early warning flags."""
        logger.info("Building derived features...")

        # --- First-year GPA as the strongest predictor ---
        # Already have feat_year_1_gpa; normalise it
        if "feat_year_1_gpa" in df.columns:
            df["feat_year_1_gpa_normalized"] = df["feat_year_1_gpa"] / 100.0
        else:
            df["feat_year_1_gpa_normalized"] = 0.0

        # --- Early warning: low first-year GPA ---
        if "feat_year_1_gpa" in df.columns:
            df["feat_early_warning_low_y1"] = (df["feat_year_1_gpa"] < 50).astype(float)
        else:
            df["feat_early_warning_low_y1"] = 0.0

        # --- Early warning: declining marks ---
        if "feat_mark_trend_y2_vs_y1" in df.columns:
            df["feat_early_warning_decline"] = (df["feat_mark_trend_y2_vs_y1"] < -10).astype(float)
        else:
            df["feat_early_warning_decline"] = 0.0

        # --- Early warning: high absence ---
        if "feat_absence_rate" in df.columns:
            df["feat_early_warning_absence"] = (df["feat_absence_rate"] > 0.2).astype(float)
        else:
            df["feat_early_warning_absence"] = 0.0

        # --- Combined early warning score ---
        warning_cols = [
            "feat_early_warning_low_y1",
            "feat_early_warning_decline",
            "feat_early_warning_absence",
        ]
        existing_warning_cols = [c for c in warning_cols if c in df.columns]
        if existing_warning_cols:
            df["feat_early_warning_score"] = df[existing_warning_cols].sum(axis=1)
        else:
            df["feat_early_warning_score"] = 0.0

        # --- GPA trajectory direction ---
        if "feat_mark_trend_y2_vs_y1" in df.columns:
            df["feat_gpa_trajectory"] = np.sign(df["feat_mark_trend_y2_vs_y1"].fillna(0))
        else:
            df["feat_gpa_trajectory"] = 0.0

        # --- Engagement-investment ratio ---
        if "feat_vle_total_actions" in df.columns and "feat_attendance_rate" in df.columns:
            df["feat_engagement_index"] = (
                df["feat_vle_total_actions"].fillna(0) * 0.5
                + df["feat_attendance_rate"].fillna(0.5) * 0.5
            )
        else:
            df["feat_engagement_index"] = 0.5

        return df

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_feature_names(self) -> List[str]:
        """Return the list of engineered feature names."""
        return self.feature_names

    def get_class_names(self) -> List[str]:
        """Return the ordinal class names."""
        return self.class_names


# Example usage
if __name__ == "__main__":
    from src.data.synthetic import SITSSyntheticGenerator

    # Generate synthetic data
    print("Generating synthetic data...")
    generator = SITSSyntheticGenerator(n_students=500, n_courses=50, seed=42)
    datasets = generator.generate_all_datasets()

    # Generate degree outcomes
    print("\nGenerating degree outcomes...")
    outcomes = generator.generate_degree_outcomes(
        assessments_df=datasets["assessments"],
        students_df=datasets["students"],
        courses_df=datasets["courses"],
        enrollments_df=datasets["enrollments"],
    )
    print(f"Generated {len(outcomes)} outcomes")
    print(outcomes["final_classification"].value_counts())

    # Engineer features
    print("\nEngineering degree outcome features...")
    engineer = DegreeOutcomeFeatureEngineer(n_years=3)
    df, X, y = engineer.engineer_features(
        assessments_df=datasets["assessments"],
        attendance_df=datasets["attendance"],
        vle_df=datasets["vle_engagement"],
        students_df=datasets["students"],
        enrollments_df=datasets["enrollments"],
        degree_outcomes_df=outcomes,
    )

    print(f"\nFeature matrix: {X.shape}")
    print(f"Target vector: {y.shape}")
    print(f"Target distribution (ordinal): {np.bincount(y)}")
    print(f"Class names: {engineer.get_class_names()}")
    print(f"\nFeature names ({len(engineer.get_feature_names())}):")
    for fn in engineer.get_feature_names():
        print(f"  - {fn}")

    print("\n✓ Degree outcome feature engineering complete!")
