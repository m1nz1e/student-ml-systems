"""
Data Preparation for Early Warning System.

Creates time-series features from:
- VLE engagement (weekly logins, resources, forum posts)
- Attendance records (weekly attendance rate)
- Assessment marks (temporal sequence)
- Contextual factors (static features)

Output: Sequential data suitable for LSTM and survival analysis models.
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyWarningFeatureEngineer:
    """
    Feature engineering for early warning system.

    Creates:
    - Time-series sequences (VLE, attendance, assessments)
    - Aggregated features (trends, consistency, recent performance)
    - Static features (demographics, prior attainment)
    - Target variables (dropout, failure, withdrawal)
    """

    def __init__(
        self,
        sequence_length: int = 12,  # Weeks of history
        prediction_horizon: int = 4,  # Weeks ahead to predict
        target_type: str = "dropout",  # 'dropout', 'failure', or 'both'
        min_weeks: int = 4,  # Minimum weeks of data required
    ):
        """
        Initialize feature engineer.

        Args:
            sequence_length: Number of weeks in sequence
            prediction_horizon: Weeks ahead to predict event
            target_type: Type of event to predict
            min_weeks: Minimum weeks of data required
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.target_type = target_type
        self.min_weeks = min_weeks

        self.feature_names: List[str] = []
        self.static_feature_names: List[str] = []
        self.sequence_feature_names: List[str] = []

    def create_sequences(
        self,
        vle_df: pd.DataFrame,
        attendance_df: pd.DataFrame,
        assessments_df: pd.DataFrame,
        students_df: pd.DataFrame,
        enrollments_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Create sequential datasets for LSTM model.

        Args:
            vle_df: VLE engagement data (student_id, week, metrics)
            attendance_df: Attendance data (student_id, week, status)
            assessments_df: Assessment data (student_id, date, mark)
            students_df: Student demographics
            enrollments_df: Enrollment outcomes

        Returns:
            Tuple of (sequences, static_features, targets, metadata)
        """
        logger.info("Creating early warning sequences...")

        # Get unique students with sufficient data
        student_weeks = self._get_student_weeks(vle_df, attendance_df)
        valid_students = student_weeks[student_weeks >= self.min_weeks].index.tolist()

        logger.info(f"Found {len(valid_students)} students with >= {self.min_weeks} weeks of data")

        sequences = []
        static_features = []
        targets = []
        student_ids = []
        week_indices = []

        for student_id in valid_students:
            # Create sequence for this student
            seq = self._create_student_sequence(
                student_id, vle_df, attendance_df, assessments_df
            )

            if seq is None or len(seq) < self.sequence_length:
                continue

            # Get static features
            static = self._get_static_features(student_id, students_df, enrollments_df)

            # Get target (dropout/failure in next prediction_horizon weeks)
            target = self._get_target(
                student_id, enrollments_df, self.prediction_horizon
            )

            if target is None:
                continue

            sequences.append(seq)
            static_features.append(static)
            targets.append(target)
            student_ids.append(student_id)
            week_indices.append(len(seq))

        # Convert to numpy arrays
        X_seq = np.array(sequences)
        X_static = np.array(static_features)
        y = np.array(targets)

        metadata = {
            "student_ids": student_ids,
            "sequence_lengths": week_indices,
            "n_students": len(student_ids),
            "sequence_length": self.sequence_length,
            "positive_rate": y.mean(),
        }

        logger.info(f"Created sequences: {X_seq.shape}")
        logger.info(f"Static features: {X_static.shape}")
        logger.info(f"Targets: {len(y)} (positive rate: {y.mean():.2%})")

        return X_seq, X_static, y, metadata

    def _get_student_weeks(
        self, vle_df: pd.DataFrame, attendance_df: pd.DataFrame
    ) -> pd.Series:
        """Get number of weeks of data per student."""
        vle_weeks = vle_df.groupby("student_id")["week"].nunique()
        attendance_weeks = attendance_df.groupby("student_id")["week"].nunique()

        # Combine (max of either source)
        all_students = vle_weeks.index.union(attendance_weeks.index)
        weeks = pd.Series(index=all_students, dtype=int)

        for student_id in all_students:
            vle_count = vle_weeks.get(student_id, 0)
            att_count = attendance_weeks.get(student_id, 0)
            weeks[student_id] = max(vle_count, att_count)

        return weeks

    def _create_student_sequence(
        self,
        student_id: str,
        vle_df: pd.DataFrame,
        attendance_df: pd.DataFrame,
        assessments_df: pd.DataFrame,
    ) -> Optional[np.ndarray]:
        """Create time-series sequence for one student."""
        # VLE features per week
        vle_student = vle_df[vle_df["student_id"] == student_id].copy()
        vle_student = vle_student.sort_values("week")

        # Attendance features per week
        att_student = attendance_df[attendance_df["student_id"] == student_id].copy()
        att_student = att_student.sort_values("week")

        # Get max week
        max_week = max(vle_student["week"].max(), att_student["week"].max())

        # Create weekly feature matrix
        sequence = []

        for week in range(1, min(max_week + 1, self.sequence_length + 1)):
            # VLE features
            vle_week = vle_student[vle_student["week"] == week]
            if len(vle_week) > 0:
                vle_features = [
                    vle_week["logins"].sum(),
                    vle_week["resources_accessed"].sum(),
                    vle_week["forum_posts"].sum(),
                    vle_week["quiz_attempts"].sum(),
                    vle_week["video_views"].sum(),
                    vle_week["total_actions"].sum(),
                ]
            else:
                vle_features = [0, 0, 0, 0, 0, 0]

            # Attendance features
            att_week = att_student[att_student["week"] == week]
            if len(att_week) > 0:
                attendance_rate = att_week["attended"].mean()
                absent_count = (att_week["status"] == "Absent").sum()
                authorised_absent = (att_week["status"] == "Authorised Absence").sum()
            else:
                attendance_rate = 0.0
                absent_count = 0
                authorised_absent = 0

            # Assessment features (cumulative to this week)
            assessments_before = assessments_df[
                (assessments_df["student_id"] == student_id)
            ]
            # Assume assessments have a week column or date
            if "week" in assessments_df.columns:
                assessments_before = assessments_before[assessments_before["week"] <= week]

            if len(assessments_before) > 0:
                avg_mark = assessments_before["mark"].mean()
                n_assessments = len(assessments_before)
                submission_rate = assessments_before["submitted"].mean()
                late_rate = assessments_before["late_submission"].mean()
            else:
                avg_mark = 0.0
                n_assessments = 0
                submission_rate = 0.0
                late_rate = 0.0

            # Combine all features for this week
            week_features = vle_features + [
                attendance_rate,
                absent_count,
                authorised_absent,
                avg_mark,
                n_assessments,
                submission_rate,
                late_rate,
            ]

            sequence.append(week_features)

        if len(sequence) == 0:
            return None

        # Pad or truncate to sequence_length
        sequence = np.array(sequence)
        if len(sequence) < self.sequence_length:
            # Pad with zeros
            padding = np.zeros((self.sequence_length - len(sequence), sequence.shape[1]))
            sequence = np.vstack([sequence, padding])
        elif len(sequence) > self.sequence_length:
            # Truncate (keep most recent)
            sequence = sequence[-self.sequence_length:]

        return sequence

    def _get_static_features(
        self,
        student_id: str,
        students_df: pd.DataFrame,
        enrollments_df: pd.DataFrame,
    ) -> np.ndarray:
        """Get static features for a student."""
        student = students_df[students_df["student_id"] == student_id]

        if len(student) == 0:
            # Default features
            return np.zeros(15)

        student = student.iloc[0]

        features = [
            # Demographics
            int(student.get("gender", "") == "Female"),
            int(student.get("gender", "") == "Male"),
            student.get("imd_decile", 5) / 10.0,  # Normalized
            int(student.get("polar_quintile", 3) == 1),  # Low participation
            int(student.get("care_leaver", 0)),
            int(student.get("first_generation_uni", 0)),
            int(student.get("disability", 0)),
            # Prior attainment
            student.get("ucas_tariff_points", 100) / 168.0,  # Normalized
            int(student.get("qualification_type", "") == "A-Level"),
            int(student.get("qualification_type", "") == "BTEC"),
            # Enrollment
            int(student.get("predicted_grade", False)),
            # Age proxy (from DOB if available)
            0.0,  # Would calculate from DOB
            0.0,  # Additional static features
            0.0,
            0.0,
        ]

        return np.array(features)

    def _get_target(
        self,
        student_id: str,
        enrollments_df: pd.DataFrame,
        horizon: int,
    ) -> Optional[int]:
        """
        Get target variable (event in next horizon weeks).

        Args:
            student_id: Student ID
            enrollments_df: Enrollment outcomes
            horizon: Prediction horizon in weeks

        Returns:
            1 if event occurred, 0 otherwise, None if unknown
        """
        enrollment = enrollments_df[enrollments_df["student_id"] == student_id]

        if len(enrollment) == 0:
            return None

        enrollment = enrollment.iloc[0]

        # Target depends on target_type
        if self.target_type == "dropout":
            # Dropped out or withdrawn
            target = int(enrollment.get("enrollment_status", "") == "Withdrawn")
        elif self.target_type == "failure":
            # Failed (low marks, not retained)
            target = int(enrollment.get("retained_year2", 1) == 0)
        elif self.target_type == "both":
            # Either dropout or failure
            target = int(
                (enrollment.get("enrollment_status", "") == "Withdrawn") |
                (enrollment.get("retained_year2", 1) == 0)
            )
        else:
            target = 0

        return target

    def create_aggregated_features(
        self,
        vle_df: pd.DataFrame,
        attendance_df: pd.DataFrame,
        assessments_df: pd.DataFrame,
        students_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create aggregated (non-sequential) features for XGBoost/survival models.

        Args:
            vle_df: VLE engagement data
            attendance_df: Attendance data
            assessments_df: Assessment data
            students_df: Student demographics

        Returns:
            Tuple of (feature_dataframe, feature_names)
        """
        logger.info("Creating aggregated features...")

        features_list = []

        # Get all unique students
        all_students = set(vle_df["student_id"].unique())
        all_students = all_students | set(attendance_df["student_id"].unique())
        all_students = all_students | set(assessments_df["student_id"].unique())

        for student_id in all_students:
            features = {"student_id": student_id}

            # === VLE Aggregations ===
            vle_student = vle_df[vle_df["student_id"] == student_id]
            if len(vle_student) > 0:
                features["vle_total_logins"] = vle_student["logins"].sum()
                features["vle_total_resources"] = vle_student["resources_accessed"].sum()
                features["vle_total_posts"] = vle_student["forum_posts"].sum()
                features["vle_avg_weekly_logins"] = vle_student["logins"].mean()
                features["vle_std_weekly_logins"] = vle_student["logins"].std()

                # Trend (first half vs second half)
                n_weeks = len(vle_student)
                if n_weeks >= 4:
                    first_half = vle_student.head(n_weeks // 2)["total_actions"].mean()
                    second_half = vle_student.tail(n_weeks // 2)["total_actions"].mean()
                    features["vle_trend"] = second_half - first_half
                else:
                    features["vle_trend"] = 0.0
            else:
                features["vle_total_logins"] = 0
                features["vle_total_resources"] = 0
                features["vle_total_posts"] = 0
                features["vle_avg_weekly_logins"] = 0
                features["vle_std_weekly_logins"] = 0
                features["vle_trend"] = 0.0

            # === Attendance Aggregations ===
            att_student = attendance_df[attendance_df["student_id"] == student_id]
            if len(att_student) > 0:
                features["attendance_rate"] = att_student["attended"].mean()
                features["attendance_std"] = att_student["attended"].std()
                features["total_absences"] = (att_student["status"] == "Absent").sum()
                features["authorised_absences"] = (att_student["status"] == "Authorised Absence").sum()
                features["medical_absences"] = (att_student["status"] == "Medical").sum()

                # Trend
                n_weeks = len(att_student)
                if n_weeks >= 4:
                    first_half = att_student.head(n_weeks // 2)["attended"].mean()
                    second_half = att_student.tail(n_weeks // 2)["attended"].mean()
                    features["attendance_trend"] = second_half - first_half
                else:
                    features["attendance_trend"] = 0.0
            else:
                features["attendance_rate"] = 0.0
                features["attendance_std"] = 0.0
                features["total_absences"] = 0
                features["authorised_absences"] = 0
                features["medical_absences"] = 0
                features["attendance_trend"] = 0.0

            # === Assessment Aggregations ===
            ass_student = assessments_df[assessments_df["student_id"] == student_id]
            if len(ass_student) > 0:
                features["avg_mark"] = ass_student["mark"].mean()
                features["std_mark"] = ass_student["mark"].std()
                features["min_mark"] = ass_student["mark"].min()
                features["n_assessments"] = len(ass_student)
                features["submission_rate"] = ass_student["submitted"].mean()
                features["late_submission_rate"] = ass_student["late_submission"].mean()
                features["resit_rate"] = (ass_student["attempt"] > 1).mean()
            else:
                features["avg_mark"] = 0.0
                features["std_mark"] = 0.0
                features["min_mark"] = 0.0
                features["n_assessments"] = 0
                features["submission_rate"] = 0.0
                features["late_submission_rate"] = 0.0
                features["resit_rate"] = 0.0

            # === Student Demographics ===
            student = students_df[students_df["student_id"] == student_id]
            if len(student) > 0:
                student = student.iloc[0]
                features["gender_female"] = int(student.get("gender", "") == "Female")
                features["imd_decile"] = student.get("imd_decile", 5)
                features["polar_q1"] = int(student.get("polar_quintile", 3) == 1)
                features["care_leaver"] = int(student.get("care_leaver", 0))
                features["first_gen"] = int(student.get("first_generation_uni", 0))
                features["ucas_tariff"] = student.get("ucas_tariff_points", 100)
            else:
                features["gender_female"] = 0
                features["imd_decile"] = 5
                features["polar_q1"] = 0
                features["care_leaver"] = 0
                features["first_gen"] = 0
                features["ucas_tariff"] = 100

            features_list.append(features)

        # Convert to DataFrame
        df_features = pd.DataFrame(features_list)

        # Get feature names (exclude student_id)
        feature_names = [col for col in df_features.columns if col != "student_id"]
        self.feature_names = feature_names

        logger.info(f"Created {len(feature_names)} aggregated features for {len(df_features)} students")

        return df_features, feature_names


# Example usage
if __name__ == "__main__":
    from src.data.synthetic import SITSSyntheticGenerator

    # Generate synthetic data
    print("Generating synthetic data...")
    generator = SITSSyntheticGenerator(n_students=500, n_courses=50, seed=42)
    datasets = generator.generate_all_datasets()

    # Create sequences
    print("\nCreating early warning sequences...")
    engineer = EarlyWarningFeatureEngineer(
        sequence_length=12,
        prediction_horizon=4,
        target_type="both",
        min_weeks=4,
    )

    X_seq, X_static, y, metadata = engineer.create_sequences(
        vle_df=datasets["vle_engagement"],
        attendance_df=datasets["attendance"],
        assessments_df=datasets["assessments"],
        students_df=datasets["students"],
        enrollments_df=datasets["enrollments"],
    )

    print(f"\nSequence shape: {X_seq.shape}")
    print(f"Static features: {X_static.shape}")
    print(f"Targets: {len(y)} (positive rate: {y.mean():.2%})")
    print(f"Students: {metadata['n_students']}")

    # Create aggregated features
    print("\nCreating aggregated features...")
    df_agg, feature_names = engineer.create_aggregated_features(
        vle_df=datasets["vle_engagement"],
        attendance_df=datasets["attendance"],
        assessments_df=datasets["assessments"],
        students_df=datasets["students"],
    )

    print(f"Aggregated features: {df_agg.shape}")
    print(f"Feature names: {len(feature_names)}")

    print("\n✓ Early warning feature engineering complete!")
