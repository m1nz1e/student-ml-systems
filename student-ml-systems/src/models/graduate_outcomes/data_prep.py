"""
Graduate Outcomes Feature Engineering.

Prepares features for multi-task graduate outcome prediction:
- Employment status (4 classes)
- Salary band (4 classes)
- Further study destination (3 classes)
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

# Label encodings
EMPLOYMENT_STATUS_MAP = {
    'Unemployed': 0,
    'Further Study Only': 1,
    'Both Employed and Study': 2,
    'Employed': 3
}

SALARY_BAND_MAP = {
    'Under £20,000': 0,
    '£20,000 - £30,000': 1,
    '£30,000 - £40,000': 2,
    'Over £40,000': 3
}

STUDY_DEST_MAP = {
    'Not Studying': 0,
    'UK': 1,
    'EU': 2,
    'International': 3
}


class GraduateOutcomeFeatureEngineer:
    """
    Feature engineer for graduate outcome prediction.

    Combines:
    - Degree outcome predictions
    - Student demographics
    - Course characteristics
    - Engagement metrics
    """

    def __init__(self):
        self.feature_names = []

    def engineer_features(
        self,
        graduate_outcomes_df: pd.DataFrame,
        students_df: pd.DataFrame,
        degree_outcomes_df: pd.DataFrame,
        vle_df: pd.DataFrame,
        assessments_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]:
        """
        Engineer features for graduate outcome prediction.

        Args:
            graduate_outcomes_df: Graduate outcome data (with is_employed, degree_class_ordinal, etc.)
            students_df: Student records (SITS schema)
            degree_outcomes_df: Degree classification results
            vle_df: VLE engagement records
            assessments_df: Assessment records

        Returns:
            Tuple of (df, X, y_dict) where:
            - df: Feature DataFrame
            - X: Feature matrix (n_samples, n_features)
            - y_dict: Dict of target arrays {employment, salary, study}
        """
        # Start with graduate outcomes (which already has some computed features)
        df = graduate_outcomes_df.copy()

        # Add student demographics
        student_cols = ['student_id', 'gender', 'ethnicity', 'disability',
                       'imd_decile', 'polar_quintile', 'care_leaver',
                       'first_generation_uni']
        available_cols = [c for c in student_cols if c in students_df.columns]
        df = df.merge(students_df[available_cols], on='student_id', how='left')

        # Merge degree outcome
        classification_to_ordinal = {
            'Fail': 0, 'Third': 1, '2:2': 2, '2:1': 3, 'First': 4
        }
        degree_df = degree_outcomes_df.copy()
        degree_df['gpa'] = degree_df['weighted_gpa']
        degree_df['predicted_class_ordinal'] = (
            degree_df['final_classification']
            .map(classification_to_ordinal)
            .fillna(2)
        )
        df = df.merge(
            degree_df[['student_id', 'final_classification', 'gpa', 'predicted_class_ordinal']],
            on='student_id', how='left'
        )
        df['gpa'] = df['gpa'].fillna(df['gpa'].median())
        df['predicted_class_ordinal'] = df['predicted_class_ordinal'].fillna(2)

        # Use VLE engagement from outcomes if available, else compute from vle_df
        if 'total_logins' not in df.columns and vle_df is not None:
            vle_agg = vle_df.groupby('student_id').agg({
                'logins': 'sum',
                'resources_accessed': 'sum',
                'total_actions': 'sum'
            }).reset_index()
            vle_agg.columns = ['student_id', 'total_logins', 'total_resources', 'total_actions']
            df = df.merge(vle_agg, on='student_id', how='left')

        # Fill any missing values
        for col in ['total_logins', 'total_resources', 'total_actions']:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Attendance rate from assessments
        if assessments_df is not None:
            attendance_agg = assessments_df.groupby('student_id').agg({
                'mark': ['mean', 'std', 'count']
            }).reset_index()
            attendance_agg.columns = ['student_id', 'avg_mark', 'mark_std', 'n_assessments']
            df = df.merge(attendance_agg, on='student_id', how='left')
            df['avg_mark'] = df['avg_mark'].fillna(df['avg_mark'].median())
            df['attendance_rate'] = (df['avg_mark'] / 100 * 100).clip(0, 100)
        else:
            df['attendance_rate'] = 70.0  # Default

        # Use pre-computed degree_class_ordinal from outcomes if available
        if 'degree_class_ordinal' not in df.columns:
            df['degree_class_ordinal'] = df['final_classification'].map(classification_to_ordinal).fillna(2)

        # Career readiness score — degree drives it
        df['career_readiness_score'] = (
            (df['degree_class_ordinal'] / 4.0 * 60) +
            (df['total_logins'] / (df['total_logins'].max() + 1) * 40)
        ).clip(0, 100)

        # Encode targets
        y_employment = df['employment_status'].map(EMPLOYMENT_STATUS_MAP).values
        y_salary = df['salary_band'].map(SALARY_BAND_MAP).values
        y_study = df['study_destination'].map(STUDY_DEST_MAP).values

        # Build feature list — numeric features with strong signal
        numeric_features = [
            'gpa', 'predicted_class_ordinal', 'career_readiness_score',
            'attendance_rate', 'total_logins', 'total_resources',
            'imd_decile', 'polar_quintile', 'is_employed', 'degree_class_ordinal'
        ]
        categorical_features = ['gender', 'ethnicity', 'disability',
                              'care_leaver', 'first_generation_uni']

        feature_cols = []
        for col in numeric_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median() if df[col].dtype != object else 0)
                feature_cols.append(col)

        for col in categorical_features:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown').astype(str)
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                feature_cols.extend(dummies.columns.tolist())
                df = pd.concat([df, dummies], axis=1)

        self.feature_names = feature_cols
        X = df[feature_cols].values

        return df, X, {
            'employment': y_employment,
            'salary': y_salary,
            'study': y_study
        }
