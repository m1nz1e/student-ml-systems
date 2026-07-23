"""
NSS Prediction Feature Engineering.

Prepares features for predicting student satisfaction:
- Overall satisfaction (binary)
- Theme scores (7 x 0-100)
- NPS (0-10)
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd


class NSSFeatureEngineer:
    """
    Feature engineer for NSS prediction.

    Combines:
    - Degree outcome predictions
    - Student demographics
    - Engagement metrics
    - Historical NSS patterns
    """

    def __init__(self):
        self.feature_names = []

    def engineer_features(
        self,
        nss_outcomes_df: pd.DataFrame,
        students_df: pd.DataFrame,
        degree_outcomes_df: pd.DataFrame,
        vle_df: pd.DataFrame,
        assessments_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]:
        """
        Engineer features for NSS prediction.

        Args:
            nss_outcomes_df: NSS outcome data
            students_df: Student records
            degree_outcomes_df: Degree classification results
            vle_df: VLE engagement records
            assessments_df: Assessment records

        Returns:
            Tuple of (df, X, y_dict) where:
            - df: Feature DataFrame
            - X: Feature matrix
            - y_dict: Dict of targets {overall_satisfied, theme_scores, nps_score}
        """
        # Start with NSS outcomes
        df = nss_outcomes_df.copy()

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

        # VLE engagement
        if vle_df is not None:
            vle_agg = vle_df.groupby('student_id').agg({
                'logins': 'sum',
                'resources_accessed': 'sum',
                'total_actions': 'sum'
            }).reset_index()
            vle_agg.columns = ['student_id', 'total_logins', 'total_resources', 'total_actions']
            df = df.merge(vle_agg, on='student_id', how='left')
            for col in ['total_logins', 'total_resources', 'total_actions']:
                if col in df.columns:
                    df[col] = df[col].fillna(0)

        # Attendance from assessments
        if assessments_df is not None:
            attendance_agg = assessments_df.groupby('student_id').agg({
                'mark': ['mean', 'count']
            }).reset_index()
            attendance_agg.columns = ['student_id', 'avg_mark', 'n_assessments']
            df = df.merge(attendance_agg, on='student_id', how='left')
            df['avg_mark'] = df['avg_mark'].fillna(df['avg_mark'].median())
            df['attendance_rate'] = (df['avg_mark'] / 100 * 100).clip(0, 100)
        else:
            df['attendance_rate'] = 70.0

        # Targets
        y_satisfied = df['overall_satisfied'].values
        y_nps = df['nps_score'].values
        y_themes = df[[
            'teaching_score', 'assessment_score', 'feedback_score',
            'support_score', 'organisation_score', 'learning_resources_score',
            'student_voice_score'
        ]].values

        # Features
        numeric_features = [
            'gpa', 'predicted_class_ordinal', 'engagement_rate',
            'attendance_rate', 'total_logins', 'total_resources',
            'imd_decile', 'polar_quintile',
            'module_evals_submitted', 'feedback_response_rate'
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
            'satisfied': y_satisfied,
            'nps': y_nps,
            'themes': y_themes
        }
