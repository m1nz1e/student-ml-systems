"""
Student Retention Feature Engineering.

Prepares features for predicting student dropout/retention.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd


class RetentionFeatureEngineer:
    """
    Feature engineer for student retention prediction.

    Combines:
    - Student demographics
    - Engagement patterns
    - Academic performance
    - Risk indicators
    """

    def __init__(self):
        self.feature_names = []

    def engineer_features(
        self,
        retention_df: pd.DataFrame,
        students_df: pd.DataFrame,
        vle_df: pd.DataFrame,
        assessments_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]:
        """
        Engineer features for retention prediction.

        Args:
            retention_df: Retention outcome data
            students_df: Student records
            vle_df: VLE engagement records
            assessments_df: Assessment records

        Returns:
            Tuple of (df, X, y_dict) where:
            - df: Feature DataFrame
            - X: Feature matrix
            - y_dict: Dict of targets {risk_score, retention_risk, risk_category, departure_year}
        """
        df = retention_df.copy()

        # Merge student demographics
        student_cols = ['student_id', 'gender', 'ethnicity', 'disability',
                       'imd_decile', 'polar_quintile', 'care_leaver',
                       'first_generation_uni', 'date_of_birth']
        available_cols = [c for c in student_cols if c in students_df.columns]
        df = df.merge(students_df[available_cols], on='student_id', how='left')

        # Age at entry
        if 'date_of_birth' in df.columns:
            df['age'] = 2024 - pd.to_datetime(df['date_of_birth'], errors='coerce').dt.year
            df['age'] = df['age'].fillna(19)

        # VLE engagement trends
        if vle_df is not None and len(vle_df) > 0:
            vle_agg = vle_df.groupby('student_id').agg({
                'logins': ['sum', 'mean'],
                'resources_accessed': 'sum',
                'total_actions': ['sum', 'mean']
            }).reset_index()
            vle_agg.columns = ['student_id', 'total_logins', 'avg_logins',
                               'total_resources', 'total_actions', 'avg_actions']
            df = df.merge(vle_agg, on='student_id', how='left')

        # Assessment trends
        if assessments_df is not None and len(assessments_df) > 0:
            assess_agg = assessments_df.groupby('student_id').agg({
                'mark': ['mean', 'std', 'min', 'max', 'count']
            }).reset_index()
            assess_agg.columns = ['student_id', 'avg_mark', 'mark_std', 'min_mark', 'max_mark', 'n_assessments']
            df = df.merge(assess_agg, on='student_id', how='left')

        # Fill NaNs
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())

        # Targets
        y_risk = df['risk_score'].values
        y_binary = df['retention_risk'].values
        y_departure = df['departure_year'].values
        risk_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
        y_category = df['risk_category'].map(risk_map).fillna(0).values

        # Features
        numeric_features = [
            'engagement_rate', 'avg_mark', 'mark_std', 'min_mark', 'max_mark',
            'n_assessments', 'total_logins', 'total_resources', 'total_actions',
            'imd_decile', 'polar_quintile', 'age'
        ]
        categorical_features = ['gender', 'ethnicity', 'disability', 'care_leaver', 'first_generation_uni']

        feature_cols = []
        for col in numeric_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median() if hasattr(df[col], 'median') else 0)
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
            'risk_score': y_risk,
            'retention_risk': y_binary,
            'risk_category': y_category,
            'departure_year': y_departure
        }
