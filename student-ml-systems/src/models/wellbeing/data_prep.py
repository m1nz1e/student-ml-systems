"""Student Wellbeing Feature Engineering."""
from typing import Tuple, Dict
import numpy as np
import pandas as pd

class WellbeingFeatureEngineer:
    """Feature engineer for student wellbeing prediction."""
    
    def __init__(self):
        self.feature_names = []
    
    def engineer_features(
        self,
        wellbeing_df: pd.DataFrame,
        students_df: pd.DataFrame,
        vle_df: pd.DataFrame,
        assessments_df: pd.DataFrame,
        attendance_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]:
        """
        Engineer features for wellbeing prediction.
        
        Args:
            wellbeing_df: Wellbeing outcome data
            students_df: Student records
            vle_df: VLE engagement records
            assessments_df: Assessment records
            attendance_df: Attendance records
            
        Returns:
            Tuple of (df, X, y_dict) with targets:
            {wellbeing_score, at_risk, risk_level, support_need}
        """
        df = wellbeing_df.copy()
        
        # Student demographics
        student_cols = ['student_id', 'gender', 'ethnicity', 'disability',
                       'imd_decile', 'polar_quintile', 'care_leaver', 'first_generation_uni']
        available_cols = [c for c in student_cols if c in students_df.columns]
        df = df.merge(students_df[available_cols], on='student_id', how='left')
        
        # Engagement features already in wellbeing_df
        
        # Assessment features
        if assessments_df is not None:
            assess_stats = assessments_df.groupby('student_id').agg({
                'mark': ['mean', 'std', 'min', 'max', 'count']
            }).reset_index()
            assess_stats.columns = ['student_id', 'assess_mean', 'assess_std', 'assess_min', 'assess_max', 'n_assessments']
            df = df.merge(assess_stats, on='student_id', how='left')
        
        # Attendance features
        if attendance_df is not None and len(attendance_df) > 0:
            att_stats = attendance_df.groupby('student_id').agg({
                'status': lambda x: (x == 'Present').sum()
            }).reset_index()
            att_stats.columns = ['student_id', 'sessions_attended']
            total = attendance_df.groupby('student_id').size().reset_index(name='total')
            att_stats = att_stats.merge(total, on='student_id')
            att_stats['att_pct'] = (att_stats['sessions_attended'] / att_stats['total'] * 100).clip(0, 100)
            df = df.merge(att_stats[['student_id', 'att_pct']], on='student_id', how='left')
        
        # Fill NaNs
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median() if hasattr(df[col], 'median') else 0)
        
        # Targets
        y_score = df['wellbeing_score'].values
        y_at_risk = df['at_risk'].values
        support_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
        risk_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
        y_support = df['support_need'].values
        y_risk = df['risk_level'].map(risk_map).fillna(0).values
        
        # Features
        numeric_features = [
            'engagement_rate', 'attendance_rate', 'avg_mark', 'assess_mean', 'assess_std',
            'total_logins', 'total_resources', 'n_assessments', 'assess_min', 'assess_max',
            'imd_decile', 'polar_quintile', 'att_pct'
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
        
        return df, X, {'wellbeing_score': y_score, 'at_risk': y_at_risk, 'risk_level': y_risk, 'support_need': y_support}
