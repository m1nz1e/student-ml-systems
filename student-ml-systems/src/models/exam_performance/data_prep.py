"""Exam Performance Feature Engineering."""
from typing import Tuple, Dict
import numpy as np
import pandas as pd

class ExamFeatureEngineer:
    """Feature engineer for exam performance prediction."""
    
    def __init__(self):
        self.feature_names = []
    
    def engineer_features(
        self,
        exam_df: pd.DataFrame,
        students_df: pd.DataFrame,
        assessments_df: pd.DataFrame,
        vle_df: pd.DataFrame,
        modules_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]:
        """
        Engineer features for exam prediction.
        
        Args:
            exam_df: Exam outcome data
            students_df: Student records
            assessments_df: Assessment records
            vle_df: VLE engagement records
            modules_df: Module records
            
        Returns:
            Tuple of (df, X, y_dict) with targets:
            {exam_mark, pass_fail, grade_class}
        """
        df = exam_df.copy()
        
        # Student demographics
        student_cols = ['student_id', 'gender', 'ethnicity', 'disability',
                       'imd_decile', 'polar_quintile', 'care_leaver', 'first_generation_uni']
        available_cols = [c for c in student_cols if c in students_df.columns]
        df = df.merge(students_df[available_cols], on='student_id', how='left')
        
        # Coursework features already in exam_df from generate_exam_outcomes
        
        # Assessment trend
        if assessments_df is not None:
            assess_trend = assessments_df.groupby('student_id').agg({
                'mark': ['mean', 'std', 'count']
            }).reset_index()
            assess_trend.columns = ['student_id', 'assess_mean', 'assess_std', 'n_assessments']
            df = df.merge(assess_trend, on='student_id', how='left')
        
        # VLE engagement
        if vle_df is not None:
            vle_agg = vle_df.groupby('student_id').agg({
                'logins': ['sum', 'mean'],
                'resources_accessed': 'sum',
                'total_actions': ['sum', 'mean']
            }).reset_index()
            vle_agg.columns = ['student_id', 'total_logins', 'avg_logins',
                              'total_resources', 'total_actions', 'avg_actions']
            df = df.merge(vle_agg, on='student_id', how='left')
        
        # Fill NaNs
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median() if hasattr(df[col], 'median') else 0)
        
        # Targets
        y_exam = df['exam_mark'].values
        y_pass = df['pass_fail'].values
        grade_map = {'Fail': 0, 'Pass': 1, 'Merit': 2, 'Distinction': 3}
        y_grade = df['grade_class'].map(grade_map).fillna(1).values
        
        # Features
        numeric_features = [
            'cw_mean', 'cw_std', 'cw_min', 'cw_max', 'n_courseworks',
            'engagement_rate', 'total_logins', 'total_resources', 'total_actions',
            'avg_logins', 'avg_actions', 'assess_mean', 'assess_std', 'n_assessments',
            'exam_weight_pct', 'module_pass_rate', 'predicted_final_score',
            'imd_decile', 'polar_quintile'
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
        
        return df, X, {'exam_mark': y_exam, 'pass_fail': y_pass, 'grade_class': y_grade}
