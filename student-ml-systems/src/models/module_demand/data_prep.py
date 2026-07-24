"""Module Demand Feature Engineering."""
from typing import Tuple, Dict
import numpy as np
import pandas as pd

class ModuleDemandFeatureEngineer:
    def __init__(self):
        self.feature_names = []
    
    def engineer_features(
        self,
        module_demand_df: pd.DataFrame,
        modules_df: pd.DataFrame,
        module_enrollments_df: pd.DataFrame,
        students_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, np.ndarray]]:
        # Start from module_demand_df (has department, capacity + demand targets)
        df = module_demand_df.copy()
        
        # Add historical stats from enrollments
        hist_stats = module_enrollments_df.groupby('module_id').agg({
            'grade': ['mean', 'std', 'count'], 'completed': 'mean'
        }).reset_index()
        hist_stats.columns = ['module_id', 'avg_grade', 'grade_std', 'total_enrollments', 'completion_rate']
        df = df.merge(hist_stats, on='module_id', how='left')
        for col in ['avg_grade', 'grade_std', 'total_enrollments', 'completion_rate']:
            if col in df.columns:
                df[col] = df[col].fillna(70)
        
        # Department encoding
        df['department_encoded'] = pd.factorize(df['department'])[0]
        y_count = df['enrollment_count'].values if 'enrollment_count' in df.columns else np.zeros(len(df))
        y_rate = df['fill_rate'].values if 'fill_rate' in df.columns else np.zeros(len(df))
        demand_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Oversubscribed': 3}
        y_category = df['demand_category'].map(demand_map).fillna(1).values
        numeric_features = ['capacity', 'exam_weight_pct', 'coursework_weight_pct',
                           'avg_grade', 'grade_std', 'total_enrollments', 'completion_rate',
                           'department_encoded', 'popularity_score']
        feature_cols = []
        for col in numeric_features:
            if col in df.columns:
                if df[col].dtype == object:
                    df[col] = pd.factorize(df[col])[0]
                df[col] = df[col].fillna(df[col].median() if hasattr(df[col], 'median') else 0)
                feature_cols.append(col)
        self.feature_names = feature_cols
        X = df[feature_cols].values
        return df, X, {'enrollment_count': y_count, 'fill_rate': y_rate, 'demand_category': y_category}
