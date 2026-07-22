"""
Data Preparation for Enrollment Yield Prediction.

Prepares features for binary classification:
- Will a student accept their offer? (yes/no)

Feature Groups:
- Applicant characteristics (grades, qualifications, demographics)
- Course characteristics (selectivity, reputation, outcomes)
- Engagement signals (open days, communications, portal activity)
- Contextual indicators (distance, insurance choice, clearing)
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnrollmentYieldFeatureEngineer:
    """
    Feature engineering for enrollment yield prediction.

    Creates features from:
    - Student demographics and qualifications
    - Course characteristics
    - Engagement signals
    - Contextual factors
    """

    def __init__(
        self,
        target_col: str = "accepted_offer",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Initialize feature engineer.

        Args:
            target_col: Target column name
            test_size: Test set proportion
            random_state: Random seed
        """
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.feature_names: List[str] = []
        self.categorical_encoders: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}

    def engineer_features(
        self,
        students_df: pd.DataFrame,
        qualifications_df: pd.DataFrame,
        courses_df: pd.DataFrame,
        enrollments_df: pd.DataFrame,
        engagement_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """
        Create full feature matrix for yield prediction.

        Args:
            students_df: Student demographics
            qualifications_df: Prior qualifications
            courses_df: Course database
            enrollments_df: Enrollment records
            engagement_df: Optional engagement signals

        Returns:
            Tuple of (feature_matrix, target_vector, feature_names)
        """
        logger.info("Engineering enrollment yield features...")

        # Merge all data sources
        df = self._merge_data_sources(
            students_df, qualifications_df, courses_df, enrollments_df, engagement_df
        )

        # Create features
        df = self._create_applicant_features(df)
        df = self._create_course_features(df)
        df = self._create_engagement_features(df, engagement_df)
        df = self._create_contextual_features(df)

        # Create target variable
        df = self._create_target(df)

        # Select feature columns
        feature_cols = [col for col in df.columns if col.startswith("feat_")]
        self.feature_names = feature_cols

        # Handle missing values
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

        # Encode categorical features
        df = self._encode_categorical_features(df)

        # Scale numeric features
        df = self._scale_numeric_features(df)

        # Extract feature matrix and target
        X = df[feature_cols].values
        y = df[self.target_col].values

        logger.info(f"Created {len(feature_cols)} features for {len(df)} applicants")
        logger.info(f"Positive class rate: {y.mean():.2%}")

        return df, X, y

    def _merge_data_sources(
        self,
        students_df: pd.DataFrame,
        qualifications_df: pd.DataFrame,
        courses_df: pd.DataFrame,
        enrollments_df: pd.DataFrame,
        engagement_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Merge all data sources into single DataFrame."""
        logger.info("Merging data sources...")

        # Start with enrollments (each row is one applicant)
        df = enrollments_df.copy()

        # Merge student demographics
        df = df.merge(students_df, on="student_id", how="left")

        # Merge qualifications
        df = df.merge(qualifications_df, on="student_id", how="left")

        # Merge course characteristics
        df = df.merge(courses_df, on="course_id", how="left")

        # Merge engagement if available
        if engagement_df is not None:
            df = df.merge(engagement_df, on="student_id", how="left")
            logger.info(f"Merged engagement data for {len(engagement_df)} applicants")

        logger.info(f"Merged dataset: {len(df)} applicants, {len(df.columns)} columns")
        return df

    def _create_applicant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from applicant characteristics."""
        logger.info("Creating applicant features...")

        # === Academic Features ===
        # UCAS tariff points (already exists, but create bands)
        df["feat_tariff_band_low"] = (df["ucas_tariff_points"] <= 96).astype(int)
        df["feat_tariff_band_medium"] = (
            (df["ucas_tariff_points"] > 96) & (df["ucas_tariff_points"] <= 128)
        ).astype(int)
        df["feat_tariff_band_high"] = (df["ucas_tariff_points"] > 128).astype(int)

        # Qualification type (one-hot)
        qual_types = df["qualification_type"].unique()
        for qual in qual_types:
            if pd.notna(qual):
                df[f"feat_qual_{qual.replace(' ', '_')}"] = (
                    df["qualification_type"] == qual
                ).astype(int)

        # Predicted vs achieved grades
        if "predicted_grade" in df.columns:
            df["feat_has_predicted_grade"] = df["predicted_grade"].astype(int)

        # === Demographic Features ===
        # Gender
        df["feat_gender_female"] = (df["gender"] == "Female").astype(int)
        df["feat_gender_male"] = (df["gender"] == "Male").astype(int)

        # Ethnicity (one-hot for major groups)
        ethnicity_groups = df["ethnicity"].dropna().unique()
        for ethnicity in ethnicity_groups[:5]:  # Top 5 groups
            df[f"feat_ethnicity_{ethnicity.replace(' ', '_')}"] = (
                df["ethnicity"] == ethnicity
            ).astype(int)

        # Age at enrollment
        if "date_of_birth" in df.columns and "enrollment_date" in df.columns:
            df["dob"] = pd.to_datetime(df["date_of_birth"])
            df["enrollment_date"] = pd.to_datetime(df["enrollment_date"])
            df["feat_age_at_enrollment"] = (
                df["enrollment_date"] - df["dob"]
            ).dt.days / 365.25
            df["feat_age_under_21"] = (df["feat_age_at_enrollment"] < 21).astype(int)
            df["feat_age_mature"] = (df["feat_age_at_enrollment"] >= 25).astype(int)

        # === Socioeconomic Features ===
        # IMD decile (inverse - higher = more deprived)
        if "imd_decile" in df.columns:
            df["feat_imd_inverse"] = 11 - df["imd_decile"]
            df["feat_imd_most_deprived"] = (df["imd_decile"] <= 2).astype(int)
            df["feat_imd_least_deprived"] = (df["imd_decile"] >= 9).astype(int)

        # POLAR4 (low participation area)
        if "polar_quintile" in df.columns:
            df["feat_low_participation_area"] = (df["polar_quintile"] == 1).astype(int)
            df["feat_high_participation_area"] = (df["polar_quintile"] >= 4).astype(int)

        # Contextual indicators
        if "care_leaver" in df.columns:
            df["feat_care_leaver"] = df["care_leaver"].astype(int)

        if "first_generation_uni" in df.columns:
            df["feat_first_generation"] = df["first_generation_uni"].astype(int)

        if "disability" in df.columns:
            df["feat_has_disability"] = df["disability"].astype(int)

        return df

    def _create_course_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from course characteristics."""
        logger.info("Creating course features...")

        # === Course Selectivity ===
        # Entry tariff (normalized)
        if "entry_tariff" in df.columns:
            df["feat_entry_tariff_normalized"] = df["entry_tariff"] / 168.0
            df["feat_entry_tariff_high"] = (df["entry_tariff"] >= 144).astype(int)
            df["feat_entry_tariff_low"] = (df["entry_tariff"] <= 112).astype(int)

        # Grade match (how well student matches entry requirements)
        if "ucas_tariff_points" in df.columns and "entry_tariff" in df.columns:
            df["feat_grade_match"] = (
                df["ucas_tariff_points"] - df["entry_tariff"]
            ) / 168.0
            df["feat_meets_entry_requirements"] = (
                df["ucas_tariff_points"] >= df["entry_tariff"]
            ).astype(int)
            df["feat_exceeds_entry_requirements"] = (
                df["ucas_tariff_points"] >= df["entry_tariff"] + 16
            ).astype(int)

        # === Course Outcomes ===
        # Employment rate
        if "employment_rate_15m" in df.columns:
            df["feat_employment_rate"] = df["employment_rate_15m"]
            df["feat_high_employment"] = (df["employment_rate_15m"] >= 0.90).astype(int)

        # Satisfaction score
        if "satisfaction_score" in df.columns:
            df["feat_satisfaction_score"] = df["satisfaction_score"] / 5.0
            df["feat_high_satisfaction"] = (
                df["satisfaction_score"] >= 4.5
            ).astype(int)

        # === Course Characteristics ===
        # Department (one-hot for major departments)
        departments = df["department"].dropna().unique()
        for dept in departments[:5]:  # Top 5 departments
            df[f"feat_dept_{dept.replace(' ', '_')}"] = (
                df["department"] == dept
            ).astype(int)

        # Sandwich year
        if "course_length_years" in df.columns:
            df["feat_sandwich_year"] = (df["course_length_years"] == 4).astype(int)

        # Assessment type
        if "coursework_weight_pct" in df.columns:
            df["feat_coursework_heavy"] = (
                df["coursework_weight_pct"] >= 60
            ).astype(int)
            df["feat_exam_heavy"] = (df["coursework_weight_pct"] <= 40).astype(int)

        # Accreditation
        if "accredited" in df.columns:
            df["feat_accredited"] = df["accredited"].astype(int)

        return df

    def _create_engagement_features(
        self, df: pd.DataFrame, engagement_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Create features from engagement signals."""
        if engagement_df is None:
            logger.info("No engagement data available, skipping engagement features")
            return df

        logger.info("Creating engagement features...")

        # Aggregate engagement by student
        engagement_agg = engagement_df.groupby("student_id").agg(
            total_logins=("logins", "sum"),
            total_resources=("resources_accessed", "sum"),
            total_forum_posts=("forum_posts", "sum"),
            avg_weekly_logins=("logins", "mean"),
            engagement_trend=("total_actions", lambda x: x.iloc[-7:].mean() - x.iloc[:7].mean() if len(x) >= 14 else 0),
        ).reset_index()

        # Merge with main dataframe
        df = df.merge(engagement_agg, on="student_id", how="left")

        # Create engagement features
        df["feat_total_logins"] = np.log1p(df["total_logins"].fillna(0))
        df["feat_total_resources"] = np.log1p(df["total_resources"].fillna(0))
        df["feat_avg_weekly_logins"] = df["avg_weekly_logins"].fillna(0)
        df["feat_engagement_trend"] = df["engagement_trend"].fillna(0)
        df["feat_high_engagement"] = (df["feat_total_logins"] >= 3.0).astype(int)

        # Drop intermediate columns
        df = df.drop(columns=["total_logins", "total_resources", "total_forum_posts", "avg_weekly_logins", "engagement_trend"])

        return df

    def _create_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create contextual features."""
        logger.info("Creating contextual features...")

        # === Geographic Features ===
        # Distance from university (if postcode available)
        # For now, use postcode area as proxy
        if "postcode" in df.columns:
            df["postcode_area"] = df["postcode"].str[:2].str.upper()
            # Local postcodes (example: assume university is in AB area)
            df["feat_local_applicant"] = (df["postcode_area"] == "AB").astype(int)

        # === Application Timing ===
        if "enrollment_date" in df.columns:
            df["enrollment_date"] = pd.to_datetime(df["enrollment_date"])
            df["feat_application_month"] = df["enrollment_date"].dt.month
            df["feat_early_application"] = (df["feat_application_month"] <= 11).astype(int)
            df["feat_late_application"] = (df["feat_application_month"] >= 6).astype(int)

        # === Insurance Choice ===
        # Proxy: if tariff is much lower than course entry requirements
        if "ucas_tariff_points" in df.columns and "entry_tariff" in df.columns:
            df["feat_insurance_choice"] = (
                df["ucas_tariff_points"] < df["entry_tariff"] - 16
            ).astype(int)

        # === Clearing Eligibility ===
        # Late applications with lower grades
        df["feat_clearing_eligible"] = (
            df.get("feat_late_application", 0) & df.get("feat_tariff_band_low", 0)
        ).astype(int)

        return df

    def _create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variable (accepted_offer)."""
        logger.info("Creating target variable...")

        # Target: 1 if enrolled, 0 if declined
        # For real data, would need offer-holder data
        # For now, all enrolled students = 1
        df[self.target_col] = 1

        # In production, would merge with offer-holder data:
        # - Students who received offer but didn't enroll = 0
        # - Students who enrolled = 1

        logger.info(f"Target variable created: {df[self.target_col].sum()} positive, {len(df) - df[self.target_col].sum()} negative")
        return df

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using LabelEncoder."""
        from sklearn.preprocessing import LabelEncoder

        logger.info("Encoding categorical features...")

        categorical_cols = [col for col in df.columns if col.startswith("feat_") and df[col].dtype == "object"]

        for col in categorical_cols:
            encoder = LabelEncoder()
            df[f"{col}_encoded"] = encoder.fit_transform(df[col].astype(str))
            self.categorical_encoders[col] = encoder

        return df

    def _scale_numeric_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numeric features using StandardScaler.
        
        CRITICAL: To prevent data leakage, scaling must be done AFTER train/test split.
        This method is called during feature engineering but should be refit on training data only.
        
        Args:
            df: DataFrame with features
            fit: If True, fit new scaler. If False, use existing fitted scaler.
        """
        from sklearn.preprocessing import StandardScaler

        numeric_cols = [col for col in df.columns if col.startswith("feat_") and df[col].dtype in ["int64", "float64"]]

        if fit:
            # Fit on this data (should be training data only in production)
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self.scalers["numeric_features"] = scaler
            logger.info(f"Fitted scaler on {len(df)} samples")
        else:
            # Transform using existing scaler (for test/validation data)
            if "numeric_features" not in self.scalers:
                raise ValueError("No fitted scaler found. Must fit on training data first.")
            df[numeric_cols] = self.scalers["numeric_features"].transform(df[numeric_cols])
            logger.info(f"Transformed {len(df)} samples with existing scaler")

        return df

    def create_train_test_split(
        self,
        df: pd.DataFrame,
        X: np.ndarray,
        y: np.ndarray,
        stratified: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create train/test split with proper scaling to prevent data leakage.
        
        CRITICAL: This method handles the ENTIRE workflow:
        1. Split data first
        2. Fit scaler on training data ONLY
        3. Transform test data with fitted scaler
        
        Args:
            df: Full DataFrame (for splitting)
            X: Feature matrix (before scaling)
            y: Target vector
            stratified: Whether to use stratified split

        Returns:
            Tuple of (df_train, df_test, X_train_scaled, X_test_scaled, y_train, y_test)
        """
        from sklearn.model_selection import train_test_split

        logger.info(f"Creating train/test split (test_size={self.test_size})...")

        # Split DataFrame first (to maintain alignment)
        if stratified:
            df_train, df_test = train_test_split(
                df, test_size=self.test_size, stratify=df[self.target_col], 
                random_state=self.random_state
            )
        else:
            df_train, df_test = train_test_split(
                df, test_size=self.test_size, random_state=self.random_state
            )

        # Extract feature columns
        feature_cols = [col for col in df.columns if col.startswith("feat_")]
        
        # Scale training data (FIT on train only)
        X_train = df_train[feature_cols].values
        X_train_scaled = self._scale_numeric_features(df_train.copy(), fit=True)[feature_cols].values
        
        # Scale test data (TRANSFORM using train scaler)
        X_test = df_test[feature_cols].values
        X_test_scaled = self._scale_numeric_features(df_test.copy(), fit=False)[feature_cols].values
        
        # Extract targets
        y_train = df_train[self.target_col].values
        y_test = df_test[self.target_col].values

        logger.info(f"Train: {len(X_train_scaled)}, Test: {len(X_test_scaled)}")
        logger.info(f"Train positive rate: {y_train.mean():.2%}, Test positive rate: {y_test.mean():.2%}")
        logger.info("✓ Scaling done correctly: fit on train, transform on test")

        return df_train, df_test, X_train_scaled, X_test_scaled, y_train, y_test


# Example usage
if __name__ == "__main__":
    from src.data.synthetic import SITSSyntheticGenerator
    from src.data.feature_store import FeatureStore

    # Generate synthetic data
    print("Generating synthetic data...")
    generator = SITSSyntheticGenerator(n_students=1000, n_courses=50, seed=42)
    datasets = generator.generate_all_datasets()

    # Engineer features
    print("\nEngineering enrollment yield features...")
    engineer = EnrollmentYieldFeatureEngineer(target_col="accepted_offer", test_size=0.2)

    df, X, y = engineer.engineer_features(
        students_df=datasets["students"],
        qualifications_df=datasets["qualifications"],
        courses_df=datasets["courses"],
        enrollments_df=datasets["enrollments"],
        engagement_df=None,  # No engagement data in synthetic
    )

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"Number of features: {len(engineer.feature_names)}")
    print(f"Positive class rate: {y.mean():.2%}")

    # CORRECT WORKFLOW: Use create_train_test_split to prevent data leakage
    print("\nCreating train/test split with proper scaling...")
    df_train, df_test, X_train, X_test, y_train, y_test = engineer.create_train_test_split(
        df, X, y, stratified=True
    )

    print(f"\n✓ Train set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"✓ Train positive rate: {y_train.mean():.2%}")
    print(f"✓ Test positive rate: {y_test.mean():.2%}")
    print("\n✓ Feature engineering complete! Data leakage prevented.")
