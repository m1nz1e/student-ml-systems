"""
Field mapping configuration for SITS/Tribal student data systems.

This module provides standardized field mappings between:
- Synthetic field names (used in this codebase)
- SITS (Student Information Toolkit System) field names
- Tribal (SITS alternative) field names

When connecting to real data, import the appropriate mapping and rename columns.

Usage:
    from src.data.field_mapping import SITS_FIELD_MAP
    
    # Load real SITS data
    real_df = pd.read_sql(query, connection)
    
    # Rename to synthetic field names
    synthetic_df = real_df.rename(columns=SITS_FIELD_MAP['students'])
"""

from typing import Dict, TYPE_CHECKING

# Avoid circular imports - import pandas only inside functions that need it
if TYPE_CHECKING:
    import pandas as pd

# =============================================================================
# SITS (Student Information Toolkit System) Field Mappings
# =============================================================================

SITS_FIELD_MAP: Dict[str, Dict[str, str]] = {
    "students": {
        # Synthetic -> SITS
        "student_id": "SPR_CODE",
        "gender": "SPR_SEX",
        "ethnicity": "SPR_ETN",
        "date_of_birth": "SPR_DOB",
        "postcode": "SPR_PSTC",
        "disability": "SPR_DIS",
        "nationality": "SPR_NAT",
        "email": "SPR_EMAIL",
        "mobile": "SPR_MOBILE",
        "home_fee_status": "SPR_HFEE",
        "polar4_quintile": "SPR_POLAR4",
        "imd_decile": "SPR_IMD",
        "application_id": "APP_CODE",
        "offer_holder": "APP_OFFER",
        "accepted_offer": "ENR_ACCP",  # Via ENR join
    },
    
    "qualifications": {
        "student_id": "SCH_CODE",
        "qualification_type": "SCH_QUAL",
        "subject": "SCH_SUBJ",
        "grade": "SCH_GRD",
        "ucas_tariff_points": "SCH_UCAS",
        "predicted_grade": "SCH_PRED",
        "institution": "SCH_INST",
        "year_completed": "SCH_YEAR",
    },
    
    "courses": {
        "course_id": "CRS_CODE",
        "course_name": "CRS_NAME",
        "department": "CRS_DEPT",
        "entry_tariff": "CRS_TARIFF",
        "course_length_years": "CRS_LEN",
        "exam_weight_pct": "CRS_EXAM",
        "coursework_weight_pct": "CRS_CW",
        "employment_rate_15m": "CRS_EMP",
        "satisfaction_score": "CRS_SAT",
        "accredited": "CRS_ACC",
    },
    
    "enrollments": {
        "student_id": "ENR_SPR",
        "course_id": "ENR_CRS",
        "enrollment_date": "ENR_DATE",
        "enrollment_status": "ENR_STAT",
        "accepted_offer": "ENR_ACCP",
        "retained_year2": "ENR_RET",
        "final_classification": "ENR_CLASS",
        "academic_year": "ENR_YEAR",
        "year_of_study": "ENR_YOS",
    },
    
    "modules": {
        "module_id": "MOD_CODE",
        "module_name": "MOD_NAME",
        "course_id": "MOD_CRS",
        "department": "MOD_DEPT",
        "credits": "MOD_CRED",
        "level": "MOD_LEVEL",
        "semester": "MOD_SEM",
    },
    
    "assessments": {
        "student_id": "ASS_SPR",
        "module_id": "ASS_MOD",
        "assessment_type": "ASS_TYPE",
        "mark": "ASS_MARK",
        "weight": "ASS_WGHT",
        "submitted": "ASS_SUB",
        "submission_date": "ASS_DATE",
        "late_submission": "ASS_LATE",
        "attempt": "ASS_ATT",
        "academic_year": "ASS_YEAR",
    },
    
    "attendance": {
        "student_id": "ATT_SPR",
        "module_id": "ATT_MOD",
        "academic_year": "ATT_YEAR",
        "week_number": "ATT_WEEK",
        "session_type": "ATT_TYPE",
        "attendance_status": "ATT_STAT",
        "recorded_date": "ATT_DATE",
    },
    
    "vle_engagement": {
        "student_id": "VLE_SPR",
        "module_id": "VLE_MOD",
        "academic_year": "VLE_YEAR",
        "week_number": "VLE_WEEK",
        "login_count": "VLE_LOGIN",
        "time_spent_minutes": "VLE_TIME",
        "resources_accessed": "VLE_RES",
        "forum_posts": "VLE_POST",
    },
}


# =============================================================================
# Tribal (SITS Alternative) Field Mappings
# =============================================================================

TRIBAL_FIELD_MAP: Dict[str, Dict[str, str]] = {
    "students": {
        "student_id": "STUDENT_ID",
        "gender": "GENDER",
        "ethnicity": "ETHNICITY",
        "date_of_birth": "DATE_OF_BIRTH",
        "postcode": "POSTCODE",
        "disability": "DISABILITY_CODE",
        "nationality": "NATIONALITY",
        "email": "EMAIL",
        "mobile": "MOBILE",
        "home_fee_status": "FEE_STATUS",
        "polar4_quintile": "POLAR_QUINTILE",
        "imd_decile": "IMD_DECILE",
        "application_id": "APPLICATION_ID",
        "offer_holder": "OFFER_MADE",
        "accepted_offer": "OFFER_ACCEPTED",
    },
    
    "qualifications": {
        "student_id": "STUDENT_ID",
        "qualification_type": "QUAL_TYPE",
        "subject": "SUBJECT",
        "grade": "GRADE",
        "ucas_tariff_points": "UCAS_POINTS",
        "predicted_grade": "PREDICTED_GRADE",
        "institution": "INSTITUTION",
        "year_completed": "YEAR_COMPLETED",
    },
    
    "courses": {
        "course_id": "COURSE_ID",
        "course_name": "COURSE_NAME",
        "department": "DEPARTMENT",
        "entry_tariff": "ENTRY_TARIFF",
        "course_length_years": "DURATION_YEARS",
        "exam_weight_pct": "EXAM_WEIGHTING",
        "coursework_weight_pct": "COURSEWORK_WEIGHTING",
        "employment_rate_15m": "EMPLOYMENT_RATE",
        "satisfaction_score": "NSS_SCORE",
        "accredited": "ACCREDITED",
    },
    
    "enrollments": {
        "student_id": "STUDENT_ID",
        "course_id": "COURSE_ID",
        "enrollment_date": "START_DATE",
        "enrollment_status": "STATUS",
        "accepted_offer": "OFFER_ACCEPTED",
        "retained_year2": "YEAR2_RETAINED",
        "final_classification": "CLASSIFICATION",
        "academic_year": "ACADEMIC_YEAR",
        "year_of_study": "YEAR_OF_STUDY",
    },
    
    "modules": {
        "module_id": "MODULE_ID",
        "module_name": "MODULE_NAME",
        "course_id": "COURSE_ID",
        "department": "DEPARTMENT",
        "credits": "CREDITS",
        "level": "LEVEL",
        "semester": "SEMESTER",
    },
    
    "assessments": {
        "student_id": "STUDENT_ID",
        "module_id": "MODULE_ID",
        "assessment_type": "ASSESSMENT_TYPE",
        "mark": "MARK",
        "weight": "WEIGHTING",
        "submitted": "SUBMITTED",
        "submission_date": "SUBMISSION_DATE",
        "late_submission": "LATE",
        "attempt": "ATTEMPT",
        "academic_year": "ACADEMIC_YEAR",
    },
    
    "attendance": {
        "student_id": "STUDENT_ID",
        "module_id": "MODULE_ID",
        "academic_year": "ACADEMIC_YEAR",
        "week_number": "WEEK_NO",
        "session_type": "SESSION_TYPE",
        "attendance_status": "ATTENDANCE",
        "recorded_date": "RECORDED_DATE",
    },
    
    "vle_engagement": {
        "student_id": "STUDENT_ID",
        "module_id": "MODULE_ID",
        "academic_year": "ACADEMIC_YEAR",
        "week_number": "WEEK_NO",
        "login_count": "LOGIN_COUNT",
        "time_spent_minutes": "TIME_ONLINE",
        "resources_accessed": "RESOURCES_ACCESSED",
        "forum_posts": "FORUM_POSTS",
    },
}


# =============================================================================
# Utility Functions
# =============================================================================

def get_synthetic_to_real_map(table_name: str, system: str = "sits") -> Dict[str, str]:
    """
    Get mapping from synthetic field names to real field names.
    
    Args:
        table_name: Name of the table (students, courses, etc.)
        system: "sits" or "tribal"
    
    Returns:
        Dict mapping synthetic -> real field names
    """
    if system.lower() == "sits":
        return SITS_FIELD_MAP.get(table_name, {})
    elif system.lower() == "tribal":
        return TRIBAL_FIELD_MAP.get(table_name, {})
    else:
        raise ValueError(f"Unknown system: {system}. Use 'sits' or 'tribal'")


def get_real_to_synthetic_map(table_name: str, system: str = "sits") -> Dict[str, str]:
    """
    Get mapping from real field names to synthetic field names.
    
    Args:
        table_name: Name of the table (students, courses, etc.)
        system: "sits" or "tribal"
    
    Returns:
        Dict mapping real -> synthetic field names
    """
    synthetic_to_real = get_synthetic_to_real_map(table_name, system)
    return {v: k for k, v in synthetic_to_real.items()}


def rename_to_synthetic(df, table_name: str, system: str = "sits") -> "pd.DataFrame":
    """
    Rename DataFrame columns from real field names to synthetic field names.
    
    Args:
        df: DataFrame with real field names
        table_name: Name of the table
        system: "sits" or "tribal"
    
    Returns:
        DataFrame with synthetic field names
    """
    import pandas as pd
    
    real_to_synthetic = get_real_to_synthetic_map(table_name, system)
    
    # Only rename columns that exist in both
    columns_to_rename = {
        k: v for k, v in real_to_synthetic.items() 
        if k in df.columns
    }
    
    return df.rename(columns=columns_to_rename)


def rename_to_real(df, table_name: str, system: str = "sits") -> "pd.DataFrame":
    """
    Rename DataFrame columns from synthetic field names to real field names.
    
    Args:
        df: DataFrame with synthetic field names
        table_name: Name of the table
        system: "sits" or "tribal"
    
    Returns:
        DataFrame with real field names
    """
    import pandas as pd
    
    synthetic_to_real = get_synthetic_to_real_map(table_name, system)
    
    # Only rename columns that exist in both
    columns_to_rename = {
        k: v for k, v in synthetic_to_real.items()
        if k in df.columns
    }
    
    return df.rename(columns=columns_to_rename)
