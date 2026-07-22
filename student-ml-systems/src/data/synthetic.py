"""
Synthetic data generation for student ML systems.

Generates realistic UK university data without using real student records.
Mimics SITS field structure for portfolio authenticity.

Field Mapping:
    This generator uses synthetic field names. To connect to real SITS or Tribal data,
    use the field mappings in `src/data/field_mapping.py`:
    
    - SITS:    `from src.data.field_mapping import SITS_FIELD_MAP`
    - Tribal:  `from src.data.field_mapping import TRIBAL_FIELD_MAP`
    
    Then use `rename_to_synthetic(df, 'students')` to convert real data.

Generates:
- Student demographics (SPR table)
- Qualifications (SCH table)
- Courses (CRS table)
- Enrollments (ENR table)
- Modules (MOD table)
- Assessments (ASS table)
- Attendance (ATT table)
- VLE engagement (VLE table)
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid

# Import field mappings for SITS/Tribal compatibility
from .field_mapping import SITS_FIELD_MAP, TRIBAL_FIELD_MAP, rename_to_synthetic


class SITSSyntheticGenerator:
    """Generate synthetic SITS-style student data."""

    def __init__(
        self,
        n_students: int = 10000,
        n_courses: int = 200,
        n_modules_per_course: int = 12,
        n_years: int = 5,
        seed: int = 42,
    ):
        """
        Initialize synthetic data generator.

        Args:
            n_students: Number of student records
            n_courses: Number of courses
            n_modules_per_course: Average modules per course
            n_years: Years of historical data
            seed: Random seed for reproducibility
        """
        self.n_students = n_students
        self.n_courses = n_courses
        self.n_modules_per_course = n_modules_per_course
        self.n_years = n_years
        self.seed = seed

        np.random.seed(seed)

        # UK university contextual data
        self.ethnicities = [
            "White British",
            "White Irish",
            "White Other",
            "Mixed",
            "Asian/Asian British",
            "Black/Black British",
            "Other Ethnic Group",
            "Not disclosed",
        ]
        self.ethnicity_probs = [0.65, 0.04, 0.10, 0.03, 0.08, 0.04, 0.03, 0.03]

        self.genders = ["Male", "Female", "Other", "Not disclosed"]
        self.gender_probs = [0.48, 0.50, 0.01, 0.01]

        self.qualifications = [
            "A-Level",
            "BTEC",
            "Access to HE",
            "International Baccalaureate",
            "Scottish Highers",
            "Mature Student",
            "Other",
        ]
        self.qual_probs = [0.55, 0.15, 0.08, 0.05, 0.05, 0.07, 0.05]

        self.departments = [
            "Engineering",
            "Computer Science",
            "Business",
            "Health Sciences",
            "Education",
            "Arts & Humanities",
            "Social Sciences",
            "Law",
            "Sciences",
        ]

        self.enrollment_statuses = [
            "Active",
            "Completed",
            "Withdrawn",
            "Suspended",
            "Transferred",
        ]

    def generate_students(self) -> pd.DataFrame:
        """
        Generate student demographic data (SPR table).

        Returns:
            DataFrame with student records
        """
        n = self.n_students

        # Generate student IDs (SPR.SPRN format)
        student_ids = [f"SPR{str(uuid.uuid4().int)[:8].upper()}" for _ in range(n)]

        # Demographics
        genders = np.random.choice(self.genders, n, p=self.gender_probs)
        ethnicities = np.random.choice(self.ethnicities, n, p=self.ethnicity_probs)

        # Dates of birth (18-50 years old)
        dob_start = datetime(1976, 1, 1)
        dob_end = datetime(2008, 12, 31)
        dob_range = (dob_end - dob_start).days
        dobs = [
            dob_start + timedelta(days=np.random.randint(0, dob_range))
            for _ in range(n)
        ]

        # Postcodes (UK format, synthetic)
        postcodes = [
            f"{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}"
            + f"{np.random.randint(1, 10)}"
            + f"{np.random.randint(1, 10)}"
            + f"{chr(np.random.randint(65, 91))}{chr(np.random.randint(65, 91))}"
            for _ in range(n)
        ]

        # IMD deciles (1=most deprived, 10=least deprived)
        # Skewed toward middle deciles
        imd_deciles = np.random.choice(range(1, 11), n, p=[0.05, 0.08, 0.10, 0.12, 0.15, 0.15, 0.12, 0.10, 0.08, 0.05])

        # POLAR4 quintiles (1=lowest participation, 5=highest)
        polar_quintiles = np.random.choice(range(1, 6), n, p=[0.15, 0.20, 0.25, 0.25, 0.15])

        # Care leaver flag (rare)
        care_leavers = np.random.binomial(1, 0.02, n)

        # Disability flag
        disabilities = np.random.binomial(1, 0.15, n)

        # First generation university (parents didn't attend)
        first_gen = np.random.binomial(1, 0.45, n)

        df = pd.DataFrame(
            {
                "student_id": student_ids,
                "surname": [f"Student_{i}" for i in range(n)],
                "forename": [f"Student_{i}" for i in range(n)],
                "gender": genders,
                "ethnicity": ethnicities,
                "date_of_birth": dobs,
                "postcode": postcodes,
                "imd_decile": imd_deciles,
                "polar_quintile": polar_quintiles,
                "care_leaver": care_leavers,
                "disability": disabilities,
                "first_generation_uni": first_gen,
            }
        )

        return df

    def generate_qualifications(self, student_ids: list) -> pd.DataFrame:
        """
        Generate prior qualifications (SCH table).

        Args:
            student_ids: List of student IDs

        Returns:
            DataFrame with qualification records
        """
        n = len(student_ids)

        # Qualification types
        qual_types = np.random.choice(self.qualifications, n, p=self.qual_probs)

        # Generate grades based on qualification type
        grades = []
        tariff_points = []

        for qual in qual_types:
            if qual == "A-Level":
                grade_options = ["AAA", "AAB", "ABB", "BBB", "BCC", "CCC", "CDD", "DDD"]
                grade_probs = [0.05, 0.10, 0.20, 0.25, 0.20, 0.10, 0.07, 0.03]
                tariff_map = {
                    "AAA": 168,
                    "AAB": 152,
                    "ABB": 136,
                    "BBB": 120,
                    "BCC": 104,
                    "CCC": 88,
                    "CDD": 72,
                    "DDD": 56,
                }
            elif qual == "BTEC":
                grade_options = ["D*D*D*", "D*D*D", "D*DD", "DDD", "DDM", "DMM", "MMM", "MPP"]
                grade_probs = [0.03, 0.07, 0.10, 0.15, 0.25, 0.20, 0.15, 0.05]
                tariff_map = {
                    "D*D*D*": 168,
                    "D*D*D": 160,
                    "D*DD": 152,
                    "DDD": 144,
                    "DDM": 136,
                    "DMM": 128,
                    "MMM": 112,
                    "MPP": 80,
                }
            elif qual == "Access to HE":
                grade_options = ["Distinction", "Merit", "Pass"]
                grade_probs = [0.30, 0.50, 0.20]
                tariff_map = {"Distinction": 144, "Merit": 112, "Pass": 80}
            elif qual == "International Baccalaureate":
                grade_options = [str(i) for i in range(28, 46)]
                grade_probs = np.random.dirichlet(np.ones(18)).tolist()
                tariff_map = {str(i): i * 4 for i in range(28, 46)}
            else:
                grade_options = ["Pass", "Credit", "Distinction"]
                grade_probs = [0.40, 0.40, 0.20]
                tariff_map = {"Pass": 80, "Credit": 112, "Distinction": 144}

            grade = np.random.choice(grade_options, p=grade_probs)
            grades.append(grade)
            tariff_points.append(tariff_map.get(grade, 100))

        # Predicted vs actual (some students have predicted grades)
        predicted = np.random.binomial(1, 0.70, n)  # 70% predicted at application

        df = pd.DataFrame(
            {
                "student_id": student_ids,
                "qualification_type": qual_types,
                "grade": grades,
                "ucas_tariff_points": tariff_points,
                "predicted_grade": predicted.astype(bool),
            }
        )

        return df

    def generate_courses(self) -> pd.DataFrame:
        """
        Generate course database (CRS table).

        Returns:
            DataFrame with course records
        """
        n = self.n_courses

        # Course IDs
        course_ids = [f"CRS{str(i).zfill(5)}" for i in range(n)]

        # Course names
        course_prefixes = [
            "BSc",
            "BA",
            "BEng",
            "MEng",
            "MSc",
            "MA",
            "LLB",
            "BEd",
            "BNurs",
        ]
        course_subjects = [
            "Computer Science",
            "Mechanical Engineering",
            "Business Management",
            "Nursing",
            "Education Studies",
            "English Literature",
            "Psychology",
            "Law",
            "Biology",
            "Mathematics",
            "Physics",
            "Chemistry",
            "Economics",
            "Sociology",
            "History",
            "Marketing",
            "Finance",
            "Civil Engineering",
            "Electrical Engineering",
            "Data Science",
        ]

        course_names = [
            f"{np.random.choice(course_prefixes)} {np.random.choice(course_subjects)}"
            for _ in range(n)
        ]

        # Departments
        departments = np.random.choice(self.departments, n)

        # Entry requirements (UCAS tariff)
        entry_tariffs = np.random.choice(
            [80, 96, 112, 120, 128, 136, 144, 152, 160, 168],
            n,
            p=[0.05, 0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.03, 0.01, 0.01],
        )

        # Course length
        course_lengths = np.random.choice([3, 4], n, p=[0.85, 0.15])  # 15% sandwich

        # Assessment split
        exam_weights = np.random.uniform(20, 80, n)
        coursework_weights = 100 - exam_weights

        # Employment rates (realistic variation by department)
        base_employment = {
            "Engineering": 0.92,
            "Computer Science": 0.90,
            "Business": 0.85,
            "Health Sciences": 0.95,
            "Education": 0.88,
            "Arts & Humanities": 0.78,
            "Social Sciences": 0.82,
            "Law": 0.87,
            "Sciences": 0.85,
        }
        employment_rates = [
            base_employment[dept] + np.random.uniform(-0.05, 0.05)
            for dept in departments
        ]
        employment_rates = np.clip(employment_rates, 0.70, 0.98)

        # Student satisfaction (NSS-style, 1-5 scale)
        satisfaction_scores = np.random.uniform(3.5, 4.8, n)

        # Accreditation flags
        accredited = np.random.binomial(1, 0.40, n)  # 40% professionally accredited

        df = pd.DataFrame(
            {
                "course_id": course_ids,
                "course_name": course_names,
                "department": departments,
                "entry_tariff": entry_tariffs,
                "course_length_years": course_lengths,
                "exam_weight_pct": exam_weights.astype(int),
                "coursework_weight_pct": coursework_weights.astype(int),
                "employment_rate_15m": employment_rates,
                "satisfaction_score": satisfaction_scores.round(1),
                "accredited": accredited.astype(bool),
            }
        )

        return df

    def generate_enrollments(
        self, student_ids: list, course_ids: list
    ) -> pd.DataFrame:
        """
        Generate enrollment records (ENR table).

        Args:
            student_ids: List of student IDs
            course_ids: List of course IDs

        Returns:
            DataFrame with enrollment records
        """
        n = len(student_ids)

        # Each student enrolls in one course
        enrollments = []

        for i, student_id in enumerate(student_ids):
            # Course assignment (weighted by entry tariff match)
            course_idx = np.random.randint(0, len(course_ids))
            course_id = course_ids[course_idx]

            # Enrollment date (academic year start: September)
            year = np.random.randint(2019, 2024)
            enrollment_date = datetime(year, 9, 1) + timedelta(days=np.random.randint(0, 30))

            # Status (most students active/completed, some withdrawn)
            status_probs = [0.60, 0.25, 0.10, 0.03, 0.02]
            status = np.random.choice(self.enrollment_statuses, p=status_probs)

            # Retention to year 2 (if applicable)
            if status in ["Active", "Completed"]:
                retained = np.random.binomial(1, 0.90)  # 90% retention
            else:
                retained = 0

            # Final classification (for completed students)
            if status == "Completed":
                classification_probs = [0.15, 0.35, 0.35, 0.10, 0.05]
                classification = np.random.choice(
                    ["First", "2:1", "2:2", "Third", "Pass"],
                    p=classification_probs,
                )
            else:
                classification = None

            enrollments.append(
                {
                    "student_id": student_id,
                    "course_id": course_id,
                    "enrollment_date": enrollment_date,
                    "enrollment_status": status,
                    "retained_year2": retained,
                    "final_classification": classification,
                    "academic_year": year,
                }
            )

        df = pd.DataFrame(enrollments)
        return df

    def generate_modules(
        self, course_ids: list, n_modules: int = 12
    ) -> pd.DataFrame:
        """
        Generate module records (MOD table).

        Args:
            course_ids: List of course IDs
            n_modules: Modules per course

        Returns:
            DataFrame with module records
        """
        modules = []

        module_prefixes = ["Introduction to", "Advanced", "Applied", "Foundations of", "Topics in"]
        module_topics = [
            "Programming",
            "Data Structures",
            "Algorithms",
            "Machine Learning",
            "Database Systems",
            "Web Development",
            "Software Engineering",
            "Computer Networks",
            "Operating Systems",
            "Artificial Intelligence",
            "Statistics",
            "Mathematics",
            "Research Methods",
            "Project Management",
            "Professional Practice",
        ]

        for course_id in course_ids:
            for i in range(n_modules):
                module_id = f"{course_id}_MOD{str(i + 1).zfill(2)}"
                module_name = f"{np.random.choice(module_prefixes)} {np.random.choice(module_topics)}"

                # Year of study (1, 2, or 3)
                year = np.random.choice([1, 2, 3], p=[0.40, 0.35, 0.25])

                # Credits (typically 10, 15, or 20)
                credits = np.random.choice([10, 15, 20], p=[0.30, 0.50, 0.20])

                # Assessment split
                exam_weight = np.random.uniform(20, 80)
                coursework_weight = 100 - exam_weight

                # Average mark (by year - improves over time)
                base_mark = 60 + year * 3  # Year 1: ~63%, Year 3: ~69%
                avg_mark = base_mark + np.random.uniform(-10, 10)
                avg_mark = np.clip(avg_mark, 40, 95)

                # Pass rate
                pass_rate = np.random.uniform(0.85, 0.98)

                modules.append(
                    {
                        "module_id": module_id,
                        "course_id": course_id,
                        "module_name": module_name,
                        "year": year,
                        "credits": credits,
                        "exam_weight_pct": exam_weight,
                        "coursework_weight_pct": coursework_weight,
                        "average_mark": avg_mark.round(1),
                        "pass_rate": pass_rate,
                    }
                )

        df = pd.DataFrame(modules)
        return df

    def generate_assessments(
        self, student_ids: list, module_ids: list
    ) -> pd.DataFrame:
        """
        Generate assessment records (ASS table).

        Args:
            student_ids: List of student IDs
            module_ids: List of module IDs

        Returns:
            DataFrame with assessment records
        """
        assessments = []

        # Each student takes ~12 modules
        n_assessments = len(student_ids) * self.n_modules_per_course

        # Pre-compute each student's innate ability effect (std=22 around 0)
        # This creates wide between-student variation in marks.
        # With ability_std=22 and assessment_std=15:
        #   total variance ≈ 22^2 + 15^2 = 709 → std ≈ 26.6
        #   Students with ability ≥ 5 average ≥ 70 (First territory)
        #   Students with ability ≤ -25 average ≤ 40 (Fail territory)
        _rng = np.random.RandomState(self.seed + 999)
        student_abilities = {
            sid: _rng.normal(0, 22) for sid in student_ids
        }

        for _ in range(n_assessments):
            student_id = np.random.choice(student_ids)
            module_id = np.random.choice(module_ids)

            # Assessment type
            assessment_type = np.random.choice(
                ["Coursework", "Exam", "Practical", "Presentation", "Dissertation"],
                p=[0.40, 0.30, 0.15, 0.10, 0.05],
            )

            # Mark: base variation + per-student innate ability effect
            ability = student_abilities[student_id]
            mark = np.random.normal(65 + ability, 15)
            mark = np.clip(mark, 0, 100)

            # Submission status
            submitted = np.random.binomial(1, 0.92)  # 92% submission rate
            if not submitted:
                mark = 0

            # Late submission penalty
            if submitted and np.random.binomial(1, 0.08):  # 8% late
                mark = mark * 0.90  # 10% penalty

            # Attempt number (first, resit)
            attempt = np.random.choice([1, 2], p=[0.90, 0.10])

            # Ensure mark is float before rounding
            mark = float(mark) if not isinstance(mark, (float, np.floating)) else mark
            
            assessments.append(
                {
                    "student_id": student_id,
                    "module_id": module_id,
                    "assessment_type": assessment_type,
                    "mark": round(mark, 1),
                    "submitted": submitted,
                    "late_submission": not submitted or np.random.binomial(1, 0.08),
                    "attempt": attempt,
                }
            )

        df = pd.DataFrame(assessments)
        return df

    def generate_attendance(self, student_ids: list, n_weeks: int = 30) -> pd.DataFrame:
        """
        Generate attendance records (ATT table).

        Args:
            student_ids: List of student IDs
            n_weeks: Weeks per academic year

        Returns:
            DataFrame with attendance records
        """
        attendance_records = []

        # Each student has weekly attendance
        for student_id in student_ids:
            # Base attendance rate (varies by student)
            base_attendance = np.random.uniform(0.50, 0.98)

            for week in range(1, n_weeks + 1):
                # Attendance date
                date = datetime(2023, 9, 1) + timedelta(weeks=week)

                # Attendance status (correlated with base rate)
                attendance_prob = base_attendance + np.random.uniform(-0.10, 0.10)
                attendance_prob = np.clip(attendance_prob, 0.30, 0.99)
                attended = np.random.binomial(1, attendance_prob)

                if attended:
                    status = "Present"
                else:
                    status = np.random.choice(
                        ["Absent", "Authorised Absence", "Medical"],
                        p=[0.50, 0.35, 0.15],
                    )

                attendance_records.append(
                    {
                        "student_id": student_id,
                        "week": week,
                        "date": date,
                        "status": status,
                        "attended": attended,
                    }
                )

        df = pd.DataFrame(attendance_records)
        return df

    def generate_vle_engagement(
        self, student_ids: list, n_weeks: int = 30
    ) -> pd.DataFrame:
        """
        Generate VLE (Virtual Learning Environment) engagement logs.

        Args:
            student_ids: List of student IDs
            n_weeks: Weeks of data

        Returns:
            DataFrame with VLE engagement metrics per student per week
        """
        vle_records = []

        for student_id in student_ids:
            # Base engagement level (varies by student)
            base_engagement = np.random.uniform(0.30, 0.95)

            for week in range(1, n_weeks + 1):
                # Weekly engagement (correlated with base, with trend)
                week_effect = 1.0 - (week / n_weeks) * 0.20  # Slight decline over term
                engagement = base_engagement * week_effect + np.random.uniform(-0.15, 0.15)
                engagement = np.clip(engagement, 0.10, 0.99)

                # Metrics
                logins = int(np.random.poisson(5 * engagement))
                resources_accessed = int(np.random.poisson(10 * engagement))
                forum_posts = int(np.random.poisson(2 * engagement))
                quiz_attempts = int(np.random.poisson(3 * engagement))
                video_views = int(np.random.poisson(8 * engagement))

                vle_records.append(
                    {
                        "student_id": student_id,
                        "week": week,
                        "logins": logins,
                        "resources_accessed": resources_accessed,
                        "forum_posts": forum_posts,
                        "quiz_attempts": quiz_attempts,
                        "video_views": video_views,
                        "total_actions": logins + resources_accessed + forum_posts + quiz_attempts + video_views,
                    }
                )

        df = pd.DataFrame(vle_records)
        return df

    def generate_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Generate all synthetic datasets.

        Returns:
            Dictionary of DataFrames
        """
        print("Generating synthetic student data...")
        print(f"  - {self.n_students} students")
        print(f"  - {self.n_courses} courses")
        print(f"  - {self.n_years} years of history")

        # Generate base tables
        students_df = self.generate_students()
        print(f"✓ Generated {len(students_df)} student records")

        qualifications_df = self.generate_qualifications(students_df["student_id"].tolist())
        print(f"✓ Generated {len(qualifications_df)} qualification records")

        courses_df = self.generate_courses()
        print(f"✓ Generated {len(courses_df)} course records")

        enrollments_df = self.generate_enrollments(
            students_df["student_id"].tolist(),
            courses_df["course_id"].tolist(),
        )
        print(f"✓ Generated {len(enrollments_df)} enrollment records")

        modules_df = self.generate_modules(courses_df["course_id"].tolist())
        print(f"✓ Generated {len(modules_df)} module records")

        assessments_df = self.generate_assessments(
            students_df["student_id"].tolist(),
            modules_df["module_id"].tolist(),
        )
        print(f"✓ Generated {len(assessments_df)} assessment records")

        attendance_df = self.generate_attendance(students_df["student_id"].tolist())
        print(f"✓ Generated {len(attendance_df)} attendance records")

        vle_df = self.generate_vle_engagement(students_df["student_id"].tolist())
        print(f"✓ Generated {len(vle_df)} VLE engagement records")

        return {
            "students": students_df,
            "qualifications": qualifications_df,
            "courses": courses_df,
            "enrollments": enrollments_df,
            "modules": modules_df,
            "assessments": assessments_df,
            "attendance": attendance_df,
            "vle_engagement": vle_df,
        }

    def generate_degree_outcomes(
        self,
        assessments_df: pd.DataFrame,
        students_df: pd.DataFrame,
        courses_df: pd.DataFrame,
        enrollments_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate final degree classification outcomes.

        Maps assessment marks to degree classifications using credit-weighted GPA,
        with noise derived from VLE/attendance engagement patterns to create
        realistic correlations between engagement and outcomes.

        Args:
            assessments_df: Assessment records with student_id, module_id, mark, credits
            students_df: Student demographics (used for engagement signal noise)
            courses_df: Course records (used for course-level difficulty adjustment)
            enrollments_df: Enrollment records (maps students to courses)

        Returns:
            DataFrame with columns:
            - student_id: Student identifier
            - course_id: Course identifier
            - final_classification: One of 'First', '2:1', '2:2', 'Third', 'Fail'
            - weighted_gpa: Credit-weighted average mark (0-4 scale)
            - years_to_complete: Actual years taken (with some early/late completers)
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Generating degree classification outcomes...")

        # Map module_id -> course_id and credits from modules
        # assessments reference module_id, we need course_id and credits
        # Build a module lookup from the assessments themselves using course context
        # Since assessments don't directly carry course_id, we infer from enrollment

        # Get student -> course mapping from enrollments
        student_course = enrollments_df[["student_id", "course_id"]].copy()
        student_course = student_course.drop_duplicates(subset=["student_id"])

        # Build a credit map per module from assessments
        # We estimate credits as: assume modules with many assessments have more credits
        # Better approach: use the module data if available via course_id join
        # For synthetic, we'll attribute credits based on assessment weight patterns

        # Join assessments with student->course mapping
        df = assessments_df.merge(student_course, on="student_id", how="inner")

        if len(df) == 0:
            logger.warning("No matching enrollments found for assessments")
            return pd.DataFrame(
                columns=[
                    "student_id",
                    "course_id",
                    "final_classification",
                    "weighted_gpa",
                    "years_to_complete",
                ]
            )

        # Estimate module credits: use assessment count per module as proxy
        # Modules with more assessments get more credits (realistic distribution)
        module_assessment_counts = df.groupby("module_id").size()
        credit_estimates = {}
        for module_id, count in module_assessment_counts.items():
            # Each assessment ~represents a portion of a module's credit load
            # Standard: 10, 15, or 20 credits per module
            credit_estimates[module_id] = np.random.choice(
                [10, 15, 20], p=[0.30, 0.50, 0.20]
            )

        df["credits"] = df["module_id"].map(credit_estimates)

        # --- Compute per-student credit-weighted GPA ---
        # GPA = sum(mark * credits) / sum(credits), mapped to 0-4 scale
        df["mark_scaled"] = df["mark"] / 25.0  # 0-100 → 0-4
        df["weighted_mark"] = df["mark_scaled"] * df["credits"]

        student_stats = df.groupby("student_id").agg(
            total_weighted_mark=("weighted_mark", "sum"),
            total_credits=("credits", "sum"),
            course_id=("course_id", "first"),
            n_assessments=("mark", "count"),
            avg_mark=("mark", "mean"),
            std_mark=("mark", "std"),
        ).reset_index()

        # --- Raw GPA from credit-weighted marks ---
        student_stats["raw_gpa"] = (
            student_stats["total_weighted_mark"] / student_stats["total_credits"]
        )

        # Course-level difficulty: each course has a base mark adjustment ±10
        course_base_marks = {}
        for course_id in student_stats["course_id"].unique():
            # Realistic variation: engineering/cs harder, business softer
            course_base_marks[course_id] = np.random.normal(0, 5)  # ±5 mark adjustment

        student_stats["course_difficulty"] = student_stats["course_id"].map(course_base_marks)

        # --- Build engagement signal for noise adjustment ---
        # Students with high engagement get a small mark boost
        engagement_boost = np.random.normal(0.08, 0.12, len(student_stats))
        # Loosely correlate with performance proxy (high performers tend to engage more)
        performance_proxy = (student_stats["avg_mark"] - 65) / 20.0
        engagement_boost += 0.35 * performance_proxy
        student_stats["engagement_noise"] = np.clip(engagement_boost, -0.4, 0.4)

        # --- Widen the mark distribution to realistic extremes ---
        # Raw marks cluster around 65%; we need Firsts (≥70%) and Fails (<40%).
        # Apply a signed deviation push: extreme students get pushed further out.
        student_stats["adjusted_gpa"] = student_stats["raw_gpa"] + (
            student_stats["course_difficulty"] / 25.0
        )
        # deviation > 0 → push up (helps Firsts); deviation < 0 → push down (helps Fails)
        deviation_from_mean = student_stats["adjusted_gpa"] - 2.6
        student_stats["adjusted_gpa"] = (
            student_stats["adjusted_gpa"] + 0.30 * deviation_from_mean
        ).clip(0.0, 4.0)

        # --- Final GPA with engagement noise + targeted class balancing ---
        # Blend adjusted GPA with engagement signal and add a small jitter
        # to avoid mass concentration at class boundaries
        _jitter_rng = np.random.RandomState(self.seed + 7)
        student_stats["final_gpa"] = (
            0.80 * student_stats["adjusted_gpa"]
            + 0.20 * student_stats["engagement_noise"]
            + _jitter_rng.normal(0, 0.05, len(student_stats))
        ).clip(0.0, 4.0)

        # --- Targeted percentile boosts to ensure all 5 classification appear ---
        # Compute GPA percentiles and selectively shift extreme students into each class
        # to match realistic UK university distribution (~15% First, ~35% 2:1, ~35% 2:2, ~10% Third, ~5% Fail)
        gpa_series = student_stats["final_gpa"]

        # First-class: push top ~4% above 3.7 threshold
        pct_96 = gpa_series.quantile(0.96)
        mask_first = (gpa_series >= pct_96) & (gpa_series < 3.7)
        student_stats.loc[mask_first, "final_gpa"] = (
            gpa_series[mask_first] + (3.75 - gpa_series[mask_first].min()) * 0.5
        ).clip(3.70, 4.0)

        # 2:1 zone: ensure top-remaining students above 3.0
        mask_2_1 = (gpa_series < pct_96) & (gpa_series >= gpa_series.quantile(0.60))
        # Fine: leave these as-is

        # Fail: push bottom ~5% below 1.0 threshold
        pct_05 = gpa_series.quantile(0.05)
        mask_fail = (gpa_series <= pct_05) & (gpa_series >= 1.0)
        student_stats.loc[mask_fail, "final_gpa"] = (
            gpa_series[mask_fail] - (gpa_series[mask_fail].max() - 0.85) * 0.5
        ).clip(0.0, 0.99)

        # Map GPA to degree classification
        def gpa_to_classification(gpa: float) -> str:
            if gpa >= 3.7:
                return "First"
            elif gpa >= 3.0:
                return "2:1"
            elif gpa >= 2.0:
                return "2:2"
            elif gpa >= 1.0:
                return "Third"
            else:
                return "Fail"

        student_stats["final_classification"] = student_stats["final_gpa"].apply(gpa_to_classification)
        student_stats["weighted_gpa"] = student_stats["final_gpa"].round(3)

        # Years to complete (course length with some variation)
        course_lengths = courses_df.set_index("course_id")["course_length_years"].to_dict()
        student_stats["course_length"] = student_stats["course_id"].map(course_lengths).fillna(3)

        # Most students complete on time, some early/late
        years_complete = []
        for _, row in student_stats.iterrows():
            base_years = row["course_length"]
            if row["final_classification"] == "Fail":
                # Fails often take longer (resits) or don't complete
                years = base_years + np.random.choice([0, 1, 2], p=[0.2, 0.5, 0.3])
            else:
                years = base_years + np.random.choice([-1, 0, 0, 1], p=[0.05, 0.15, 0.70, 0.10])
            years = max(1, min(years, base_years + 2))
            years_complete.append(years)

        student_stats["years_to_complete"] = years_complete

        result = student_stats[[
            "student_id",
            "course_id",
            "final_classification",
            "weighted_gpa",
            "years_to_complete",
        ]].copy()

        logger.info(
            f"Generated {len(result)} degree outcomes: "
            f"{result['final_classification'].value_counts().to_dict()}"
        )

        return result

    def generate_graduate_outcomes(
        self,
        students_df: pd.DataFrame,
        degree_outcomes_df: pd.DataFrame,
        assessments_df: pd.DataFrame,
        vle_df: pd.DataFrame,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Generate graduate outcome data (employment, salary, further study).

        Uses degree outcome + engagement to predict employment probability.
        Real data: HESA DLHE (Destinations of Leavers from Higher Education).

        Args:
            students_df: Student records
            degree_outcomes_df: Degree classification results
            assessments_df: Assessment records
            vle_df: VLE engagement records
            seed: Random seed

        Returns:
            DataFrame with graduate outcomes per student
        """
        if seed is not None:
            np.random.seed(seed)

        df = students_df.copy()
        n = len(df)

        # Merge degree outcome
        df = df.merge(
            degree_outcomes_df[['student_id', 'final_classification', 'weighted_gpa']],
            on='student_id',
            how='left'
        )
        # Rename weighted_gpa -> gpa for consistency
        df['gpa'] = df['weighted_gpa']
        df['gpa'] = df['gpa'].fillna(df['gpa'].median())

        # Employment probability based STRONGLY on degree + demographics
        # Strong signal: degree directly maps to employment probability
        def emp_probability(row):
            gpa = row['gpa']
            # GPA drives employment: 0.95 for First, down to 0.4 for Fail
            degree_emp = {'First': 0.95, '2:1': 0.85, '2:2': 0.70, 'Third': 0.55, 'Fail': 0.35}
            base = degree_emp.get(row['final_classification'], 0.6)
            # Add some noise but keep signal strong (reduced from ±0.1 to ±0.05)
            noise = np.random.uniform(-0.05, 0.05)
            return max(0.2, min(0.98, base + noise))

        df['employment_prob'] = df.apply(emp_probability, axis=1)

        # Employment status: strong correlation with employment_prob
        emp_rnd = np.random.uniform(0, 1, n)
        emp_prob_arr = df['employment_prob'].values
        employment_status_arr = np.where(
            emp_rnd < emp_prob_arr * 0.5, 'Employed',  # High prob → Employed
            np.where(
                emp_rnd < emp_prob_arr * 0.75, 'Both Employed and Study',
                np.where(
                    emp_rnd < emp_prob_arr * 0.90, 'Further Study Only',
                    'Unemployed'
                )
            )
        )
        df['employment_status'] = employment_status_arr

        # Salary band: VERY strongly correlated with degree
        def salary_from_outcomes(row):
            degree_salary = {'First': 3.3, '2:1': 2.7, '2:2': 1.8, 'Third': 1.2, 'Fail': 0.5}
            base_score = degree_salary.get(row['final_classification'], 1.5)
            # Very low noise for strong signal
            score = base_score + np.random.uniform(-0.3, 0.3)
            if score < 1.0:
                return 'Under £20,000'
            elif score < 2.0:
                return '£20,000 - £30,000'
            elif score < 3.0:
                return '£30,000 - £40,000'
            else:
                return 'Over £40,000'

        df['salary_band'] = df.apply(salary_from_outcomes, axis=1)

        # Further study: negatively correlated with employment
        def study_dest(row):
            if row['employment_status'] == 'Employed':
                # Employed students rarely study further
                r = np.random.random()
                if r < 0.85:
                    return 'Not Studying'
                elif r < 0.95:
                    return 'UK'
                else:
                    return 'EU'
            else:
                # Unemployed study more
                r = np.random.random()
                if r < 0.50:
                    return 'UK'
                elif r < 0.75:
                    return 'EU'
                else:
                    return 'International'

        df['study_destination'] = df.apply(study_dest, axis=1)

        # Binary employment (for model feature)
        df['is_employed'] = df['employment_status'].isin(['Employed', 'Both Employed and Study']).astype(int)

        # Degree class ordinal (for model feature)
        deg_map = {'Fail': 0, 'Third': 1, '2:2': 2, '2:1': 3, 'First': 4}
        df['degree_class_ordinal'] = df['final_classification'].map(deg_map).fillna(2)

        # Career readiness score (0-100) — based on engagement + degree
        vle_engagement = vle_df.groupby('student_id').agg({
            'logins': 'sum',
            'resources_accessed': 'sum'
        }).reset_index()
        vle_engagement.columns = ['student_id', 'total_logins', 'total_resources']

        df = df.merge(vle_engagement, on='student_id', how='left')
        df['total_logins'] = df['total_logins'].fillna(0)
        df['total_resources'] = df['total_resources'].fillna(0)

        # Career readiness: degree matters most, engagement adds some signal
        df['career_readiness_score'] = (
            (df['degree_class_ordinal'] / 4.0 * 60) +  # Degree drives readiness
            (df['total_logins'] / df['total_logins'].max() * 40)  # Engagement adds signal
        ).clip(0, 100)

        return df[[
            'student_id', 'employment_status', 'salary_band', 'study_destination',
            'career_readiness_score', 'is_employed', 'degree_class_ordinal',
            'total_logins', 'total_resources'
        ]]


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    generator = SITSSyntheticGenerator(
        n_students=1000,  # Smaller for demo
        n_courses=50,
        n_years=3,
        seed=42,
    )

    datasets = generator.generate_all_datasets()

    # Display summaries
    print("\n" + "=" * 60)
    print("DATASET SUMMARIES")
    print("=" * 60)

    for name, df in datasets.items():
        print(f"\n{name.upper()}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Sample:\n{df.head(3)}")

    # Save to CSV (optional)
    # for name, df in datasets.items():
    #     df.to_csv(f"data/synthetic/{name}.csv", index=False)
