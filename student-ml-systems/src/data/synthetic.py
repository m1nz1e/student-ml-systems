"""
Synthetic data generation for student ML systems.

Generates realistic UK university data without using real student records.
Mimics SITS field structure for portfolio authenticity.

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

from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid


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

        for _ in range(n_assessments):
            student_id = np.random.choice(student_ids)
            module_id = np.random.choice(module_ids)

            # Assessment type
            assessment_type = np.random.choice(
                ["Coursework", "Exam", "Practical", "Presentation", "Dissertation"],
                p=[0.40, 0.30, 0.15, 0.10, 0.05],
            )

            # Mark (normal distribution, mean ~65%)
            mark = np.random.normal(65, 15)
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

            assessments.append(
                {
                    "student_id": student_id,
                    "module_id": module_id,
                    "assessment_type": assessment_type,
                    "mark": mark.round(1),
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
