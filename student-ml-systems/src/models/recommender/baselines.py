"""
Baseline models for Course Recommender System.

Implements simple baseline recommenders for comparison:
1. Random Baseline — Uniform random recommendations
2. Popularity Baseline — Most enrolled courses
3. Content-Based v1 — Simple cosine similarity

These baselines establish minimum performance thresholds
that advanced models must exceed.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomRecommender:
    """
    Random baseline recommender.

    Recommends courses uniformly at random.
    Useful as a lower bound — any real model should beat this.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize random recommender.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.course_ids: Optional[List[str]] = None
        np.random.seed(seed)

    def fit(self, courses_df: pd.DataFrame, **kwargs):
        """
        Fit recommender (store course list).

        Args:
            courses_df: Course DataFrame with course_id column
        """
        self.course_ids = courses_df["course_id"].unique().tolist()
        logger.info(f"RandomRecommender fitted with {len(self.course_ids)} courses")
        return self

    def predict(
        self,
        user_features: pd.DataFrame,
        n_recommendations: int = 10,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Generate random recommendations for users.

        Args:
            user_features: User feature DataFrame
            n_recommendations: Number of recommendations per user

        Returns:
            Dictionary mapping user_id to list of (course_id, score) tuples
        """
        if self.course_ids is None:
            raise ValueError("Must call fit() before predict()")

        recommendations = {}

        for _, user_row in user_features.iterrows():
            user_id = user_row.get("student_id", user_row.get("user_id"))

            # Random scores for all courses
            random_scores = np.random.uniform(0, 1, len(self.course_ids))

            # Get top N
            top_indices = np.argsort(random_scores)[::-1][:n_recommendations]
            top_courses = [self.course_ids[i] for i in top_indices]
            top_scores = [random_scores[i] for i in top_indices]

            recommendations[user_id] = list(zip(top_courses, top_scores))

        return recommendations

    def __str__(self):
        return "RandomRecommender"


class PopularityRecommender:
    """
    Popularity-based baseline recommender.

    Recommends the most popular courses (by enrollment count).
    Simple but often surprisingly effective baseline.
    """

    def __init__(self, popularity_metric: str = "enrollment_count"):
        """
        Initialize popularity recommender.

        Args:
            popularity_metric: One of ['enrollment_count', 'employment_rate', 'satisfaction']
        """
        self.popularity_metric = popularity_metric
        self.course_popularity: Optional[pd.DataFrame] = None
        self.course_ids: Optional[List[str]] = None

    def fit(
        self,
        courses_df: pd.DataFrame,
        enrollments_df: Optional[pd.DataFrame] = None,
        **kwargs,
    ):
        """
        Fit recommender (compute course popularity).

        Args:
            courses_df: Course DataFrame
            enrollments_df: Optional enrollment data for computing popularity
        """
        logger.info(f"Fitting PopularityRecommender (metric={self.popularity_metric})...")

        self.course_ids = courses_df["course_id"].unique().tolist()

        if self.popularity_metric == "enrollment_count":
            if enrollments_df is not None:
                # Count enrollments per course
                popularity = (
                    enrollments_df.groupby("course_id")
                    .size()
                    .reset_index(name="enrollment_count")
                )
                self.course_popularity = courses_df.merge(
                    popularity, on="course_id", how="left"
                )
                self.course_popularity["enrollment_count"] = (
                    self.course_popularity["enrollment_count"].fillna(0)
                )
            else:
                # Use employment rate as proxy
                logger.warning(
                    "No enrollment data provided, using employment_rate as proxy"
                )
                self.course_popularity = courses_df.copy()
                self.course_popularity["enrollment_count"] = self.course_popularity[
                    "employment_rate_15m"
                ]

        elif self.popularity_metric == "employment_rate":
            self.course_popularity = courses_df.copy()
            self.course_popularity["popularity_score"] = self.course_popularity[
                "employment_rate_15m"
            ]

        elif self.popularity_metric == "satisfaction":
            self.course_popularity = courses_df.copy()
            self.course_popularity["popularity_score"] = self.course_popularity[
                "satisfaction_score"
            ]

        else:
            raise ValueError(
                f"Invalid popularity_metric: {self.popularity_metric}. "
                "Must be one of ['enrollment_count', 'employment_rate', 'satisfaction']"
            )

        # Sort by popularity
        if "popularity_score" in self.course_popularity.columns:
            self.course_popularity = self.course_popularity.sort_values(
                "popularity_score", ascending=False
            )
        else:
            self.course_popularity = self.course_popularity.sort_values(
                "enrollment_count", ascending=False
            )

        logger.info(
            f"PopularityRecommender fitted with {len(self.course_popularity)} courses"
        )
        return self

    def predict(
        self,
        user_features: pd.DataFrame,
        n_recommendations: int = 10,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Generate popularity-based recommendations for all users.

        Same recommendations for everyone (global popularity).

        Args:
            user_features: User feature DataFrame (ignored, same recs for all)
            n_recommendations: Number of recommendations per user

        Returns:
            Dictionary mapping user_id to list of (course_id, score) tuples
        """
        if self.course_popularity is None:
            raise ValueError("Must call fit() before predict()")

        recommendations = {}

        # Get top N popular courses
        if "popularity_score" in self.course_popularity.columns:
            top_courses = self.course_popularity.head(n_recommendations)
            course_ids = top_courses["course_id"].tolist()
            scores = top_courses["popularity_score"].tolist()
        else:
            top_courses = self.course_popularity.head(n_recommendations)
            course_ids = top_courses["course_id"].tolist()
            scores = top_courses["enrollment_count"].tolist()

        # Same recommendations for all users
        for _, user_row in user_features.iterrows():
            user_id = user_row.get("student_id", user_row.get("user_id"))
            recommendations[user_id] = list(zip(course_ids, scores))

        return recommendations

    def __str__(self):
        return f"PopularityRecommender(metric={self.popularity_metric})"


class ContentBasedRecommender:
    """
    Content-based recommender using cosine similarity.

    Matches students to courses based on feature similarity:
    - Qualification type match
    - Grade vs entry tariff
    - Subject area alignment
    - Geographic preferences
    """

    def __init__(
        self,
        qualification_weight: float = 0.3,
        grade_weight: float = 0.3,
        subject_weight: float = 0.2,
        outcome_weight: float = 0.2,
    ):
        """
        Initialize content-based recommender.

        Args:
            qualification_weight: Weight for qualification type match
            grade_weight: Weight for grade/entry tariff match
            subject_weight: Weight for subject area alignment
            outcome_weight: Weight for employment/satisfaction
        """
        self.weights = {
            "qualification": qualification_weight,
            "grade": grade_weight,
            "subject": subject_weight,
            "outcome": outcome_weight,
        }
        self.student_features: Optional[np.ndarray] = None
        self.course_features: Optional[np.ndarray] = None
        self.student_ids: Optional[List[str]] = None
        self.course_ids: Optional[List[str]] = None
        self.feature_names: List[str] = []

    def _encode_student_features(
        self,
        student_features_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Encode student features into numeric vector.

        Args:
            student_features_df: Student feature DataFrame

        Returns:
            Feature matrix (n_students, n_features)
        """
        features = []

        # Qualification type (one-hot)
        qual_types = [
            "qual_A-Level",
            "qual_BTEC",
            "qual_Access to HE",
            "qual_International Baccalaureate",
            "qual_Mature Student",
        ]
        for qual in qual_types:
            if qual in student_features_df.columns:
                features.append(student_features_df[qual].fillna(0).values)
            else:
                features.append(np.zeros(len(student_features_df)))

        # UCAS tariff (normalized 0-1)
        if "ucas_tariff_points" in student_features_df.columns:
            tariff_normalized = student_features_df["ucas_tariff_points"] / 168.0
            features.append(tariff_normalized.values)
        else:
            features.append(np.zeros(len(student_features_df)))

        # Contextual score (normalized)
        if "contextual_score" in student_features_df.columns:
            contextual_normalized = student_features_df["contextual_score"] / 6.0
            features.append(contextual_normalized.values)
        else:
            features.append(np.zeros(len(student_features_df)))

        # IMD inverse (normalized)
        if "imd_inverse" in student_features_df.columns:
            imd_normalized = student_features_df["imd_inverse"] / 10.0
            features.append(imd_normalized.values)

        feature_matrix = np.column_stack(features)
        self.feature_names = [
            "qual_A-Level",
            "qual_BTEC",
            "qual_Access",
            "qual_IB",
            "qual_Mature",
            "ucas_tariff",
            "contextual_score",
            "imd_inverse",
        ]

        return feature_matrix

    def _encode_course_features(
        self,
        course_features_df: pd.DataFrame,
    ) -> np.ndarray:
        """
        Encode course features into numeric vector.

        Args:
            course_features_df: Course feature DataFrame

        Returns:
            Feature matrix (n_courses, n_features)
        """
        features = []

        # Department (one-hot, top departments only)
        top_depts = ["Engineering", "Computer Science", "Business", "Health Sciences"]
        for dept in top_depts:
            col = f"dept_{dept}"
            if col in course_features_df.columns:
                features.append(course_features_df[col].fillna(0).values)
            else:
                features.append(np.zeros(len(course_features_df)))

        # Entry tariff (normalized, inverted — lower is easier to get in)
        if "entry_tariff" in course_features_df.columns:
            tariff_inverse = 1.0 - (course_features_df["entry_tariff"] / 168.0)
            features.append(tariff_inverse.clip(0, 1).values)

        # Employment rate
        if "employment_rate_15m" in course_features_df.columns:
            features.append(course_features_df["employment_rate_15m"].values)

        # Satisfaction score (normalized)
        if "satisfaction_score" in course_features_df.columns:
            satisfaction_normalized = course_features_df["satisfaction_score"] / 5.0
            features.append(satisfaction_normalized.values)

        # Sandwich year flag
        if "sandwich_year" in course_features_df.columns:
            features.append(course_features_df["sandwich_year"].values)

        # Coursework heavy flag
        if "coursework_heavy" in course_features_df.columns:
            features.append(course_features_df["coursework_heavy"].values)

        feature_matrix = np.column_stack(features)
        self.course_feature_names = [
            "dept_Engineering",
            "dept_CS",
            "dept_Business",
            "dept_Health",
            "entry_tariff_inverse",
            "employment_rate",
            "satisfaction",
            "sandwich_year",
            "coursework_heavy",
        ]

        return feature_matrix

    def fit(
        self,
        student_features_df: pd.DataFrame,
        course_features_df: pd.DataFrame,
        **kwargs,
    ):
        """
        Fit recommender (encode features).

        Args:
            student_features_df: Student feature DataFrame
            course_features_df: Course feature DataFrame
        """
        logger.info("Fitting ContentBasedRecommender...")

        self.student_ids = student_features_df["student_id"].tolist()
        self.course_ids = course_features_df["course_id"].tolist()

        # Encode features
        self.student_features = self._encode_student_features(student_features_df)
        self.course_features = self._encode_course_features(course_features_df)

        logger.info(
            f"Encoded {len(self.student_ids)} students and {len(self.course_ids)} courses"
        )
        logger.info(f"Student feature dim: {self.student_features.shape}")
        logger.info(f"Course feature dim: {self.course_features.shape}")

        return self

    def predict(
        self,
        user_features: Optional[pd.DataFrame] = None,
        n_recommendations: int = 10,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Generate content-based recommendations.

        Computes cosine similarity between student and course features.

        Args:
            user_features: Optional new user features (if None, uses fitted data)
            n_recommendations: Number of recommendations per user

        Returns:
            Dictionary mapping user_id to list of (course_id, similarity_score) tuples
        """
        if self.student_features is None or self.course_features is None:
            raise ValueError("Must call fit() before predict()")

        # Use fitted student features or encode new ones
        if user_features is not None:
            student_feats = self._encode_student_features(user_features)
            student_ids = user_features["student_id"].tolist()
        else:
            student_feats = self.student_features
            student_ids = self.student_ids

        recommendations = {}

        # Compute similarity matrix (students x courses)
        similarity_matrix = cosine_similarity(student_feats, self.course_features)

        # Clip to [0, 1] range
        similarity_matrix = np.clip(similarity_matrix, 0, 1)

        # Get top N for each student
        for i, user_id in enumerate(student_ids):
            user_similarities = similarity_matrix[i]

            # Get top N indices
            top_indices = np.argsort(user_similarities)[::-1][:n_recommendations]
            top_courses = [self.course_ids[j] for j in top_indices]
            top_scores = [user_similarities[j] for j in top_indices]

            recommendations[user_id] = list(zip(top_courses, top_scores))

        return recommendations

    def explain_recommendation(
        self,
        student_id: str,
        course_id: str,
    ) -> Dict[str, Any]:
        """
        Explain why a course was recommended to a student.

        Args:
            student_id: Student ID
            course_id: Course ID

        Returns:
            Explanation dictionary with feature contributions
        """
        if self.student_ids is None or self.course_ids is None:
            raise ValueError("Must call fit() before explain()")

        student_idx = self.student_ids.index(student_id)
        course_idx = self.course_ids.index(course_id)

        student_feat = self.student_features[student_idx]
        course_feat = self.course_features[course_idx]

        # Feature-wise similarity
        feature_similarities = 1.0 - np.abs(student_feat - course_feat)

        explanations = []
        for i, name in enumerate(self.feature_names[: len(feature_similarities)]):
            explanations.append(
                {
                    "feature": name,
                    "student_value": float(student_feat[i]),
                    "course_value": float(course_feat[i]),
                    "similarity": float(feature_similarities[i]),
                }
            )

        # Sort by contribution
        explanations.sort(key=lambda x: x["similarity"], reverse=True)

        return {
            "student_id": student_id,
            "course_id": course_id,
            "overall_similarity": float(
                cosine_similarity(
                    student_feat.reshape(1, -1), course_feat.reshape(1, -1)
                )[0, 0]
            ),
            "feature_contributions": explanations[:5],  # Top 5 features
        }

    def __str__(self):
        return "ContentBasedRecommender"


def evaluate_baselines(
    student_features: pd.DataFrame,
    course_features: pd.DataFrame,
    enrollments_df: pd.DataFrame,
    n_recommendations: int = 10,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all baseline models.

    Args:
        student_features: Student feature DataFrame
        course_features: Course feature DataFrame
        enrollments_df: Enrollment records (for ground truth)
        n_recommendations: Number of recommendations

    Returns:
        Dictionary of model_name -> metric -> score
    """
    from src.evaluation.ranking_metrics import ndcg_at_k, precision_at_k, recall_at_k

    logger.info("Evaluating baseline models...")

    # Create ground truth (enrolled courses)
    ground_truth = enrollments_df.groupby("student_id")["course_id"].apply(list).to_dict()

    results = {}

    # === Random Baseline ===
    logger.info("Evaluating RandomRecommender...")
    random_rec = RandomRecommender(seed=42)
    random_rec.fit(course_features)
    random_recs = random_rec.predict(student_features, n_recommendations)

    random_ndcg = []
    random_precision = []
    for user_id, true_courses in ground_truth.items():
        if user_id not in random_recs:
            continue

        rec_courses = [c[0] for c in random_recs[user_id]]
        y_true = np.array([1 if c in true_courses else 0 for c in rec_courses])
        y_pred = np.array([random_recs[user_id][i][1] for i in range(len(rec_courses))])

        if y_true.sum() > 0:
            random_ndcg.append(ndcg_at_k(y_true, y_pred, k=n_recommendations))
            random_precision.append(precision_at_k(y_true, y_pred, k=n_recommendations))

    results["RandomRecommender"] = {
        "ndcg_at_10": np.mean(random_ndcg) if random_ndcg else 0.0,
        "precision_at_10": np.mean(random_precision) if random_precision else 0.0,
    }

    # === Popularity Baseline ===
    logger.info("Evaluating PopularityRecommender...")
    pop_rec = PopularityRecommender(popularity_metric="enrollment_count")
    pop_rec.fit(course_features, enrollments_df)
    pop_recs = pop_rec.predict(student_features, n_recommendations)

    pop_ndcg = []
    pop_precision = []
    for user_id, true_courses in ground_truth.items():
        if user_id not in pop_recs:
            continue

        rec_courses = [c[0] for c in pop_recs[user_id]]
        y_true = np.array([1 if c in true_courses else 0 for c in rec_courses])
        y_pred = np.array([pop_recs[user_id][i][1] for i in range(len(rec_courses))])

        if y_true.sum() > 0:
            pop_ndcg.append(ndcg_at_k(y_true, y_pred, k=n_recommendations))
            pop_precision.append(precision_at_k(y_true, y_pred, k=n_recommendations))

    results["PopularityRecommender"] = {
        "ndcg_at_10": np.mean(pop_ndcg) if pop_ndcg else 0.0,
        "precision_at_10": np.mean(pop_precision) if pop_precision else 0.0,
    }

    # === Content-Based Baseline ===
    logger.info("Evaluating ContentBasedRecommender...")
    cb_rec = ContentBasedRecommender()
    cb_rec.fit(student_features, course_features)
    cb_recs = cb_rec.predict(student_features, n_recommendations)

    cb_ndcg = []
    cb_precision = []
    for user_id, true_courses in ground_truth.items():
        if user_id not in cb_recs:
            continue

        rec_courses = [c[0] for c in cb_recs[user_id]]
        y_true = np.array([1 if c in true_courses else 0 for c in rec_courses])
        y_pred = np.array([cb_recs[user_id][i][1] for i in range(len(rec_courses))])

        if y_true.sum() > 0:
            cb_ndcg.append(ndcg_at_k(y_true, y_pred, k=n_recommendations))
            cb_precision.append(precision_at_k(y_true, y_pred, k=n_recommendations))

    results["ContentBasedRecommender"] = {
        "ndcg_at_10": np.mean(cb_ndcg) if cb_ndcg else 0.0,
        "precision_at_10": np.mean(cb_precision) if cb_precision else 0.0,
    }

    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE MODEL EVALUATION RESULTS")
    logger.info("=" * 60)
    for model_name, metrics in results.items():
        logger.info(f"\n{model_name}:")
        for metric, score in metrics.items():
            logger.info(f"  {metric}: {score:.4f}")
    logger.info("=" * 60)

    return results


# Example usage
if __name__ == "__main__":
    from src.data.synthetic import SITSSyntheticGenerator
    from src.data.feature_store import FeatureStore

    # Generate synthetic data
    print("Generating synthetic data...")
    generator = SITSSyntheticGenerator(n_students=500, n_courses=50, seed=42)
    datasets = generator.generate_all_datasets()

    # Engineer features
    print("\nEngineering features...")
    feature_store = FeatureStore()

    student_features = feature_store.engineer_student_features(
        datasets["students"], datasets["qualifications"]
    )
    course_features = feature_store.engineer_course_features(datasets["courses"])

    print(f"Student features: {student_features.shape}")
    print(f"Course features: {course_features.shape}")

    # Evaluate baselines
    print("\n" + "=" * 60)
    print("EVALUATING BASELINE MODELS")
    print("=" * 60)

    results = evaluate_baselines(
        student_features,
        course_features,
        datasets["enrollments"],
        n_recommendations=10,
    )

    print("\n✓ Baseline evaluation complete!")
    print("\nNext steps:")
    print("  1. Implement collaborative filtering (LightFM)")
    print("  2. Implement advanced content-based (TF-IDF + embeddings)")
    print("  3. Build hybrid ensemble")
    print("  4. Tune hyperparameters with Optuna")
