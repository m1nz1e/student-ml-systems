"""
Hybrid Ensemble for Course Recommender System.

Combines all recommendation strategies into a unified, production-ready system:
1. Weighted Ensemble — Multi-model fusion with configurable weights
2. Learning-to-Rank (LambdaMART) — XGBoost meta-learner
3. Diversity Injection (MMR) — Maximal Marginal Relevance
4. Cold-Start Handler — Intelligent model routing
5. Auto Model Selection — Context-aware model choice

This is the final layer that brings together:
- Baselines (Random, Popularity, Content-Based)
- Collaborative Filtering (MF, LightFM, NCF)
- Content-Based (TF-IDF, Semantic, Multi-Modal, PGVector)
"""

from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleRecommender:
    """
    Weighted ensemble combining multiple recommenders.

    Aggregates predictions from multiple models using weighted averaging.
    Supports dynamic weight adjustment and model-specific confidence scores.
    """

    def __init__(
        self,
        recommenders: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None,
        normalize_weights: bool = True,
    ):
        """
        Initialize ensemble recommender.

        Args:
            recommenders: Dictionary of {name: recommender_instance}
            weights: Dictionary of {name: weight} (default: equal weights)
            normalize_weights: Whether to normalize weights to sum to 1
        """
        self.recommenders = recommenders
        self.normalize_weights = normalize_weights

        # Default equal weights
        if weights is None:
            self.weights = {name: 1.0 / len(recommenders) for name in recommenders}
        else:
            self.weights = weights.copy()

        # Normalize if requested
        if normalize_weights:
            total = sum(self.weights.values())
            self.weights = {k: v / total for k, v in self.weights.items()}

        logger.info(f"Initialized Ensemble with {len(recommenders)} models")
        logger.info(f"Weights: {self.weights}")

    def fit(
        self,
        courses_df: pd.DataFrame,
        student_features_df: pd.DataFrame,
        enrollments_df: Optional[pd.DataFrame] = None,
        **kwargs,
    ):
        """
        Fit all component recommenders.

        Args:
            courses_df: Course DataFrame
            student_features_df: Student feature DataFrame
            enrollments_df: Optional enrollment records
        """
        logger.info("Fitting ensemble component models...")

        for name, recommender in self.recommenders.items():
            logger.info(f"  Fitting {name}...")
            try:
                if hasattr(recommender, "fit"):
                    # Try to fit with available data
                    if enrollments_df is not None:
                        recommender.fit(
                            courses_df,
                            student_features_df=student_features_df,
                            enrollments_df=enrollments_df,
                            **kwargs,
                        )
                    else:
                        recommender.fit(courses_df, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to fit {name}: {e}")

        logger.info("Ensemble fitting complete")
        return self

    def predict(
        self,
        student_features: pd.DataFrame,
        n_recommendations: int = 10,
        return_all_scores: bool = False,
    ) -> Union[
        Dict[str, List[Tuple[str, float]]],
        Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, Dict[str, float]]],
    ]:
        """
        Generate ensemble recommendations with weighted fusion.

        Args:
            student_features: Student feature DataFrame
            n_recommendations: Number of recommendations
            return_all_scores: Whether to return individual model scores

        Returns:
            Dictionary mapping student_id to (course_id, fused_score) tuples
            Optionally also returns per-model scores
        """
        logger.info("Generating ensemble recommendations...")

        # Collect predictions from all models
        all_predictions = {}
        for name, recommender in self.recommenders.items():
            try:
                predictions = recommender.predict(student_features, n_recommendations)
                all_predictions[name] = predictions
            except Exception as e:
                logger.warning(f"{name} failed: {e}")
                all_predictions[name] = {}

        # Fuse predictions
        fused_recommendations = {}
        all_scores = defaultdict(lambda: defaultdict(dict))

        student_ids = student_features["student_id"].tolist()

        for student_id in student_ids:
            course_scores = defaultdict(list)

            # Collect scores from each model
            for model_name, predictions in all_predictions.items():
                if student_id not in predictions:
                    continue

                for course_id, score in predictions[student_id]:
                    # Normalize score to [0, 1] range (sigmoid-like)
                    normalized_score = 1 / (1 + np.exp(-score))
                    weighted_score = self.weights.get(model_name, 0) * normalized_score
                    course_scores[course_id].append((model_name, weighted_score))
                    all_scores[student_id][course_id][model_name] = float(score)

            # Aggregate scores (weighted sum)
            aggregated_scores = {}
            for course_id, model_scores in course_scores.items():
                total_score = sum(score for _, score in model_scores)
                n_models = len(model_scores)
                # Average across models that provided a score
                aggregated_scores[course_id] = total_score / max(n_models, 1)

            # Get top N
            sorted_courses = sorted(
                aggregated_scores.items(), key=lambda x: x[1], reverse=True
            )[:n_recommendations]

            fused_recommendations[student_id] = sorted_courses

        if return_all_scores:
            return fused_recommendations, dict(all_scores)
        else:
            return fused_recommendations

    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update model weights dynamically.

        Args:
            new_weights: New weight dictionary
        """
        if self.normalize_weights:
            total = sum(new_weights.values())
            self.weights = {k: v / total for k, v in new_weights.items()}
        else:
            self.weights = new_weights.copy()

        logger.info(f"Updated ensemble weights: {self.weights}")

    def get_model_performance(
        self,
        student_features: pd.DataFrame,
        ground_truth: Dict[str, List[str]],
        n_recommendations: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate individual model performance.

        Args:
            student_features: Student feature DataFrame
            ground_truth: Dictionary of {student_id: [enrolled_courses]}
            n_recommendations: Number of recommendations

        Returns:
            Dictionary of {model_name: ndcg_score}
        """
        from src.evaluation.ranking_metrics import ndcg_at_k

        model_scores = {}

        for name, recommender in self.recommenders.items():
            try:
                predictions = recommender.predict(student_features, n_recommendations)

                ndcg_scores = []
                for student_id, true_courses in ground_truth.items():
                    if student_id not in predictions:
                        continue

                    recs = predictions[student_id]
                    rec_course_ids = [c[0] for c in recs]
                    rec_scores = np.array([c[1] for c in recs])

                    # Create relevance vector
                    y_true = np.array([1 if c in true_courses else 0 for c in rec_course_ids])

                    if y_true.sum() > 0:
                        ndcg = ndcg_at_k(y_true, rec_scores, k=n_recommendations)
                        ndcg_scores.append(ndcg)

                model_scores[name] = np.mean(ndcg_scores) if ndcg_scores else 0.0

            except Exception as e:
                logger.warning(f"Failed to evaluate {name}: {e}")
                model_scores[name] = 0.0

        return model_scores


class LearningToRankEnsemble:
    """
    Learning-to-Rank ensemble using XGBoost LambdaMART.

    Learns optimal combination of model predictions using gradient boosting
    with ranking loss (NDCG optimization).

    Features:
    - Meta-features from multiple models
    - NDCG optimization (directly optimizes ranking metric)
    - Handles missing predictions gracefully
    """

    def __init__(
        self,
        base_recommenders: Dict[str, Any],
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        ndcg_cutoff: int = 10,
        random_state: int = 42,
    ):
        """
        Initialize LTR ensemble.

        Args:
            base_recommenders: Dictionary of base recommender models
            n_estimators: Number of boosting rounds
            max_depth: Max tree depth
            learning_rate: Learning rate
            ndcg_cutoff: NDCG cutoff (e.g., NDCG@10)
            random_state: Random seed
        """
        self.base_recommenders = base_recommenders
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.ndcg_cutoff = ndcg_cutoff
        self.random_state = random_state

        self.model = None
        self.feature_names = None
        self.course_id_map = None
        self.reverse_course_map = None

    def _prepare_training_data(
        self,
        student_features: pd.DataFrame,
        courses_df: pd.DataFrame,
        enrollments_df: pd.DataFrame,
        n_recommendations: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data for learning-to-rank.

        Creates (student, course) pairs with:
        - Features: Predictions from each base model
        - Labels: Relevance (1 if enrolled, 0 otherwise)
        - Query IDs: Student IDs (for grouping)

        Args:
            student_features: Student feature DataFrame
            courses_df: Course DataFrame
            enrollments_df: Enrollment records
            n_recommendations: Number of candidate courses per student

        Returns:
            Tuple of (feature_matrix, labels, query_ids)
        """
        logger.info("Preparing LTR training data...")

        # Get predictions from all base models
        all_predictions = {}
        for name, recommender in self.base_recommenders.items():
            try:
                predictions = recommender.predict(student_features, n_recommendations)
                all_predictions[name] = predictions
            except Exception as e:
                logger.warning(f"{name} failed: {e}")
                all_predictions[name] = {}

        # Build feature matrix
        features_list = []
        labels_list = []
        query_ids = []

        # Ground truth from enrollments
        ground_truth = enrollments_df.groupby("student_id")["course_id"].apply(set).to_dict()
        course_ids = courses_df["course_id"].unique().tolist()

        self.course_id_map = {cid: idx for idx, cid in enumerate(course_ids)}
        self.reverse_course_map = {idx: cid for cid, idx in self.course_id_map.items()}

        for student_idx, (_, student_row) in enumerate(student_features.iterrows()):
            student_id = student_row["student_id"]

            if student_id not in ground_truth:
                continue

            enrolled_courses = ground_truth[student_id]

            # Create positive samples (enrolled courses)
            for course_id in enrolled_courses:
                if course_id not in self.course_id_map:
                    continue

                # Get predictions from all models
                feature_vector = []
                for model_name, predictions in all_predictions.items():
                    if student_id in predictions:
                        # Find score for this course
                        course_score = 0.0
                        for cid, score in predictions[student_id]:
                            if cid == course_id:
                                course_score = score
                                break
                        feature_vector.append(course_score)
                    else:
                        feature_vector.append(0.0)

                features_list.append(feature_vector)
                labels_list.append(1.0)  # Positive sample
                query_ids.append(student_idx)

            # Create negative samples (non-enrolled courses)
            # Sample from model recommendations that weren't enrolled
            negative_samples = set()
            for model_name, predictions in all_predictions.items():
                if student_id in predictions:
                    for course_id, score in predictions[student_id]:
                        if course_id not in enrolled_courses:
                            negative_samples.add(course_id)

            # Limit negative samples (balance classes)
            negative_samples = list(negative_samples)[:len(enrolled_courses) * 2]

            for course_id in negative_samples:
                if course_id not in self.course_id_map:
                    continue

                feature_vector = []
                for model_name, predictions in all_predictions.items():
                    if student_id in predictions:
                        course_score = 0.0
                        for cid, score in predictions[student_id]:
                            if cid == course_id:
                                course_score = score
                                break
                        feature_vector.append(course_score)
                    else:
                        feature_vector.append(0.0)

                features_list.append(feature_vector)
                labels_list.append(0.0)  # Negative sample
                query_ids.append(student_idx)

        X = np.array(features_list)
        y = np.array(labels_list)
        q = np.array(query_ids)

        # Create query groups (count samples per query)
        unique_queries, query_counts = np.unique(q, return_counts=True)

        logger.info(f"Created {len(X)} samples with {len(unique_queries)} queries")
        logger.info(f"Feature dimension: {X.shape[1]}")

        self.feature_names = list(self.base_recommenders.keys())

        return X, y, query_counts

    def fit(
        self,
        student_features: pd.DataFrame,
        courses_df: pd.DataFrame,
        enrollments_df: pd.DataFrame,
        **kwargs,
    ):
        """
        Fit LTR model on training data.

        Args:
            student_features: Student feature DataFrame
            courses_df: Course DataFrame
            enrollments_df: Enrollment records
        """
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("XGBoost not installed. Install with: pip install xgboost")
            raise

        logger.info("Fitting Learning-to-Rank ensemble...")

        # Prepare training data
        X, y, query_counts = self._prepare_training_data(
            student_features, courses_df, enrollments_df
        )

        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X, label=y)
        dtrain.set_info(qid=query_counts)

        # LTR parameters (LambdaMART)
        params = {
            "objective": "rank:ndcg",
            "eval_metric": "ndcg@{}".format(self.ndcg_cutoff),
            "eta": self.learning_rate,
            "max_depth": self.max_depth,
            "seed": self.random_state,
        }

        # Train model
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=[(dtrain, "train")],
            verbose_eval=10,
        )

        logger.info("LTR ensemble training complete")
        return self

    def predict(
        self,
        student_features: pd.DataFrame,
        courses_df: pd.DataFrame,
        n_recommendations: int = 10,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Generate LTR-based recommendations.

        Args:
            student_features: Student feature DataFrame
            courses_df: Course DataFrame
            n_recommendations: Number of recommendations

        Returns:
            Dictionary mapping student_id to (course_id, score) tuples
        """
        try:
            import xgboost as xgb
        except ImportError:
            logger.error("XGBoost not installed")
            raise

        if self.model is None:
            raise ValueError("Must call fit() before predict()")

        logger.info("Generating LTR recommendations...")

        # Get predictions from base models
        all_predictions = {}
        for name, recommender in self.base_recommenders.items():
            try:
                predictions = recommender.predict(student_features, n_recommendations * 2)
                all_predictions[name] = predictions
            except Exception as e:
                logger.warning(f"{name} failed: {e}")
                all_predictions[name] = {}

        recommendations = {}
        course_ids = courses_df["course_id"].unique().tolist()

        for _, student_row in student_features.iterrows():
            student_id = student_row["student_id"]

            # Collect candidate courses from all models
            candidate_courses = set()
            for model_name, predictions in all_predictions.items():
                if student_id in predictions:
                    for course_id, _ in predictions[student_id]:
                        candidate_courses.add(course_id)

            if not candidate_courses:
                # Fallback to all courses
                candidate_courses = set(course_ids)

            # Build feature matrix for candidates
            features_list = []
            candidate_list = []

            for course_id in candidate_courses:
                feature_vector = []
                for model_name, predictions in all_predictions.items():
                    if student_id in predictions:
                        course_score = 0.0
                        for cid, score in predictions[student_id]:
                            if cid == course_id:
                                course_score = score
                                break
                        feature_vector.append(course_score)
                    else:
                        feature_vector.append(0.0)

                features_list.append(feature_vector)
                candidate_list.append(course_id)

            X = np.array(features_list)
            dtest = xgb.DMatrix(X)

            # Predict scores
            scores = self.model.predict(dtest)

            # Get top N
            top_indices = np.argsort(scores)[::-1][:n_recommendations]
            top_courses = [candidate_list[i] for i in top_indices]
            top_scores = [float(scores[i]) for i in top_indices]

            recommendations[student_id] = list(zip(top_courses, top_scores))

        return recommendations


class MaximalMarginalRelevance:
    """
    Maximal Marginal Relevance (MMR) for diversity injection.

    Re-ranks recommendations to balance relevance and diversity.
    Prevents recommending too many similar courses.

    MMR = argmax [ λ * Relevance - (1-λ) * MaxSimilarity ]
    """

    def __init__(self, lambda_param: float = 0.7):
        """
        Initialize MMR re-ranker.

        Args:
            lambda_param: Balance between relevance and diversity
                         (0 = all diversity, 1 = all relevance)
        """
        self.lambda_param = lambda_param

    def _compute_similarity(
        self,
        course_a: str,
        course_b: str,
        course_features: pd.DataFrame,
    ) -> float:
        """
        Compute similarity between two courses.

        Uses cosine similarity on course features.

        Args:
            course_a: Course ID A
            course_b: Course ID B
            course_features: Course feature DataFrame

        Returns:
            Similarity score [0, 1]
        """
        # Get course indices
        if "course_id" not in course_features.columns:
            return 0.0

        try:
            idx_a = course_features[course_features["course_id"] == course_a].index[0]
            idx_b = course_features[course_features["course_id"] == course_b].index[0]
        except (IndexError, KeyError):
            return 0.0

        # Get numeric features
        numeric_cols = course_features.select_dtypes(include=[np.number]).columns
        vec_a = course_features.loc[idx_a, numeric_cols].values
        vec_b = course_features.loc[idx_b, numeric_cols].values

        # Cosine similarity
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)
        return float((similarity + 1) / 2)  # Normalize to [0, 1]

    def re_rank(
        self,
        recommendations: List[Tuple[str, float]],
        course_features: pd.DataFrame,
        n_recommendations: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Re-rank recommendations using MMR.

        Args:
            recommendations: Initial ranked list of (course_id, score)
            course_features: Course feature DataFrame
            n_recommendations: Number of recommendations to return

        Returns:
            Re-ranked list of (course_id, mmr_score) tuples
        """
        if len(recommendations) <= 1:
            return recommendations

        # Initialize
        selected = []
        remaining = recommendations.copy()

        while len(selected) < min(n_recommendations, len(recommendations)) and remaining:
            best_mmr_score = -float("inf")
            best_course = None
            best_idx = None

            for i, (course_id, relevance_score) in enumerate(remaining):
                # Compute max similarity to already selected courses
                max_similarity = 0.0
                for selected_course, _ in selected:
                    sim = self._compute_similarity(
                        course_id, selected_course, course_features
                    )
                    max_similarity = max(max_similarity, sim)

                # MMR score
                mmr_score = (
                    self.lambda_param * relevance_score
                    - (1 - self.lambda_param) * max_similarity
                )

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_course = course_id
                    best_idx = i

            if best_course is not None:
                selected.append((best_course, float(best_mmr_score)))
                remaining.pop(best_idx)

        return selected


class ColdStartHandler:
    """
    Intelligent handler for cold-start scenarios.

    Routes users to appropriate models based on data availability:
    - New user, no interactions → Content-based or popularity
    - New course, no enrollments → Content-based or semantic
    - Some data available → Hybrid or collaborative

    Also provides fallback chains when primary models fail.
    """

    def __init__(
        self,
        recommenders: Dict[str, Any],
        fallback_chain: Optional[List[str]] = None,
    ):
        """
        Initialize cold-start handler.

        Args:
            recommenders: Dictionary of available recommenders
            fallback_chain: Ordered list of model names for fallback
        """
        self.recommenders = recommenders
        self.fallback_chain = fallback_chain or [
            "content_based",
            "popularity",
            "random",
        ]

    def select_model(
        self,
        student_features: pd.DataFrame,
        student_id: str,
        has_interactions: bool = False,
    ) -> str:
        """
        Select best model for a given user.

        Decision logic:
        - If user has interactions → Use collaborative filtering
        - If user has qualifications → Use content-based
        - Otherwise → Use popularity or random

        Args:
            student_features: Student feature DataFrame
            student_id: User ID
            has_interactions: Whether user has historical interactions

        Returns:
            Selected model name
        """
        # Check if user exists in data
        if student_id not in student_features["student_id"].values:
            # Complete cold start → popularity
            return "popularity"

        student_row = student_features[student_features["student_id"] == student_id].iloc[0]

        # Check data availability
        has_qualifications = (
            "ucas_tariff_points" in student_row
            and pd.notna(student_row["ucas_tariff_points"])
        )

        # Decision tree
        if has_interactions:
            # User has enrollment history → collaborative filtering
            if "lightfm" in self.recommenders:
                return "lightfm"
            elif "matrix_factorization" in self.recommenders:
                return "matrix_factorization"

        if has_qualifications:
            # User has qualifications → content-based
            if "semantic" in self.recommenders:
                return "semantic"
            elif "tfidf" in self.recommenders:
                return "tfidf"
            elif "content_based" in self.recommenders:
                return "content_based"

        # Fallback chain
        for model_name in self.fallback_chain:
            if model_name in self.recommenders:
                return model_name

        # Last resort: first available model
        return list(self.recommenders.keys())[0]

    def recommend(
        self,
        student_features: pd.DataFrame,
        student_id: str,
        n_recommendations: int = 10,
        **kwargs,
    ) -> List[Tuple[str, float]]:
        """
        Generate recommendations with automatic model selection and fallback.

        Args:
            student_features: Student feature DataFrame
            student_id: User ID
            n_recommendations: Number of recommendations

        Returns:
            List of (course_id, score) tuples
        """
        # Select best model
        selected_model = self.select_model(
            student_features, student_id, has_interactions=False
        )

        logger.info(f"Selected model for {student_id}: {selected_model}")

        # Try selected model
        try:
            recommender = self.recommenders[selected_model]
            predictions = recommender.predict(
                student_features[student_features["student_id"] == student_id],
                n_recommendations,
            )

            if student_id in predictions and predictions[student_id]:
                return predictions[student_id]

        except Exception as e:
            logger.warning(f"{selected_model} failed: {e}")

        # Fallback chain
        for model_name in self.fallback_chain:
            if model_name not in self.recommenders:
                continue

            try:
                logger.info(f"Falling back to {model_name}")
                recommender = self.recommenders[model_name]
                predictions = recommender.predict(
                    student_features[student_features["student_id"] == student_id],
                    n_recommendations,
                )

                if student_id in predictions and predictions[student_id]:
                    return predictions[student_id]

            except Exception as e:
                logger.warning(f"Fallback {model_name} failed: {e}")

        # All models failed
        logger.error(f"All models failed for user {student_id}")
        return []


class HybridCourseRecommender:
    """
    Production-ready hybrid course recommender system.

    Combines all components:
    - Ensemble of multiple models
    - Learning-to-rank meta-learner
    - MMR diversity re-ranking
    - Cold-start handling
    - Automatic model selection

    This is the main interface for the course recommender system.
    """

    def __init__(
        self,
        mode: str = "ensemble",
        diversity_lambda: float = 0.7,
        ensemble_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize hybrid recommender.

        Args:
            mode: One of ['ensemble', 'ltr', 'mmr', 'hybrid']
            diversity_lambda: MMR diversity parameter
            ensemble_weights: Custom weights for ensemble
        """
        self.mode = mode
        self.diversity_lambda = diversity_lambda
        self.ensemble_weights = ensemble_weights

        self.recommenders = {}
        self.ensemble = None
        self.ltr = None
        self.mmr = None
        self.cold_start_handler = None

        self.course_features = None

    def add_recommender(self, name: str, recommender: Any):
        """
        Add a recommender to the system.

        Args:
            name: Recommender name
            recommender: Recommender instance
        """
        self.recommenders[name] = recommender
        logger.info(f"Added recommender: {name}")

    def fit(
        self,
        courses_df: pd.DataFrame,
        student_features_df: pd.DataFrame,
        enrollments_df: pd.DataFrame,
        **kwargs,
    ):
        """
        Fit all components.

        Args:
            courses_df: Course DataFrame
            student_features_df: Student feature DataFrame
            enrollments_df: Enrollment records
        """
        logger.info(f"Fitting HybridCourseRecommender (mode={self.mode})...")

        self.course_features = courses_df

        # Initialize MMR
        self.mmr = MaximalMarginalRelevance(lambda_param=self.diversity_lambda)

        # Initialize cold-start handler
        self.cold_start_handler = ColdStartHandler(
            recommenders=self.recommenders,
            fallback_chain=["content_based", "popularity", "random"],
        )

        # Initialize ensemble
        if self.recommenders:
            self.ensemble = EnsembleRecommender(
                recommenders=self.recommenders,
                weights=self.ensemble_weights,
            )
            self.ensemble.fit(courses_df, student_features_df, enrollments_df)

        # Initialize LTR (if in LTR mode)
        if self.mode == "ltr" and self.recommenders:
            self.ltr = LearningToRankEnsemble(
                base_recommenders=self.recommenders,
            )
            self.ltr.fit(student_features_df, courses_df, enrollments_df)

        logger.info("HybridCourseRecommender fitting complete")
        return self

    def predict(
        self,
        student_features: pd.DataFrame,
        n_recommendations: int = 10,
        apply_diversity: bool = True,
        handle_cold_start: bool = True,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Generate hybrid recommendations.

        Args:
            student_features: Student feature DataFrame
            n_recommendations: Number of recommendations
            apply_diversity: Whether to apply MMR re-ranking
            handle_cold_start: Whether to use cold-start handler

        Returns:
            Dictionary mapping student_id to (course_id, score) tuples
        """
        logger.info(f"Generating {self.mode} recommendations...")

        # Cold-start handling
        if handle_cold_start and self.cold_start_handler:
            recommendations = {}
            for _, row in student_features.iterrows():
                student_id = row["student_id"]
                recs = self.cold_start_handler.recommend(
                    student_features, student_id, n_recommendations
                )
                recommendations[student_id] = recs
            return recommendations

        # LTR mode
        if self.mode == "ltr" and self.ltr is not None:
            recommendations = self.ltr.predict(
                student_features, self.course_features, n_recommendations
            )
        # Ensemble mode
        elif self.ensemble is not None:
            recommendations = self.ensemble.predict(
                student_features, n_recommendations
            )
        else:
            # Single model
            if self.recommenders:
                model_name = list(self.recommenders.keys())[0]
                recommendations = self.recommenders[model_name].predict(
                    student_features, n_recommendations
                )
            else:
                raise ValueError("No recommenders available")

        # Apply diversity re-ranking
        if apply_diversity and self.mmr:
            diverse_recommendations = {}
            for student_id, recs in recommendations.items():
                diverse_recs = self.mmr.re_rank(
                    recs, self.course_features, n_recommendations
                )
                diverse_recommendations[student_id] = diverse_recs
            return diverse_recommendations

        return recommendations

    def explain(
        self,
        student_id: str,
        course_id: str,
    ) -> Dict[str, Any]:
        """
        Get explanation for a recommendation.

        Args:
            student_id: Student ID
            course_id: Course ID

        Returns:
            Explanation dictionary
        """
        explanation = {
            "student_id": student_id,
            "course_id": course_id,
            "model_explanations": {},
        }

        # Get explanations from all models
        for name, recommender in self.recommenders.items():
            if hasattr(recommender, "explain_recommendation"):
                try:
                    exp = recommender.explain_recommendation(student_id, course_id)
                    explanation["model_explanations"][name] = exp
                except Exception as e:
                    explanation["model_explanations"][name] = {"error": str(e)}

        return explanation


# Example usage
if __name__ == "__main__":
    from src.data.synthetic import SITSSyntheticGenerator
    from src.data.feature_store import FeatureStore
    from src.models.recommender import (
        RandomRecommender,
        PopularityRecommender,
        ContentBasedRecommender,
        TFIDFRecommender,
    )

    # Generate synthetic data
    print("Generating synthetic data...")
    generator = SITSSyntheticGenerator(n_students=300, n_courses=40, seed=42)
    datasets = generator.generate_all_datasets()

    # Engineer features
    print("\nEngineering features...")
    feature_store = FeatureStore()
    student_features = feature_store.engineer_student_features(
        datasets["students"], datasets["qualifications"]
    )
    course_features = feature_store.engineer_course_features(datasets["courses"])

    # Add synthetic text
    course_features["course_description"] = course_features["course_name"]

    # Initialize hybrid recommender
    print("\n" + "=" * 60)
    print("HYBRID COURSE RECOMMENDER")
    print("=" * 60)

    hybrid = HybridCourseRecommender(mode="ensemble", diversity_lambda=0.7)

    # Add recommenders
    hybrid.add_recommender("random", RandomRecommender(seed=42))
    hybrid.add_recommender("popularity", PopularityRecommender())
    hybrid.add_recommender("content_based", ContentBasedRecommender())
    hybrid.add_recommender("tfidf", TFIDFRecommender(max_features=500))

    # Fit
    hybrid.fit(course_features, student_features, datasets["enrollments"])

    # Predict
    test_student = student_features.iloc[0:1]
    recommendations = hybrid.predict(test_student, n_recommendations=5)

    test_id = test_student["student_id"].iloc[0]
    print(f"\nRecommendations for {test_id}:")
    for course_id, score in recommendations[test_id]:
        print(f"  {course_id}: {score:.4f}")

    # Explain
    if recommendations[test_id]:
        top_course = recommendations[test_id][0][0]
        explanation = hybrid.explain(test_id, top_course)
        print(f"\nExplanation for {top_course}:")
        print(f"  Models used: {list(explanation['model_explanations'].keys())}")

    print("\n✓ Hybrid recommender system ready!")
