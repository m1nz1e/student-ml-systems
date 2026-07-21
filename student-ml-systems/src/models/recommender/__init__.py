"""Course Recommender models module."""

from .baselines import (
    RandomRecommender,
    PopularityRecommender,
    ContentBasedRecommender,
    evaluate_baselines,
)

from .collaborative import (
    MatrixFactorization,
    LightFMRecommender,
    NeuralCollaborativeFiltering,
)

from .content_based import (
    TFIDFRecommender,
    SemanticEmbeddingRecommender,
    MultiModalRecommender,
    PGVectorRecommender,
)

from .ensemble import (
    EnsembleRecommender,
    LearningToRankEnsemble,
    MaximalMarginalRelevance,
    ColdStartHandler,
    HybridCourseRecommender,
)

__all__ = [
    # Baselines
    "RandomRecommender",
    "PopularityRecommender",
    "ContentBasedRecommender",
    "evaluate_baselines",
    # Collaborative Filtering
    "MatrixFactorization",
    "LightFMRecommender",
    "NeuralCollaborativeFiltering",
    # Advanced Content-Based
    "TFIDFRecommender",
    "SemanticEmbeddingRecommender",
    "MultiModalRecommender",
    "PGVectorRecommender",
    # Ensemble & Hybrid
    "EnsembleRecommender",
    "LearningToRankEnsemble",
    "MaximalMarginalRelevance",
    "ColdStartHandler",
    "HybridCourseRecommender",
]
