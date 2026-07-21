"""
Advanced Content-Based Filtering for Course Recommender System.

Implements:
1. TF-IDF + Cosine Similarity — Text-based course matching
2. Semantic Embeddings — Sentence-BERT for semantic understanding
3. pgvector Integration — PostgreSQL vector similarity search
4. Multi-modal Fusion — Text + structured features

Unlike the simple content-based baseline, these models:
- Process unstructured text (descriptions, modules, prerequisites)
- Understand semantic meaning (not just keyword overlap)
- Scale efficiently with vector databases
- Combine multiple signal types
"""

from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextBasedRecommender(ABC):
    """
    Abstract base class for text-based recommenders.

    All text-based models inherit from this to ensure
    consistent interface for fitting, predicting, and explaining.
    """

    @abstractmethod
    def fit(self, courses_df: pd.DataFrame, text_column: str = "course_description"):
        """Fit the model on course text data."""
        pass

    @abstractmethod
    def predict(
        self,
        student_features: pd.DataFrame,
        n_recommendations: int = 10,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """Generate recommendations for students."""
        pass

    @abstractmethod
    def explain_recommendation(
        self,
        student_id: str,
        course_id: str,
    ) -> Dict[str, Any]:
        """Explain why a course was recommended."""
        pass


class TFIDFRecommender(TextBasedRecommender):
    """
    TF-IDF based course recommender.

    Converts course descriptions to TF-IDF vectors,
    then matches student profiles (qualification keywords,
    career goals, subject interests) to courses using cosine similarity.

    Advantages:
    - Interpretable (can show which keywords matched)
    - Fast (sparse matrix operations)
    - No external dependencies

    Limitations:
    - Bag-of-words (ignores word order, context)
    - No semantic understanding (synonyms don't match)
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
    ):
        """
        Initialize TF-IDF recommender.

        Args:
            max_features: Maximum vocabulary size
            ngram_range: (min_n, max_n) for n-grams
            min_df: Minimum document frequency
            max_df: Maximum document frequency
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words="english",
        )
        self.cosine_similarity = cosine_similarity

        self.course_vectors = None
        self.course_ids = None
        self.course_texts = None
        self.vocabulary = None

    def _prepare_course_text(self, courses_df: pd.DataFrame) -> List[str]:
        """
        Prepare course text for vectorization.

        Combines multiple text fields into single document per course.

        Args:
            courses_df: Course DataFrame

        Returns:
            List of text documents (one per course)
        """
        texts = []

        for _, row in courses_df.iterrows():
            # Combine multiple text fields
            text_parts = []

            # Course name
            if "course_name" in row:
                text_parts.append(str(row["course_name"]))

            # Department
            if "department" in row:
                text_parts.append(str(row["department"]))

            # Description (if available)
            if "course_description" in row and pd.notna(row["course_description"]):
                text_parts.append(str(row["course_description"]))

            # Modules (if available)
            if "modules" in row and pd.notna(row["modules"]):
                text_parts.append(str(row["modules"]))

            # Career outcomes (if available)
            if "career_outcomes" in row and pd.notna(row["career_outcomes"]):
                text_parts.append(str(row["career_outcomes"]))

            # Entry requirements
            if "entry_requirements" in row and pd.notna(row["entry_requirements"]):
                text_parts.append(str(row["entry_requirements"]))

            texts.append(" ".join(text_parts))

        return texts

    def _prepare_student_text(self, student_features: pd.DataFrame) -> List[str]:
        """
        Prepare student text profiles for matching.

        Combines qualification, interests, and career goals.

        Args:
            student_features: Student feature DataFrame

        Returns:
            List of text documents (one per student)
        """
        texts = []

        for _, row in student_features.iterrows():
            text_parts = []

            # Qualification type
            if "qualification_type" in row and pd.notna(row["qualification_type"]):
                text_parts.append(str(row["qualification_type"]))

            # Subject interests (if available as list or string)
            if "subject_interests" in row and pd.notna(row["subject_interests"]):
                interests = row["subject_interests"]
                if isinstance(interests, list):
                    text_parts.append(" ".join(interests))
                else:
                    text_parts.append(str(interests))

            # Career goals
            if "career_aspirations" in row and pd.notna(row["career_aspirations"]):
                text_parts.append(str(row["career_aspirations"]))

            # Grade keywords
            if "ucas_tariff_points" in row and pd.notna(row["ucas_tariff_points"]):
                tariff = int(row["ucas_tariff_points"])
                if tariff >= 144:
                    text_parts.append("high grades distinction")
                elif tariff >= 112:
                    text_parts.append("good grades merit")
                else:
                    text_parts.append("standard grades pass")

            texts.append(" ".join(text_parts))

        return texts

    def fit(
        self,
        courses_df: pd.DataFrame,
        text_column: Optional[str] = None,
        **kwargs,
    ):
        """
        Fit TF-IDF model on course texts.

        Args:
            courses_df: Course DataFrame with text fields
            text_column: Ignored (uses all text fields)
        """
        logger.info("Fitting TF-IDF recommender...")

        # Prepare course texts
        course_texts = self._prepare_course_text(courses_df)
        self.course_texts = course_texts

        # Fit TF-IDF vectorizer
        self.course_vectors = self.vectorizer.fit_transform(course_texts)
        self.course_ids = courses_df["course_id"].tolist()
        self.vocabulary = self.vectorizer.get_feature_names_out()

        logger.info(f"TF-IDF vocabulary size: {len(self.vocabulary)}")
        logger.info(f"Course vectors shape: {self.course_vectors.shape}")

        return self

    def predict(
        self,
        student_features: pd.DataFrame,
        n_recommendations: int = 10,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Generate TF-IDF based recommendations.

        Args:
            student_features: Student feature DataFrame
            n_recommendations: Number of recommendations per student

        Returns:
            Dictionary mapping student_id to (course_id, score) tuples
        """
        if self.course_vectors is None:
            raise ValueError("Must call fit() before predict()")

        logger.info("Generating TF-IDF recommendations...")

        # Prepare student texts
        student_texts = self._prepare_student_text(student_features)

        # Transform student texts using fitted vectorizer
        student_vectors = self.vectorizer.transform(student_texts)

        # Compute cosine similarity (students x courses)
        similarity_matrix = self.cosine_similarity(student_vectors, self.course_vectors)

        # Generate recommendations
        recommendations = {}
        student_ids = student_features["student_id"].tolist()

        for i, student_id in enumerate(student_ids):
            scores = similarity_matrix[i]

            # Get top N
            top_indices = np.argsort(scores)[::-1][:n_recommendations]
            top_courses = [self.course_ids[j] for j in top_indices]
            top_scores = [scores[j] for j in top_indices]

            recommendations[student_id] = list(zip(top_courses, top_scores))

        return recommendations

    def explain_recommendation(
        self,
        student_id: str,
        course_id: str,
        student_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Explain TF-IDF based recommendation.

        Shows which keywords contributed most to the match.

        Args:
            student_id: Student ID
            course_id: Course ID
            student_text: Optional pre-computed student text

        Returns:
            Explanation dictionary with keyword contributions
        """
        if self.course_vectors is None:
            raise ValueError("Must call fit() before explain()")

        # Get student index
        student_idx = None
        # Would need to store student vectors or recompute

        # Get course index
        if course_id not in self.course_ids:
            return {"error": "Course not found"}

        course_idx = self.course_ids.index(course_id)

        # Get course TF-IDF vector
        course_vector = self.course_vectors[course_idx].toarray().flatten()

        # Get top keywords for this course
        top_keyword_indices = np.argsort(course_vector)[::-1][:10]
        top_keywords = [self.vocabulary[i] for i in top_keyword_indices]
        top_weights = [course_vector[i] for i in top_keyword_indices]

        explanation = {
            "student_id": student_id,
            "course_id": course_id,
            "method": "TF-IDF Cosine Similarity",
            "top_keywords": [
                {"keyword": kw, "tfidf_weight": w}
                for kw, w in zip(top_keywords, top_weights)
            ],
            "vocabulary_size": len(self.vocabulary),
        }

        return explanation


class SemanticEmbeddingRecommender(TextBasedRecommender):
    """
    Semantic embedding-based course recommender using Sentence-BERT.

    Unlike TF-IDF (keyword matching), this model:
    - Understands semantic meaning ("computer science" ≈ "computing")
    - Handles synonyms and related concepts
    - Captures context and word order
    - Produces dense embeddings (768 dimensions)

    Uses pre-trained Sentence Transformers (no training required).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ):
        """
        Initialize semantic embedding recommender.

        Args:
            model_name: Sentence Transformer model name
            embedding_dim: Embedding dimension (depends on model)
            batch_size: Batch size for encoding
            normalize_embeddings: L2 normalize embeddings (for cosine sim)
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        self.model = None
        self.course_embeddings = None
        self.course_ids = None
        self.course_texts = None

    def _load_model(self):
        """Load Sentence Transformer model."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading Sentence Transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")

        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise

    def _prepare_course_text(self, courses_df: pd.DataFrame) -> List[str]:
        """Prepare course text (same as TF-IDF)."""
        texts = []

        for _, row in courses_df.iterrows():
            text_parts = []

            if "course_name" in row:
                text_parts.append(str(row["course_name"]))
            if "department" in row:
                text_parts.append(str(row["department"]))
            if "course_description" in row and pd.notna(row["course_description"]):
                text_parts.append(str(row["course_description"]))
            if "modules" in row and pd.notna(row["modules"]):
                text_parts.append(str(row["modules"]))
            if "career_outcomes" in row and pd.notna(row["career_outcomes"]):
                text_parts.append(str(row["career_outcomes"]))

            texts.append(" ".join(text_parts))

        return texts

    def _prepare_student_text(self, student_features: pd.DataFrame) -> List[str]:
        """Prepare student text profiles."""
        texts = []

        for _, row in student_features.iterrows():
            text_parts = []

            if "qualification_type" in row and pd.notna(row["qualification_type"]):
                text_parts.append(str(row["qualification_type"]))

            if "subject_interests" in row and pd.notna(row["subject_interests"]):
                interests = row["subject_interests"]
                if isinstance(interests, list):
                    text_parts.append(" ".join(interests))
                else:
                    text_parts.append(str(interests))

            if "career_aspirations" in row and pd.notna(row["career_aspirations"]):
                text_parts.append(str(row["career_aspirations"]))

            if "ucas_tariff_points" in row and pd.notna(row["ucas_tariff_points"]):
                tariff = int(row["ucas_tariff_points"])
                if tariff >= 144:
                    text_parts.append("high achieving student with excellent grades")
                elif tariff >= 112:
                    text_parts.append("good student with solid academic performance")
                else:
                    text_parts.append("student meeting standard entry requirements")

            texts.append(" ".join(text_parts))

        return texts

    def fit(
        self,
        courses_df: pd.DataFrame,
        text_column: Optional[str] = None,
        **kwargs,
    ):
        """
        Generate semantic embeddings for courses.

        Args:
            courses_df: Course DataFrame
            text_column: Ignored (uses all text fields)
        """
        logger.info("Fitting Semantic Embedding recommender...")

        # Load model
        if self.model is None:
            self._load_model()

        # Prepare course texts
        course_texts = self._prepare_course_text(courses_df)
        self.course_texts = course_texts
        self.course_ids = courses_df["course_id"].tolist()

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(course_texts)} courses...")
        self.course_embeddings = self.model.encode(
            course_texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        )

        logger.info(f"Course embeddings shape: {self.course_embeddings.shape}")
        return self

    def predict(
        self,
        student_features: pd.DataFrame,
        n_recommendations: int = 10,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Generate semantic recommendations.

        Args:
            student_features: Student feature DataFrame
            n_recommendations: Number of recommendations

        Returns:
            Dictionary mapping student_id to (course_id, score) tuples
        """
        if self.course_embeddings is None:
            raise ValueError("Must call fit() before predict()")

        logger.info("Generating semantic recommendations...")

        # Prepare student texts
        student_texts = self._prepare_student_text(student_features)

        # Generate student embeddings
        student_embeddings = self.model.encode(
            student_texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
        )

        # Compute cosine similarity (embeddings are already normalized)
        similarity_matrix = np.dot(student_embeddings, self.course_embeddings.T)

        # Generate recommendations
        recommendations = {}
        student_ids = student_features["student_id"].tolist()

        for i, student_id in enumerate(student_ids):
            scores = similarity_matrix[i]

            # Get top N
            top_indices = np.argsort(scores)[::-1][:n_recommendations]
            top_courses = [self.course_ids[j] for j in top_indices]
            top_scores = [scores[j] for j in top_indices]

            recommendations[student_id] = list(zip(top_courses, top_scores))

        return recommendations

    def explain_recommendation(
        self,
        student_id: str,
        course_id: str,
    ) -> Dict[str, Any]:
        """
        Explain semantic recommendation.

        Since embeddings are dense and not directly interpretable,
        we use approximate keyword extraction.

        Args:
            student_id: Student ID
            course_id: Course ID

        Returns:
            Explanation dictionary
        """
        if self.course_embeddings is None:
            raise ValueError("Must call fit() before explain()")

        # Get course index
        if course_id not in self.course_ids:
            return {"error": "Course not found"}

        course_idx = self.course_ids.index(course_id)
        course_text = self.course_texts[course_idx]

        # Compute similarity score
        # (would need student embedding - simplified here)

        explanation = {
            "student_id": student_id,
            "course_id": course_id,
            "method": "Semantic Embedding (Sentence-BERT)",
            "model": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "course_text_preview": course_text[:200] + "..." if len(course_text) > 200 else course_text,
            "interpretation": "Semantic matching captures meaning beyond keywords. "
            "Courses are matched based on conceptual similarity in embedding space.",
        }

        return explanation


class MultiModalRecommender(TextBasedRecommender):
    """
    Multi-modal recommender combining text embeddings with structured features.

    Fuses:
    - Text embeddings (semantic meaning from descriptions)
    - Structured features (grades, department, outcomes)
    - Collaborative signals (optional, from other models)

    Uses weighted ensemble or learned fusion (XGBoost).
    """

    def __init__(
        self,
        text_recommender: Optional[TextBasedRecommender] = None,
        structured_recommender: Optional[Any] = None,
        text_weight: float = 0.4,
        structured_weight: float = 0.4,
        collaborative_weight: float = 0.2,
        fusion_method: str = "weighted",
    ):
        """
        Initialize multi-modal recommender.

        Args:
            text_recommender: TF-IDF or Semantic recommender
            structured_recommender: ContentBasedRecommender from baselines
            text_weight: Weight for text-based scores
            structured_weight: Weight for structured feature scores
            collaborative_weight: Weight for collaborative filtering scores
            fusion_method: 'weighted' or 'learned' (XGBoost)
        """
        self.text_recommender = text_recommender
        self.structured_recommender = structured_recommender
        self.text_weight = text_weight
        self.structured_weight = structured_weight
        self.collaborative_weight = collaborative_weight
        self.fusion_method = fusion_method

        self.course_ids = None

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
            enrollments_df: Enrollment records (for collaborative)
        """
        logger.info("Fitting MultiModal recommender...")

        self.course_ids = courses_df["course_id"].tolist()

        # Fit text recommender
        if self.text_recommender is not None:
            logger.info("Fitting text recommender component...")
            self.text_recommender.fit(courses_df)

        # Fit structured recommender
        if self.structured_recommender is not None:
            logger.info("Fitting structured recommender component...")
            self.structured_recommender.fit(student_features_df, courses_df)

        self.course_ids = courses_df["course_id"].tolist()
        logger.info("MultiModal fitting complete")
        return self

    def predict(
        self,
        student_features: pd.DataFrame,
        n_recommendations: int = 10,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Generate multi-modal recommendations with weighted fusion.

        Args:
            student_features: Student feature DataFrame
            n_recommendations: Number of recommendations

        Returns:
            Dictionary mapping student_id to (course_id, fused_score) tuples
        """
        logger.info("Generating multi-modal recommendations...")

        # Get recommendations from each component
        text_recs = {}
        structured_recs = {}

        if self.text_recommender is not None:
            text_recs = self.text_recommender.predict(student_features, n_recommendations)

        if self.structured_recommender is not None:
            structured_recs = self.structured_recommender.predict(
                student_features, n_recommendations
            )

        # Fuse scores
        recommendations = {}
        student_ids = student_features["student_id"].tolist()

        for student_id in student_ids:
            course_scores = {}

            # Aggregate text scores
            if student_id in text_recs:
                for course_id, score in text_recs[student_id]:
                    course_scores[course_id] = course_scores.get(course_id, 0)
                    course_scores[course_id] += self.text_weight * score

            # Aggregate structured scores
            if student_id in structured_recs:
                for course_id, score in structured_recs[student_id]:
                    course_scores[course_id] = course_scores.get(course_id, 0)
                    course_scores[course_id] += self.structured_weight * score

            # Normalize scores (to account for different recommenders returning different counts)
            total_weight = 0
            if student_id in text_recs:
                total_weight += self.text_weight
            if student_id in structured_recs:
                total_weight += self.structured_weight

            if total_weight > 0:
                for course_id in course_scores:
                    course_scores[course_id] /= total_weight

            # Get top N
            sorted_courses = sorted(
                course_scores.items(), key=lambda x: x[1], reverse=True
            )[:n_recommendations]

            recommendations[student_id] = sorted_courses

        return recommendations

    def explain_recommendation(
        self,
        student_id: str,
        course_id: str,
    ) -> Dict[str, Any]:
        """
        Explain multi-modal recommendation.

        Combines explanations from text and structured components.

        Args:
            student_id: Student ID
            course_id: Course ID

        Returns:
            Combined explanation dictionary
        """
        explanation = {
            "student_id": student_id,
            "course_id": course_id,
            "method": "Multi-Modal Fusion",
            "weights": {
                "text": self.text_weight,
                "structured": self.structured_weight,
                "collaborative": self.collaborative_weight,
            },
            "component_explanations": {},
        }

        # Get text explanation
        if self.text_recommender is not None:
            try:
                text_exp = self.text_recommender.explain_recommendation(
                    student_id, course_id
                )
                explanation["component_explanations"]["text"] = text_exp
            except Exception as e:
                explanation["component_explanations"]["text"] = {"error": str(e)}

        # Get structured explanation
        if self.structured_recommender is not None:
            try:
                structured_exp = self.structured_recommender.explain_recommendation(
                    student_id, course_id
                )
                explanation["component_explanations"]["structured"] = structured_exp
            except Exception as e:
                explanation["component_explanations"]["structured"] = {"error": str(e)}

        return explanation


class PGVectorRecommender(TextBasedRecommender):
    """
    PostgreSQL pgvector-based recommender for efficient vector search.

    Stores course embeddings in PostgreSQL with pgvector extension,
    enabling efficient approximate nearest neighbor (ANN) search.

    Advantages:
    - Scales to millions of courses
    - Persistent storage (no reloading)
    - SQL integration (filter by department, entry requirements, etc.)
    - HNSW indexing for fast approximate search

    Requirements:
    - PostgreSQL with pgvector extension
    - Connection string
    """

    def __init__(
        self,
        connection_string: str,
        table_name: str = "course_embeddings",
        embedding_dim: int = 384,
        index_type: str = "hnsw",
    ):
        """
        Initialize pgvector recommender.

        Args:
            connection_string: PostgreSQL connection string
            table_name: Table name for storing embeddings
            embedding_dim: Embedding dimension
            index_type: 'hnsw' or 'ivfflat' for ANN search
        """
        self.connection_string = connection_string
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.index_type = index_type

        self.conn = None
        self.model = None

    def _connect(self):
        """Establish PostgreSQL connection."""
        try:
            import psycopg2
            from pgvector.psycopg2 import register_vector

            self.conn = psycopg2.connect(self.connection_string)
            register_vector(self.conn)
            logger.info("Connected to PostgreSQL with pgvector")

        except ImportError as e:
            logger.error(
                f"Missing dependencies: {e}. "
                "Install with: pip install psycopg2-binary pgvector"
            )
            raise

    def _create_table(self):
        """Create embeddings table if not exists."""
        if self.conn is None:
            self._connect()

        cursor = self.conn.cursor()

        # Create table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                course_id VARCHAR(50) PRIMARY KEY,
                course_name TEXT,
                department VARCHAR(100),
                embedding VECTOR({self.embedding_dim}),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create index for ANN search
        if self.index_type == "hnsw":
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS embedding_hnsw_idx
                ON {self.table_name}
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            """)
        elif self.index_type == "ivfflat":
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS embedding_ivfflat_idx
                ON {self.table_name}
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """)

        self.conn.commit()
        cursor.close()
        logger.info(f"Created table {self.table_name} with {self.index_type} index")

    def fit(
        self,
        courses_df: pd.DataFrame,
        text_column: Optional[str] = None,
        batch_size: int = 100,
    ):
        """
        Generate embeddings and store in PostgreSQL.

        Args:
            courses_df: Course DataFrame
            text_column: Ignored (uses all text fields)
            batch_size: Batch size for database inserts
        """
        logger.info("Fitting PGVector recommender...")

        # Load model
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            logger.error("sentence-transformers not installed")
            raise

        # Connect and create table
        if self.conn is None:
            self._connect()
        self._create_table()

        # Prepare texts and generate embeddings
        texts = []
        course_data = []

        for _, row in courses_df.iterrows():
            text_parts = []
            if "course_name" in row:
                text_parts.append(str(row["course_name"]))
            if "department" in row:
                text_parts.append(str(row["department"]))
            if "course_description" in row and pd.notna(row["course_description"]):
                text_parts.append(str(row["course_description"]))

            texts.append(" ".join(text_parts))
            course_data.append(
                {
                    "course_id": row["course_id"],
                    "course_name": row.get("course_name", ""),
                    "department": row.get("department", ""),
                }
            )

        # Generate embeddings
        logger.info(f"Generating {len(texts)} embeddings...")
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        # Insert into database
        cursor = self.conn.cursor()
        for i, data in enumerate(course_data):
            embedding_str = "[" + ",".join(map(str, embeddings[i])) + "]"

            cursor.execute(f"""
                INSERT INTO {self.table_name}
                (course_id, course_name, department, embedding)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (course_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    course_name = EXCLUDED.course_name,
                    department = EXCLUDED.department
            """, (data["course_id"], data["course_name"], data["department"], embedding_str))

            if (i + 1) % batch_size == 0:
                self.conn.commit()
                logger.info(f"Inserted {i + 1}/{len(course_data)} courses")

        self.conn.commit()
        cursor.close()
        logger.info(f"Stored {len(course_data)} course embeddings in PostgreSQL")

        return self

    def predict(
        self,
        student_features: pd.DataFrame,
        n_recommendations: int = 10,
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Generate recommendations using pgvector ANN search.

        Args:
            student_features: Student feature DataFrame
            n_recommendations: Number of recommendations

        Returns:
            Dictionary mapping student_id to (course_id, score) tuples
        """
        if self.conn is None:
            self._connect()

        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("all-MiniLM-L6-v2")

        recommendations = {}

        for _, row in student_features.iterrows():
            student_id = row["student_id"]

            # Prepare student text
            text_parts = []
            if "qualification_type" in row and pd.notna(row["qualification_type"]):
                text_parts.append(str(row["qualification_type"]))
            if "subject_interests" in row and pd.notna(row["subject_interests"]):
                interests = row["subject_interests"]
                if isinstance(interests, list):
                    text_parts.append(" ".join(interests))
                else:
                    text_parts.append(str(interests))
            if "career_aspirations" in row and pd.notna(row["career_aspirations"]):
                text_parts.append(str(row["career_aspirations"]))

            student_text = " ".join(text_parts)

            # Generate student embedding
            student_embedding = self.model.encode(
                student_text,
                normalize_embeddings=True,
            )
            embedding_str = "[" + ",".join(map(str, student_embedding)) + "]"

            # Query for nearest neighbors
            cursor = self.conn.cursor()
            cursor.execute(f"""
                SELECT course_id, course_name,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {self.table_name}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (embedding_str, embedding_str, n_recommendations))

            results = cursor.fetchall()
            cursor.close()

            recommendations[student_id] = [
                (row[0], float(row[2])) for row in results
            ]

        return recommendations

    def explain_recommendation(
        self,
        student_id: str,
        course_id: str,
    ) -> Dict[str, Any]:
        """Get course details from database."""
        if self.conn is None:
            self._connect()

        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT course_id, course_name, department
            FROM {self.table_name}
            WHERE course_id = %s
        """, (course_id,))

        result = cursor.fetchone()
        cursor.close()

        if result is None:
            return {"error": "Course not found in database"}

        return {
            "student_id": student_id,
            "course_id": course_id,
            "method": "pgvector ANN Search",
            "course_name": result[1],
            "department": result[2],
            "index_type": self.index_type,
        }

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


# Example usage
if __name__ == "__main__":
    from src.data.synthetic import SITSSyntheticGenerator
    from src.data.feature_store import FeatureStore
    from src.models.recommender.baselines import ContentBasedRecommender

    # Generate synthetic data
    print("Generating synthetic data...")
    generator = SITSSyntheticGenerator(n_students=200, n_courses=30, seed=42)
    datasets = generator.generate_all_datasets()

    # Engineer features
    print("\nEngineering features...")
    feature_store = FeatureStore()
    student_features = feature_store.engineer_student_features(
        datasets["students"], datasets["qualifications"]
    )
    course_features = feature_store.engineer_course_features(datasets["courses"])

    # Add synthetic text fields for demonstration
    course_features["course_description"] = course_features["course_name"] + " - " + course_features["department"]
    course_features["career_outcomes"] = "Graduates work in " + course_features["department"] + " roles"

    # === Test TF-IDF ===
    print("\n" + "=" * 60)
    print("TF-IDF RECOMMENDER")
    print("=" * 60)

    tfidf = TFIDFRecommender(max_features=1000, ngram_range=(1, 2))
    tfidf.fit(course_features)

    test_student = student_features.iloc[0:1]
    recs = tfidf.predict(test_student, n_recommendations=5)

    test_id = test_student["student_id"].iloc[0]
    print(f"\nRecommendations for {test_id}:")
    for course_id, score in recs[test_id]:
        print(f"  {course_id}: {score:.4f}")

    # === Test Semantic Embedding ===
    print("\n" + "=" * 60)
    print("SEMANTIC EMBEDDING RECOMMENDER")
    print("=" * 60)

    try:
        semantic = SemanticEmbeddingRecommender(model_name="all-MiniLM-L6-v2")
        semantic.fit(course_features)

        recs = semantic.predict(test_student, n_recommendations=5)
        print(f"\nRecommendations for {test_id}:")
        for course_id, score in recs[test_id]:
            print(f"  {course_id}: {score:.4f}")

        # Get explanation
        exp = semantic.explain_recommendation(test_id, recs[test_id][0][0])
        print(f"\nExplanation: {exp['method']}")
    except ImportError as e:
        print(f"Sentence Transformers not installed: {e}")

    # === Test Multi-Modal ===
    print("\n" + "=" * 60)
    print("MULTI-MODAL RECOMMENDER")
    print("=" * 60)

    structured_rec = ContentBasedRecommender()
    multi = MultiModalRecommender(
        text_recommender=tfidf,
        structured_recommender=structured_rec,
        text_weight=0.5,
        structured_weight=0.5,
    )
    multi.fit(course_features, student_features)

    recs = multi.predict(test_student, n_recommendations=5)
    print(f"\nRecommendations for {test_id}:")
    for course_id, score in recs[test_id]:
        print(f"  {course_id}: {score:.4f}")

    print("\n✓ Advanced content-based models tested!")
