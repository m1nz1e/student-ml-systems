"""
Collaborative Filtering models for Course Recommender System.

Implements:
1. Matrix Factorization (SVD) — Classic CF
2. LightFM — Hybrid collaborative filtering
3. Neural Collaborative Filtering — Deep learning approach

These models learn from user-item interactions to discover
latent patterns in course enrollment behavior.
"""

from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MatrixFactorization:
    """
    Matrix Factorization using SVD.

    Decomposes user-item interaction matrix into lower-dimensional
    latent factor matrices for users and items.

    R ≈ U × V^T

    Where:
    - R: User-item rating matrix
    - U: User latent factors (n_users, n_factors)
    - V: Item latent factors (n_items, n_factors)
    """

    def __init__(
        self,
        n_factors: int = 50,
        n_epochs: int = 30,
        learning_rate: float = 0.005,
        regularization: float = 0.02,
        random_state: int = 42,
    ):
        """
        Initialize matrix factorization model.

        Args:
            n_factors: Number of latent factors
            n_epochs: Number of training epochs
            learning_rate: SGD learning rate
            regularization: L2 regularization term
            random_state: Random seed
        """
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.random_state = random_state

        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None
        self.user_bias: Optional[np.ndarray] = None
        self.item_bias: Optional[np.ndarray] = None
        self.global_bias: float = 0.0

        self.user_id_map: Optional[Dict[str, int]] = None
        self.item_id_map: Optional[Dict[str, int]] = None
        self.reverse_user_map: Optional[Dict[int, str]] = None
        self.reverse_item_map: Optional[Dict[int, str]] = None

        np.random.seed(random_state)

    def _build_interaction_matrix(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = "student_id",
        item_col: str = "course_id",
        rating_col: Optional[str] = None,
    ) -> csr_matrix:
        """
        Build sparse user-item interaction matrix.

        Args:
            interactions_df: DataFrame with user-item interactions
            user_col: User ID column name
            item_col: Item ID column name
            rating_col: Optional rating column (binary if None)

        Returns:
            Sparse interaction matrix (n_users, n_items)
        """
        # Create ID mappings
        unique_users = interactions_df[user_col].unique()
        unique_items = interactions_df[item_col].unique()

        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        self.reverse_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.reverse_item_map = {idx: iid for iid, idx in self.item_id_map.items()}

        # Map to indices
        user_indices = interactions_df[user_col].map(self.user_id_map).values
        item_indices = interactions_df[item_col].map(self.item_id_map).values

        # Ratings (binary if not provided)
        if rating_col and rating_col in interactions_df.columns:
            ratings = interactions_df[rating_col].values
        else:
            ratings = np.ones(len(interactions_df))

        # Build sparse matrix
        n_users = len(unique_users)
        n_items = len(unique_items)

        interaction_matrix = csr_matrix(
            (ratings, (user_indices, item_indices)),
            shape=(n_users, n_items),
        )

        logger.info(f"Built interaction matrix: {interaction_matrix.shape}")
        logger.info(f"Sparsity: {1 - interaction_matrix.nnz / (n_users * n_items):.2%}")

        return interaction_matrix

    def fit(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = "student_id",
        item_col: str = "course_id",
        rating_col: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Fit model using SGD.

        Args:
            interactions_df: User-item interaction DataFrame
            user_col: User ID column
            item_col: Item ID column
            rating_col: Optional rating column
            verbose: Print training progress
        """
        logger.info("Fitting MatrixFactorization model...")

        # Build interaction matrix
        R = self._build_interaction_matrix(
            interactions_df, user_col, item_col, rating_col
        )
        n_users, n_items = R.shape

        # Initialize factors (small random values)
        self.user_factors = np.random.normal(
            0, 0.1, (n_users, self.n_factors)
        )
        self.item_factors = np.random.normal(
            0, 0.1, (n_items, self.n_factors)
        )
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)

        # Global bias (mean rating)
        self.global_bias = R.data.mean() if R.nnz > 0 else 0.0

        # Get non-zero entries
        users, items = R.nonzero()
        ratings = R.data

        # SGD training
        for epoch in range(self.n_epochs):
            # Shuffle training data
            perm = np.random.permutation(len(ratings))
            users = users[perm]
            items = items[perm]
            ratings = ratings[perm]

            epoch_loss = 0.0

            for u, i, r in zip(users, items, ratings):
                # Predict
                pred = (
                    self.global_bias
                    + self.user_bias[u]
                    + self.item_bias[i]
                    + np.dot(self.user_factors[u], self.item_factors[i])
                )

                # Error
                error = r - pred
                epoch_loss += error ** 2

                # Update biases
                self.user_bias[u] += self.learning_rate * (
                    error - self.regularization * self.user_bias[u]
                )
                self.item_bias[i] += self.learning_rate * (
                    error - self.regularization * self.item_bias[i]
                )

                # Update factors
                user_factor = self.user_factors[u].copy()
                self.user_factors[u] += self.learning_rate * (
                    error * self.item_factors[i]
                    - self.regularization * self.user_factors[u]
                )
                self.item_factors[i] += self.learning_rate * (
                    error * user_factor
                    - self.regularization * self.item_factors[i]
                )

            if verbose and (epoch + 1) % 5 == 0:
                rmse = np.sqrt(epoch_loss / len(ratings))
                logger.info(f"  Epoch {epoch + 1}/{self.n_epochs}, RMSE: {rmse:.4f}")

        logger.info(f"MatrixFactorization training complete")
        return self

    def predict(
        self,
        user_id: str,
        item_id: str,
    ) -> float:
        """
        Predict rating for a user-item pair.

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            Predicted rating
        """
        if user_id not in self.user_id_map or item_id not in self.item_id_map:
            return self.global_bias  # Cold start

        u = self.user_id_map[user_id]
        i = self.item_id_map[item_id]

        pred = (
            self.global_bias
            + self.user_bias[u]
            + self.item_bias[i]
            + np.dot(self.user_factors[u], self.item_factors[i])
        )

        return pred

    def recommend(
        self,
        user_id: str,
        n_recommendations: int = 10,
        exclude_items: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Generate top N recommendations for a user.

        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            exclude_items: Items to exclude (already enrolled)

        Returns:
            List of (item_id, score) tuples
        """
        if user_id not in self.user_id_map:
            # Cold start: return global average for all items
            scores = [self.global_bias] * len(self.item_id_map)
        else:
            u = self.user_id_map[user_id]
            scores = (
                self.global_bias
                + self.user_bias[u]
                + np.dot(self.user_factors[u], self.item_factors.T)
            )

        # Get all item IDs and scores
        all_items = list(self.item_id_map.keys())
        all_scores = scores.tolist()

        # Exclude items
        if exclude_items:
            filtered_items = []
            filtered_scores = []
            for item, score in zip(all_items, all_scores):
                if item not in exclude_items:
                    filtered_items.append(item)
                    filtered_scores.append(score)
            all_items = filtered_items
            all_scores = filtered_scores

        # Get top N
        top_indices = np.argsort(all_scores)[::-1][:n_recommendations]
        top_items = [all_items[i] for i in top_indices]
        top_scores = [all_scores[i] for i in top_indices]

        return list(zip(top_items, top_scores))

    def get_item_similarity(
        self,
        item_id: str,
        n_similar: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Find similar items based on latent factors.

        Args:
            item_id: Item ID
            n_similar: Number of similar items

        Returns:
            List of (item_id, similarity) tuples
        """
        if item_id not in self.item_id_map:
            return []

        i = self.item_id_map[item_id]
        item_vector = self.item_factors[i]

        # Cosine similarity with all items
        norms = np.linalg.norm(self.item_factors, axis=1)
        norms[norms == 0] = 1  # Avoid division by zero

        similarities = np.dot(self.item_factors, item_vector) / norms
        similarities /= norms[i]

        # Get top N (excluding self)
        all_items = list(self.item_id_map.keys())
        similarities[self.item_id_map[item_id]] = -1  # Exclude self

        top_indices = np.argsort(similarities)[::-1][:n_similar]
        top_items = [all_items[idx] for idx in top_indices]
        top_sims = [similarities[idx] for idx in top_indices]

        return list(zip(top_items, top_sims))


class LightFMRecommender:
    """
    Hybrid Collaborative Filtering with LightFM.

    Combines collaborative filtering with content-based features
    for both users and items. Handles cold-start better than pure CF.

    Supports multiple loss functions:
    - 'logistic': Binary classification
    - 'bpr': Bayesian Personalized Ranking
    - 'warp': Weighted Approximate-Rank Pairwise (best for implicit)
    """

    def __init__(
        self,
        n_factors: int = 30,
        learning_rate: float = 0.05,
        epochs: int = 20,
        loss: str = "warp",
        regularization: float = 0.002,
        random_state: int = 42,
    ):
        """
        Initialize LightFM recommender.

        Args:
            n_factors: Number of latent factors
            learning_rate: Learning rate for AdaGrad
            epochs: Number of training epochs
            loss: Loss function ('logistic', 'bpr', 'warp')
            regularization: L2 regularization
            random_state: Random seed
        """
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss = loss
        self.regularization = regularization
        self.random_state = random_state
        self.random_state = random_state

        self.model = None
        self.user_features = None
        self.item_features = None
        self.user_id_map = None
        self.item_id_map = None

    def _build_feature_matrices(
        self,
        user_features_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
        user_id_col: str = "student_id",
        item_id_col: str = "course_id",
    ):
        """
        Build feature matrices for LightFM.

        Args:
            user_features_df: User feature DataFrame
            item_features_df: Item feature DataFrame
            user_id_col: User ID column
            item_id_col: Item ID column
        """
        from lightfm import LightFM
        from scipy.sparse import csr_matrix

        # Create ID mappings
        self.user_id_map = {
            uid: idx for idx, uid in enumerate(user_features_df[user_id_col])
        }
        self.item_id_map = {
            iid: idx for idx, iid in enumerate(item_features_df[item_id_col])
        }

        # Extract numeric features (exclude ID columns)
        user_feature_cols = [
            col
            for col in user_features_df.columns
            if col != user_id_col and user_features_df[col].dtype in ["int64", "float64"]
        ]
        item_feature_cols = [
            col
            for col in item_features_df.columns
            if col != item_id_col and item_features_df[col].dtype in ["int64", "float64"]
        ]

        # Build feature matrices
        self.user_features = csr_matrix(
            user_features_df[user_feature_cols].fillna(0).values
        )
        self.item_features = csr_matrix(
            item_features_df[item_feature_cols].fillna(0).values
        )

        logger.info(f"User features: {self.user_features.shape}")
        logger.info(f"Item features: {self.item_features.shape}")

    def fit(
        self,
        interactions_df: pd.DataFrame,
        user_features_df: Optional[pd.DataFrame] = None,
        item_features_df: Optional[pd.DataFrame] = None,
        user_col: str = "student_id",
        item_col: str = "course_id",
        rating_col: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Fit LightFM model.

        Args:
            interactions_df: User-item interaction DataFrame
            user_features_df: Optional user feature DataFrame
            item_features_df: Optional item feature DataFrame
            user_col: User ID column
            item_col: Item ID column
            rating_col: Optional rating column
            verbose: Print training progress
        """
        from lightfm import LightFM

        logger.info(f"Fitting LightFM model (loss={self.loss})...")

        # Initialize model
        self.model = LightFM(
            no_components=self.n_factors,
            learning_rate=self.learning_rate,
            loss=self.loss,
            reg=self.regularization,
            random_state=self.random_state,
        )

        # Build feature matrices if provided
        if user_features_df is not None and item_features_df is not None:
            self._build_feature_matrices(
                user_features_df, item_features_df, user_col, item_col
            )

        # Build interaction matrix
        unique_users = interactions_df[user_col].unique()
        unique_items = interactions_df[item_col].unique()

        user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}

        user_indices = interactions_df[user_col].map(user_id_map).values
        item_indices = interactions_df[item_col].map(item_id_map).values

        if rating_col and rating_col in interactions_df.columns:
            ratings = interactions_df[rating_col].values
        else:
            ratings = np.ones(len(interactions_df))

        interaction_matrix = csr_matrix(
            (ratings, (user_indices, item_indices)),
            shape=(len(unique_users), len(unique_items)),
        )

        # Fit model
        self.model.fit(
            interaction_matrix,
            user_features=self.user_features,
            item_features=self.item_features,
            epochs=self.epochs,
            verbose=verbose,
        )

        logger.info("LightFM training complete")
        return self

    def predict(
        self,
        user_id: str,
        item_id: str,
    ) -> float:
        """
        Predict score for a user-item pair.

        Args:
            user_id: User ID
            item_id: Item ID

        Returns:
            Predicted score
        """
        if self.model is None:
            raise ValueError("Must call fit() before predict()")

        # Handle cold start
        if user_id not in self.user_id_map or item_id not in self.item_id_map:
            return 0.0

        user_idx = self.user_id_map[user_id]
        item_idx = self.item_id_map[item_id]

        # Get user and item features
        user_feat = (
            self.user_features[user_idx]
            if self.user_features is not None
            else None
        )
        item_feat = (
            self.item_features[item_idx]
            if self.item_features is not None
            else None
        )

        # Predict
        pred = self.model.predict(
            np.array([user_idx]),
            np.array([item_idx]),
            user_features=user_feat,
            item_features=item_feat,
        )[0]

        return pred

    def recommend(
        self,
        user_id: str,
        n_recommendations: int = 10,
        exclude_items: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Generate top N recommendations for a user.

        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            exclude_items: Items to exclude

        Returns:
            List of (item_id, score) tuples
        """
        if self.model is None:
            raise ValueError("Must call fit() before recommend()")

        if user_id not in self.user_id_map:
            # Cold start: return popular items
            logger.warning(f"Cold start for user {user_id}, returning popular items")
            return []

        user_idx = self.user_id_map[user_id]

        # Get user features
        user_feat = self.user_features[user_idx] if self.user_features is not None else None

        # Get all item indices
        all_item_indices = np.arange(len(self.item_id_map))

        # Get item features
        item_feats = self.item_features if self.item_features is not None else None

        # Predict scores for all items
        scores = self.model.predict(
            np.array([user_idx]),
            all_item_indices,
            user_features=user_feat,
            item_features=item_feats,
        )[0]

        # Map back to item IDs
        reverse_item_map = {idx: iid for iid, idx in self.item_id_map.items()}
        all_items = [reverse_item_map[idx] for idx in all_item_indices]

        # Exclude items
        if exclude_items:
            filtered_items = []
            filtered_scores = []
            for item, score in zip(all_items, scores):
                if item not in exclude_items:
                    filtered_items.append(item)
                    filtered_scores.append(score)
            all_items = filtered_items
            all_scores = filtered_scores
        else:
            all_scores = scores.tolist()

        # Get top N
        top_indices = np.argsort(all_scores)[::-1][:n_recommendations]
        top_items = [all_items[i] for i in top_indices]
        top_scores = [all_scores[i] for i in top_indices]

        return list(zip(top_items, top_scores))


class NeuralCollaborativeFiltering:
    """
    Neural Collaborative Filtering (NCF) with PyTorch.

    Combines matrix factorization with multi-layer perceptron
    to learn non-linear user-item interactions.

    Architecture:
    - User embedding + Item embedding
    - Concatenation → MLP layers → Output
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int = 32,
        mlp_layers: List[int] = [64, 32, 16],
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        random_state: int = 42,
    ):
        """
        Initialize NCF model.

        Args:
            n_users: Number of users
            n_items: Number of items
            n_factors: Embedding dimension
            mlp_layers: List of MLP hidden layer sizes
            dropout: Dropout rate
            learning_rate: Learning rate
            random_state: Random seed
        """
        import torch

        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.mlp_layers = mlp_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.random_state = random_state

        torch.manual_seed(random_state)

        # Build model
        self.model = self._build_model()
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.user_id_map = None
        self.item_id_map = None
        self.reverse_user_map = None
        self.reverse_item_map = None

    def _build_model(self):
        """Build NCF model architecture."""
        import torch
        import torch.nn as nn

        class NCFModel(nn.Module):
            def __init__(
                self,
                n_users,
                n_items,
                n_factors,
                mlp_layers,
                dropout,
            ):
                super().__init__()

                # Embeddings
                self.user_embedding = nn.Embedding(n_users, n_factors)
                self.item_embedding = nn.Embedding(n_items, n_factors)

                # MLP layers
                mlp_input_size = n_factors * 2  # Concatenate user + item
                mlp_modules = []
                for hidden_size in mlp_layers:
                    mlp_modules.extend(
                        [
                            nn.Linear(mlp_input_size, hidden_size),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                        ]
                    )
                    mlp_input_size = hidden_size

                self.mlp = nn.Sequential(*mlp_modules)

                # Output layer
                self.output = nn.Linear(mlp_layers[-1], 1)

                # Initialize weights
                nn.init.xavier_uniform_(self.user_embedding.weight)
                nn.init.xavier_uniform_(self.item_embedding.weight)

            def forward(self, user_ids, item_ids):
                user_emb = self.user_embedding(user_ids)
                item_emb = self.item_embedding(item_ids)

                # Concatenate
                x = torch.cat([user_emb, item_emb], dim=1)

                # MLP
                x = self.mlp(x)

                # Output
                return self.output(x)

        return NCFModel(
            self.n_users,
            self.n_items,
            self.n_factors,
            self.mlp_layers,
            self.dropout,
        )

    def fit(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = "student_id",
        item_col: str = "course_id",
        rating_col: Optional[str] = None,
        n_epochs: int = 20,
        batch_size: int = 256,
        verbose: bool = True,
    ):
        """
        Fit NCF model.

        Args:
            interactions_df: User-item interaction DataFrame
            user_col: User ID column
            item_col: Item ID column
            rating_col: Optional rating column (binary if None)
            n_epochs: Number of training epochs
            batch_size: Training batch size
            verbose: Print training progress
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        logger.info("Fitting NeuralCollaborativeFiltering model...")

        # Create ID mappings
        unique_users = interactions_df[user_col].unique()
        unique_items = interactions_df[item_col].unique()

        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        self.reverse_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.reverse_item_map = {idx: iid for iid, idx in self.item_id_map.items()}

        # Map to indices
        user_indices = (
            torch.LongTensor(
                interactions_df[user_col].map(self.user_id_map).values
            )
        )
        item_indices = (
            torch.LongTensor(
                interactions_df[item_col].map(self.item_id_map).values
            )
        )

        # Ratings (binary)
        if rating_col and rating_col in interactions_df.columns:
            ratings = torch.FloatTensor(interactions_df[rating_col].values)
        else:
            ratings = torch.ones(len(interactions_df))

        # Create DataLoader
        dataset = TensorDataset(user_indices, item_indices, ratings)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_users, batch_items, batch_ratings in dataloader:
                self.optimizer.zero_grad()

                # Forward pass
                predictions = self.model(batch_users, batch_items).squeeze()

                # Loss
                loss = self.criterion(predictions, batch_ratings)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if verbose and (epoch + 1) % 5 == 0:
                avg_loss = epoch_loss / n_batches
                logger.info(f"  Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

        logger.info("NCF training complete")
        return self

    def recommend(
        self,
        user_id: str,
        n_recommendations: int = 10,
        exclude_items: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a user.

        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            exclude_items: Items to exclude

        Returns:
            List of (item_id, score) tuples
        """
        import torch

        if self.model is None:
            raise ValueError("Must call fit() before recommend()")

        if user_id not in self.user_id_map:
            logger.warning(f"Cold start for user {user_id}")
            return []

        user_idx = self.user_id_map[user_id]

        self.model.eval()
        with torch.no_grad():
            # Get scores for all items
            user_tensor = torch.LongTensor([user_idx])
            all_item_indices = torch.arange(self.n_items)

            scores = self.model(
                user_tensor.repeat(self.n_items), all_item_indices
            ).squeeze()

            scores = scores.numpy()

        # Map to item IDs
        all_items = [
            self.reverse_item_map[idx] for idx in range(self.n_items)
        ]
        all_scores = scores.tolist()

        # Exclude items
        if exclude_items:
            filtered_items = []
            filtered_scores = []
            for item, score in zip(all_items, all_scores):
                if item not in exclude_items:
                    filtered_items.append(item)
                    filtered_scores.append(score)
            all_items = filtered_items
            all_scores = filtered_scores

        # Get top N
        top_indices = np.argsort(all_scores)[::-1][:n_recommendations]
        top_items = [all_items[i] for i in top_indices]
        top_scores = [all_scores[i] for i in top_indices]

        return list(zip(top_items, top_scores))


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

    # Prepare interactions
    interactions = datasets["enrollments"][["student_id", "course_id"]].copy()
    interactions["rating"] = 1  # Binary interactions

    print(f"Interactions: {len(interactions)}")

    # === Test Matrix Factorization ===
    print("\n" + "=" * 60)
    print("MATRIX FACTORIZATION")
    print("=" * 60)

    mf = MatrixFactorization(n_factors=30, n_epochs=20, learning_rate=0.01)
    mf.fit(interactions, verbose=True)

    # Get recommendations for a user
    test_user = interactions["student_id"].iloc[0]
    recs = mf.recommend(test_user, n_recommendations=5)
    print(f"\nRecommendations for {test_user}:")
    for course_id, score in recs:
        print(f"  {course_id}: {score:.4f}")

    # === Test LightFM ===
    print("\n" + "=" * 60)
    print("LIGHTFM")
    print("=" * 60)

    try:
        lightfm_rec = LightFMRecommender(
            n_factors=30, epochs=10, loss="warp"
        )
        lightfm_rec.fit(
            interactions,
            user_features_df=student_features,
            item_features_df=course_features,
            verbose=True,
        )

        recs = lightfm_rec.recommend(test_user, n_recommendations=5)
        print(f"\nRecommendations for {test_user}:")
        for course_id, score in recs:
            print(f"  {course_id}: {score:.4f}")
    except ImportError as e:
        print(f"LightFM not installed: {e}")
        print("Install with: pip install lightfm")

    # === Test Neural CF ===
    print("\n" + "=" * 60)
    print("NEURAL COLLABORATIVE FILTERING")
    print("=" * 60)

    try:
        ncf = NeuralCollaborativeFiltering(
            n_users=len(student_features),
            n_items=len(course_features),
            n_factors=32,
            mlp_layers=[64, 32],
        )
        ncf.fit(interactions, n_epochs=10, verbose=True)

        recs = ncf.recommend(test_user, n_recommendations=5)
        print(f"\nRecommendations for {test_user}:")
        for course_id, score in recs:
            print(f"  {course_id}: {score:.4f}")
    except ImportError as e:
        print(f"PyTorch not available: {e}")

    print("\n✓ Collaborative filtering models tested!")
