"""
Ranking metrics for Course Recommender System.

Implements:
- NDCG@K (Normalized Discounted Cumulative Gain)
- MAP@K (Mean Average Precision)
- MRR (Mean Reciprocal Rank)
- Precision@K
- Recall@K
- Intra-list distance (diversity)
- Coverage (catalog coverage)
- Novelty
"""

from typing import List, Union, Tuple
import numpy as np
from collections import defaultdict


def dcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """
    Discounted Cumulative Gain at K.

    Args:
        y_true: Ground truth relevance scores (n_samples,)
        y_pred: Predicted relevance scores (n_samples,)
        k: Cut-off rank

    Returns:
        DCG@K score
    """
    # Sort by predicted scores (descending)
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = y_true[order[:k]]

    # DCG formula: sum(rel_i / log2(i+1))
    gains = y_true_sorted
    discounts = np.log2(np.arange(len(gains)) + 2)
    dcg = np.sum(gains / discounts)

    return dcg


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """
    Normalized Discounted Cumulative Gain at K.

    Primary metric for Course Recommender.

    Args:
        y_true: Ground truth relevance scores (n_samples,)
        y_pred: Predicted relevance scores (n_samples,)
        k: Cut-off rank

    Returns:
        NDCG@K score (0-1, higher is better)
    """
    dcg = dcg_at_k(y_true, y_pred, k)

    # Ideal DCG (perfect ranking)
    idcg = dcg_at_k(y_true, y_true, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def apk(actual: List[int], predicted: List[int], k: int = 10) -> float:
    """
    Average Precision at K.

    Args:
        actual: List of relevant item IDs
        predicted: List of predicted item IDs (ordered)
        k: Cut-off rank

    Returns:
        AP@K score
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, item in enumerate(predicted):
        if item in actual and item not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def map_at_k(actual: List[List[int]], predicted: List[List[int]], k: int = 10) -> float:
    """
    Mean Average Precision at K.

    Args:
        actual: List of lists of relevant item IDs (one per user)
        predicted: List of lists of predicted item IDs (one per user)
        k: Cut-off rank

    Returns:
        MAP@K score
    """
    aps = [apk(a, p, k) for a, p in zip(actual, predicted)]
    return np.mean(aps)


def mrr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Reciprocal Rank.

    Args:
        y_true: Ground truth relevance (binary or multi-level)
        y_pred: Predicted scores

    Returns:
        MRR score
    """
    # Find rank of first relevant item
    order = np.argsort(y_pred)[::-1]
    y_sorted = y_true[order]

    # Find first relevant item
    relevant_indices = np.where(y_sorted > 0)[0]
    if len(relevant_indices) == 0:
        return 0.0

    first_relevant = relevant_indices[0]
    return 1.0 / (first_relevant + 1)


def precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """
    Precision at K.

    Args:
        y_true: Binary relevance (1=relevant, 0=not)
        y_pred: Predicted scores
        k: Cut-off rank

    Returns:
        Precision@K
    """
    order = np.argsort(y_pred)[::-1]
    y_sorted = y_true[order[:k]]

    return np.mean(y_sorted)


def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """
    Recall at K.

    Args:
        y_true: Binary relevance
        y_pred: Predicted scores
        k: Cut-off rank

    Returns:
        Recall@K
    """
    order = np.argsort(y_pred)[::-1]
    y_sorted = y_true[order[:k]]

    total_relevant = np.sum(y_true)
    if total_relevant == 0:
        return 0.0

    return np.sum(y_sorted) / total_relevant


def intra_list_distance(
    recommendations: List[List[int]],
    item_features: np.ndarray,
) -> float:
    """
    Intra-list distance (diversity metric).

    Measures how diverse the recommended items are.

    Args:
        recommendations: List of recommended item IDs per user
        item_features: Feature matrix for all items (n_items, n_features)

    Returns:
        Average intra-list distance (higher = more diverse)
    """
    distances = []

    for recs in recommendations:
        if len(recs) < 2:
            continue

        # Get features for recommended items
        rec_features = item_features[recs]

        # Calculate pairwise distances
        pairwise_dist = []
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                dist = np.linalg.norm(rec_features[i] - rec_features[j])
                pairwise_dist.append(dist)

        if pairwise_dist:
            distances.append(np.mean(pairwise_dist))

    return np.mean(distances) if distances else 0.0


def coverage_score(
    all_items: Union[List[int], np.ndarray],
    recommended_items: Union[List[int], np.ndarray],
) -> float:
    """
    Catalog coverage (what % of items are recommended at least once).

    Args:
        all_items: All available item IDs
        recommended_items: All items that were recommended

    Returns:
        Coverage ratio (0-1)
    """
    all_items = set(all_items)
    recommended_items = set(recommended_items)

    if len(all_items) == 0:
        return 0.0

    return len(recommended_items & all_items) / len(all_items)


def novelty_score(
    recommendations: List[List[int]],
    item_popularity: np.ndarray,
    n_users: int,
) -> float:
    """
    Novelty (average popularity of recommended items).

    Args:
        recommendations: List of recommended item IDs per user
        item_popularity: Popularity count for each item
        n_users: Total number of users

    Returns:
        Average novelty (higher = less popular items recommended)
    """
    novelty_scores = []

    for recs in recommendations:
        if not recs:
            continue

        # Calculate self-information: -log2(p(item))
        rec_novelty = []
        for item_id in recs:
            p_item = item_popularity[item_id] / n_users
            if p_item > 0:
                novelty = -np.log2(p_item)
                rec_novelty.append(novelty)

        if rec_novelty:
            novelty_scores.append(np.mean(rec_novelty))

    return np.mean(novelty_scores) if novelty_scores else 0.0


def evaluate_ranking(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k_values: List[int] = [5, 10, 20],
) -> dict:
    """
    Comprehensive ranking evaluation.

    Args:
        y_true: Ground truth relevance scores
        y_pred: Predicted scores
        k_values: List of K values to evaluate

    Returns:
        Dictionary of all metrics
    """
    metrics = {}

    # NDCG@K
    for k in k_values:
        metrics[f"ndcg_at_{k}"] = ndcg_at_k(y_true, y_pred, k)

    # Precision@K and Recall@K
    for k in k_values:
        metrics[f"precision_at_{k}"] = precision_at_k(y_true, y_pred, k)
        metrics[f"recall_at_{k}"] = recall_at_k(y_true, y_pred, k)

    # MRR
    metrics["mrr"] = mrr(y_true, y_pred)

    return metrics


# Example usage
if __name__ == "__main__":
    # Synthetic test data
    np.random.seed(42)
    n_samples = 100

    # Ground truth relevance (0-3 scale)
    y_true = np.random.randint(0, 4, n_samples)

    # Predicted scores
    y_pred = np.random.randn(n_samples)

    # Evaluate
    print("Ranking Metrics:")
    print(f"  NDCG@5:  {ndcg_at_k(y_true, y_pred, k=5):.4f}")
    print(f"  NDCG@10: {ndcg_at_k(y_true, y_pred, k=10):.4f}")
    print(f"  NDCG@20: {ndcg_at_k(y_true, y_pred, k=20):.4f}")
    print(f"  MRR:     {mrr(y_true, y_pred):.4f}")
    print(f"  Precision@10: {precision_at_k(y_true, y_pred, k=10):.4f}")
    print(f"  Recall@10:    {recall_at_k(y_true, y_pred, k=10):.4f}")

    # Full evaluation
    all_metrics = evaluate_ranking(y_true, y_pred, k_values=[5, 10, 20])
    print("\nFull Evaluation:")
    for metric, score in all_metrics.items():
        print(f"  {metric}: {score:.4f}")
