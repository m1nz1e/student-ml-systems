"""
Hyperparameter tuning with Optuna.

Implements Bayesian optimization with pruning for:
- Course Recommender (LightFM + XGBoost)
- Enrollment Yield (XGBoost)
- Early Warning (LSTM/PyTorch)
"""

from typing import Callable, Dict, Any, Optional, Tuple
import optuna
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTuner:
    """Base class for Optuna tuners."""

    def __init__(
        self,
        objective: Callable[[optuna.Trial], float],
        study_name: str = "optimization",
        n_trials: int = 100,
        timeout: Optional[int] = None,
        pruning: bool = True,
        random_state: int = 42,
    ):
        """
        Initialize tuner.

        Args:
            objective: Objective function to optimize
            study_name: Name for the study
            n_trials: Number of trials to run
            timeout: Timeout in seconds (None for no timeout)
            pruning: Whether to enable pruning
            random_state: Random seed
        """
        self.objective = objective
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state

        # Create sampler (TPE for Bayesian optimization)
        sampler = TPESampler(seed=random_state)

        # Create pruner
        if pruning:
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        else:
            pruner = None

        # Create study
        self.study = optuna.create_study(
            study_name=study_name,
            sampler=sampler,
            pruner=pruner,
            direction="maximize",
        )

    def optimize(self) -> Dict[str, Any]:
        """
        Run optimization.

        Returns:
            Best parameters dictionary
        """
        logger.info(f"Starting optimization: {self.n_trials} trials")

        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True,
        )

        logger.info(f"Optimization complete. Best score: {self.study.best_value:.4f}")
        logger.info(f"Best params: {self.study.best_params}")

        return self.study.best_params

    @property
    def best_params(self) -> Dict[str, Any]:
        """Get best parameters."""
        return self.study.best_params

    @property
    def best_score(self) -> float:
        """Get best score."""
        return self.study.best_value

    @property
    def best_trial(self) -> optuna.Trial:
        """Get best trial."""
        return self.study.best_trial

    def get_trials_dataframe(self) -> Any:
        """Get all trials as DataFrame."""
        return self.study.trials_dataframe()


class RecommenderTuner(BaseTuner):
    """Hyperparameter tuner for Course Recommender (Hybrid model)."""

    def __init__(
        self,
        objective: Callable[[optuna.Trial], float],
        n_trials: int = 50,
        timeout: Optional[int] = 3600,
        **kwargs,
    ):
        super().__init__(
            objective=objective,
            study_name="course_recommender_tuning",
            n_trials=n_trials,
            timeout=timeout,
            pruning=True,
            **kwargs,
        )


class EnrollmentTuner(BaseTuner):
    """Hyperparameter tuner for Enrollment Yield (XGBoost)."""

    def __init__(
        self,
        objective: Callable[[optuna.Trial], float],
        n_trials: int = 100,
        timeout: Optional[int] = 7200,
        **kwargs,
    ):
        super().__init__(
            objective=objective,
            study_name="enrollment_yield_tuning",
            n_trials=n_trials,
            timeout=timeout,
            pruning=True,
            **kwargs,
        )


class EarlyWarningTuner(BaseTuner):
    """Hyperparameter tuner for Early Warning (LSTM)."""

    def __init__(
        self,
        objective: Callable[[optuna.Trial], float],
        n_trials: int = 75,
        timeout: Optional[int] = 10800,
        **kwargs,
    ):
        super().__init__(
            objective=objective,
            study_name="early_warning_tuning",
            n_trials=n_trials,
            timeout=timeout,
            pruning=True,
            **kwargs,
        )


# Example objective functions
def example_recommender_objective(trial: optuna.Trial) -> float:
    """
    Example objective for Course Recommender.

    Args:
        trial: Optuna trial object

    Returns:
        NDCG@10 score to maximize
    """
    # Hyperparameter search space
    n_factors = trial.suggest_int("n_factors", 32, 256)
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    min_child_weight = trial.suggest_int("min_child_weight", 1, 10)

    # Import inside to avoid circular imports
    from lightfm import LightFM
    from xgboost import XGBRanker
    from sklearn.model_selection import cross_val_score
    import numpy as np

    # Synthetic data for demo
    np.random.seed(42)
    n_users, n_items = 1000, 200
    X = np.random.randn(n_users, 50)
    y = np.random.randint(0, 3, n_users)

    # Create model with trial parameters
    model = XGBRanker(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        random_state=42,
    )

    # Cross-validation with custom scoring
    from src.evaluation.ranking_metrics import ndcg_at_k

    def ndcg_scorer(estimator, X, y):
        y_pred = estimator.predict(X)
        return ndcg_at_k(y, y_pred, k=10)

    # 5-fold CV
    scores = cross_val_score(model, X, y, cv=5, scoring=ndcg_scorer)

    return np.mean(scores)


def example_enrollment_objective(trial: optuna.Trial) -> float:
    """
    Example objective for Enrollment Yield (XGBoost).

    Args:
        trial: Optuna trial object

    Returns:
        PR-AUC score to maximize
    """
    # XGBoost hyperparameters
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1, 10),
    }

    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import average_precision_score
    import numpy as np

    # Synthetic imbalanced data
    np.random.seed(42)
    n_samples = 5000
    X = np.random.randn(n_samples, 30)
    y = np.random.binomial(1, 0.3, n_samples)  # 30% positive rate

    # Create model
    model = XGBClassifier(**params, random_state=42)

    # Cross-validation with PR-AUC
    def pr_auc_scorer(estimator, X, y):
        y_pred = estimator.predict_proba(X)[:, 1]
        return average_precision_score(y, y_pred)

    scores = cross_val_score(model, X, y, cv=5, scoring=pr_auc_scorer)

    return np.mean(scores)


def example_early_warning_objective(trial: optuna.Trial) -> float:
    """
    Example objective for Early Warning (LSTM).

    Args:
        trial: Optuna trial object

    Returns:
        Recall@K score to maximize
    """
    # LSTM hyperparameters
    hidden_size = trial.suggest_int("hidden_size", 32, 256)
    num_layers = trial.suggest_int("num_layers", 1, 4)
    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.01, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    sequence_length = trial.suggest_int("sequence_length", 4, 12)

    # For demo, return random score (replace with actual LSTM training)
    # In production, train LSTM and evaluate Recall@K
    score = np.random.uniform(0.7, 0.9)

    # Report intermediate scores for pruning
    for step in range(10):
        intermediate_score = score * (0.9 + 0.01 * step)
        trial.report(intermediate_score, step)

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

    return score


# Example usage
if __name__ == "__main__":
    # Example: Tune Course Recommender
    print("=" * 60)
    print("Course Recommender Hyperparameter Tuning")
    print("=" * 60)

    tuner = RecommenderTuner(
        objective=example_recommender_objective,
        n_trials=20,  # Reduced for demo
        timeout=600,  # 10 minutes for demo
    )

    best_params = tuner.optimize()
    print(f"\nBest NDCG@10: {tuner.best_score:.4f}")
    print(f"Best params: {best_params}")

    # Example: Tune Enrollment Yield
    print("\n" + "=" * 60)
    print("Enrollment Yield Hyperparameter Tuning")
    print("=" * 60)

    tuner = EnrollmentTuner(
        objective=example_enrollment_objective,
        n_trials=20,
        timeout=600,
    )

    best_params = tuner.optimize()
    print(f"\nBest PR-AUC: {tuner.best_score:.4f}")
    print(f"Best params: {best_params}")

    # Example: Tune Early Warning
    print("\n" + "=" * 60)
    print("Early Warning Hyperparameter Tuning")
    print("=" * 60)

    tuner = EarlyWarningTuner(
        objective=example_early_warning_objective,
        n_trials=15,
        timeout=600,
    )

    best_params = tuner.optimize()
    print(f"\nBest Recall@K: {tuner.best_score:.4f}")
    print(f"Best params: {best_params}")
