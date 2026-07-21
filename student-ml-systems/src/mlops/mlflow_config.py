"""
MLflow Integration — experiment tracking and model registry.

Provides:
    - Auto-logging for PyTorch/sklearn models
    - Experiment versioning
    - Model comparison
    - Artifact storage
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    Wrapper for MLflow experiment tracking and model registry.

    Usage:
        tracker = MLflowTracker(tracking_uri="sqlite:///mlflow.db")
        tracker.start_experiment("early-warning-lstm")
        tracker.log_params({"hidden_dim": 64, "n_layers": 2})
        tracker.log_metrics({"val_loss": 0.36, "c_index": 0.81})
        tracker.log_model(model, artifact_path="lstm-ew")
        tracker.register_model("lstm-early-warning", "runs:/.../lstm-ew")
    """

    def __init__(
        self,
        tracking_uri: str = "sqlite:///mlflow.db",
        experiment_name: Optional[str] = None,
        artifact_root: str = "./mlruns",
    ):
        """
        Initialize the MLflow tracker.

        Args:
            tracking_uri: Database URI for MLflow tracking (e.g., sqlite, postgresql)
            experiment_name: Default experiment name
            artifact_root: Local directory for artifacts
        """
        self.tracking_uri = tracking_uri
        self.artifact_root = artifact_root
        self._client: Optional[MlflowClient] = None

        # Configure MLflow
        mlflow.set_tracking_uri(tracking_uri)
        # Artifact root is set per-experiment; use MLFLOW_ARTIFACT_ROOT env var or log_artifacts()

        if experiment_name:
            mlflow.set_experiment(experiment_name)

        logger.info(f"MLflowTracker initialized — tracking_uri={tracking_uri}")

    @property
    def client(self) -> MlflowClient:
        if self._client is None:
            self._client = MlflowClient(self.tracking_uri)
        return self._client

    # ── Experiment management ─────────────────────────────────────────────────

    def start_experiment(self, experiment_name: str) -> str:
        """
        Create or switch to an experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Experiment ID
        """
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            exp_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created experiment '{experiment_name}' (id={exp_id})")
        else:
            exp_id = exp.experiment_id
            mlflow.set_experiment(experiment_name)
            logger.info(f"Switched to experiment '{experiment_name}' (id={exp_id})")

        return exp_id

    def get_experiment(self, experiment_name: str) -> Optional[Any]:
        """Get experiment info by name."""
        return mlflow.get_experiment_by_name(experiment_name)

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments."""
        exps = mlflow.search_experiments()
        return [
            {
                "name": e.name if hasattr(e, 'name') else str(e),
                "experiment_id": e.experiment_id if hasattr(e, 'experiment_id') else str(e),
                "lifecycle_stage": e.lifecycle_stage if hasattr(e, 'lifecycle_stage') else "unknown",
            }
            for e in exps
        ]

    # ── Run context ──────────────────────────────────────────────────────────

    def start_run(
        self,
        run_name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Start a new MLflow run.

        Args:
            run_name: Human-readable run name
            description: Run description
            tags: Metadata tags

        Returns:
            Active MLflow run
        """
        run = mlflow.start_run(
            run_name=run_name,
            description=description,
            tags=tags,
        )
        logger.info(f"Started run: {run.info.run_id}")
        return run

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run."""
        mlflow.end_run(status=status)
        logger.info(f"Run ended with status: {status}")

    @property
    def active_run(self) -> Optional[Any]:
        """Get the currently active run."""
        return mlflow.active_run()

    # ── Logging ──────────────────────────────────────────────────────────────

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters.

        Args:
            params: Dictionary of parameter names → values
        """
        mlflow.log_params(params)
        for k, v in params.items():
            logger.debug(f"  param: {k}={v}")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """
        Log metrics.

        Args:
            metrics: Dictionary of metric names → values
            step: Optional step number for time-series metrics
        """
        mlflow.log_metrics(metrics, step=step)
        for k, v in metrics.items():
            logger.debug(f"  metric: {k}={v}")

    def log_artifacts(
        self,
        local_dir: str,
        artifact_path: Optional[str] = None,
    ) -> None:
        """
        Log a directory of artifacts.

        Args:
            local_dir: Local directory to upload
            artifact_path: Path within MLflow artifacts
        """
        mlflow.log_artifacts(local_dir, artifact_path=artifact_path)
        logger.info(f"Logged artifacts: {local_dir} → {artifact_path or 'root'}")

    def log_figure(
        self,
        figure: Any,
        artifact_name: str,
    ) -> None:
        """
        Log a matplotlib/plotly figure.

        Args:
            figure: Matplotlib figure or Plotly figure
            artifact_name: Output filename
        """
        try:
            mlflow.log_figure(figure, artifact_name)
            logger.debug(f"Logged figure: {artifact_name}")
        except Exception as e:
            logger.warning(f"Could not log figure {artifact_name}: {e}")

    def log_dict(
        self,
        dictionary: Dict[str, Any],
        artifact_file: str,
    ) -> None:
        """Log a dictionary as a JSON file."""
        mlflow.log_dict(dictionary, artifact_file)
        logger.debug(f"Logged dict: {artifact_file}")

    def log_text(
        self,
        text: str,
        artifact_file: str,
    ) -> None:
        """Log text as a file."""
        mlflow.log_text(text, artifact_file)
        logger.debug(f"Logged text: {artifact_file}")

    # ── Model logging ─────────────────────────────────────────────────────────

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        conda_env: Optional[Dict[str, Any]] = None,
        code_paths: Optional[List[str]] = None,
        registered_model_name: Optional[str] = None,
        await_registration_for: int = 0,
        **kwargs,
    ) -> Any:
        """
        Log a model to MLflow.

        Supports sklearn, pytorch, xgboost, lightgbm models automatically.

        Args:
            model: The model to log
            artifact_path: Path within MLflow artifacts
            conda_env: Conda environment specification
            code_paths: Paths to include in the model
            registered_model_name: If set, register in model registry
            await_registration_for: Seconds to wait for model registration
            **kwargs: Additional args passed to mlflow.<framework>.log_model()

        Returns:
            MLflow model URI
        """
        # Auto-detect framework
        framework = self._detect_framework(model)
        logger.info(f"Logging {framework} model to {artifact_path}")

        if framework == "pytorch":
            model_uri = mlflow.pytorch.log_model(
                model, artifact_path, **kwargs
            )
        elif framework == "sklearn":
            model_uri = mlflow.sklearn.log_model(
                model, artifact_path, **kwargs
            )
        elif framework == "xgboost":
            model_uri = mlflow.xgboost.log_model(
                model, artifact_path, **kwargs
            )
        elif framework == "lightgbm":
            model_uri = mlflow.lightgbm.log_model(
                model, artifact_path, **kwargs
            )
        else:
            # Generic fallback
            model_uri = mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=model,
                **kwargs,
            )

        # Register if requested
        if registered_model_name:
            model_uri = mlflow.register_model(model_uri.model_uri, registered_model_name)
            if await_registration_for > 0:
                client = MlflowClient(self.tracking_uri)
                client.wait_for_model_version(
                    registered_model_name,
                    model_uri.version,
                    timeout=await_registration_for,
                )

        return model_uri

    def _detect_framework(self, model: Any) -> str:
        """Auto-detect the ML framework."""
        class_name = type(model).__name__.lower()
        module = type(model).__module__.split(".")[0]

        if module == "torch" or "lstm" in class_name or "nn" in class_name:
            return "pytorch"
        if hasattr(model, "predict_proba") and hasattr(model, "fit"):
            if hasattr(model, "feature_importances_"):
                if "xgb" in class_name:
                    return "xgboost"
                if "lgb" in class_name:
                    return "lightgbm"
            return "sklearn"
        return "unknown"

    # ── Model registry ────────────────────────────────────────────────────────

    def register_model(
        self,
        model_name: str,
        model_uri: str,
        description: Optional[str] = None,
        await_registration_for: int = 300,
    ) -> Any:
        """
        Register a model in the MLflow Model Registry.

        Args:
            model_name: Name for the registered model
            model_uri: URI of the logged model (e.g., "runs:/...")
            description: Model description
            await_registration_for: Seconds to wait for registration

        Returns:
            ModelVersion object
        """
        mv = mlflow.register_model(model_uri, model_name)
        logger.info(f"Registered model '{model_name}' version {mv.version}")

        if description:
            client = MlflowClient(self.tracking_uri)
            client.update_model_version(
                name=model_name,
                version=mv.version,
                description=description,
            )

        return mv

    def transition_model_stage(
        self,
        model_name: str,
        version: int,
        stage: str,
    ) -> None:
        """
        Transition a model version to a new stage.

        Args:
            model_name: Registered model name
            version: Model version number
            stage: Target stage (Staging, Production, Archived)
        """
        client = MlflowClient(self.tracking_uri)
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
        )
        logger.info(f"Model '{model_name}' v{version} → {stage}")

    def get_latest_model_version(self, model_name: str, stage: Optional[str] = None) -> Optional[int]:
        """Get the latest version number for a registered model."""
        client = MlflowClient(self.tracking_uri)
        versions = client.get_latest_versions(model_name, stage=stage)
        if versions:
            return versions[0].version
        return None

    def list_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """List all versions of a registered model."""
        client = MlflowClient(self.tracking_uri)
        mvs = client.search_model_versions(f"name='{model_name}'")
        return [
            {
                "name": mv.name,
                "version": mv.version,
                "stage": mv.current_stage,
                "status": mv.status,
                "run_id": mv.run_id,
                "creation_timestamp": mv.creation_timestamp,
            }
            for mv in mvs
        ]

    # ── Run querying ────────────────────────────────────────────────────────

    def search_runs(
        self,
        experiment_ids: Optional[List[str]] = None,
        filter_str: Optional[str] = None,
        max_results: int = 100,
        order_by: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for runs.

        Args:
            experiment_ids: List of experiment IDs (default: current experiment)
            filter_str: MLflow filter string (e.g., "metrics.val_loss < 0.5")
            max_results: Maximum number of runs to return
            order_by: Sort order (e.g., ["metrics.val_loss ASC"])

        Returns:
            List of run dictionaries
        """
        runs = mlflow.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_str,
            max_results=max_results,
            order_by=order_by,
        )
        return runs.to_dict("records")

    def get_run(self, run_id: str) -> Dict[str, Any]:
        """Get run info by ID."""
        run = mlflow.get_run(run_id)
        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
        }

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def delete_experiment(self, experiment_name: str) -> None:
        """Delete an experiment and all its runs."""
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp:
            client = MlflowClient(self.tracking_uri)
            client.delete_experiment(exp.experiment_id)
            logger.info(f"Deleted experiment: {experiment_name}")

    def clear_artifacts(self, local_dir: Optional[str] = None) -> None:
        """Remove local artifact directory."""
        dir_path = Path(local_dir or self.artifact_root)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            logger.info(f"Cleared artifacts: {dir_path}")

    def summary(self) -> str:
        """Return a string summary."""
        exps = self.list_experiments()
        return (
            f"MLflowTracker(\n"
            f"  tracking_uri={self.tracking_uri},\n"
            f"  artifact_root={self.artifact_root},\n"
            f"  experiments={len(exps)}\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience decorators
# ─────────────────────────────────────────────────────────────────────────────


def mlflow_autolog(
    experiment_name: str,
    model_name: Optional[str] = None,
    tracking_uri: str = "sqlite:///mlflow.db",
):
    """
    Decorator to automatically log training runs.

    Usage:
        @mlflow_autolog("lstm-training", "lstm-ew")
        def train_lstm(X_train, y_train):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracker = MLflowTracker(tracking_uri=tracking_uri, experiment_name=experiment_name)
            tracker.start_run(run_name=func.__name__)

            try:
                result = func(*args, **kwargs)
                tracker.end_run("FINISHED")
                return result
            except Exception as e:
                tracker.end_run("FAILED")
                raise

        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# Setup helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_local_mlflow(
    db_path: str = "sqlite:///mlflow.db",
    artifact_path: str = "./mlruns",
    create_dirs: bool = True,
) -> MLflowTracker:
    """
    Set up MLflow with local SQLite backend.

    Args:
        db_path: Path to SQLite database
        artifact_path: Local artifact directory
        create_dirs: Create directories if they don't exist

    Returns:
        Configured MLflowTracker
    """
    if create_dirs:
        Path(artifact_path).mkdir(parents=True, exist_ok=True)

    tracker = MLflowTracker(
        tracking_uri=db_path,
        artifact_root=artifact_path,
    )

    logger.info(
        "MLflow local server ready.\n"
        f"  Tracking URI: {db_path}\n"
        f"  Artifacts: {artifact_path}\n"
        "  Start UI: mlflow server --host 0.0.0.0 --port 5000\n"
        "  Or: cd to project dir and run `mlflow ui`"
    )

    return tracker


def setup_remote_mlflow(tracking_uri: str, artifact_root: str) -> MLflowTracker:
    """
    Set up MLflow with a remote PostgreSQL + S3 backend.

    Args:
        tracking_uri: PostgreSQL connection string
        artifact_root: S3 bucket path (e.g., s3://bucket/artifacts)

    Returns:
        Configured MLflowTracker
    """
    tracker = MLflowTracker(
        tracking_uri=tracking_uri,
        artifact_root=artifact_root,
    )

    logger.info(f"MLflow remote server connected: {tracking_uri}")
    return tracker


# ─────────────────────────────────────────────────────────────────────────────
# Module exports
# ─────────────────────────────────────────────────────────────────────────────
__all__ = [
    "MLflowTracker",
    "MLflowClient",
    "mlflow_autolog",
    "setup_local_mlflow",
    "setup_remote_mlflow",
]
