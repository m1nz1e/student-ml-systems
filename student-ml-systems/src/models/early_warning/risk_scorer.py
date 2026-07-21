"""
Risk Scorer — converts model outputs to interpretable 0-100 risk scores.

Provides:
    - Risk score normalization (0-100)
    - 5-tier stratification (Low/Medium/High/Very High/Critical)
    - SHAP-based explanations
    - Lead time calculation
    - Alert generation for tutors
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None
    warnings.warn("SHAP not installed. Run: pip install shap for explanations.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Risk tiers
# ─────────────────────────────────────────────────────────────────────────────

# (low, high] — upper bound inclusive
RISK_TIERS = {
    "LOW": (0.0, 20.0),
    "MEDIUM": (20.0, 40.0),
    "HIGH": (40.0, 60.0),
    "VERY_HIGH": (60.0, 80.0),
    "CRITICAL": (80.0, 100.0),
}

RISK_COLORS = {
    "LOW": "#22c55e",       # Green
    "MEDIUM": "#eab308",     # Yellow
    "HIGH": "#f97316",       # Orange
    "VERY_HIGH": "#ef4444",  # Red
    "CRITICAL": "#a855f7",    # Purple
}

RISK_EMOJI = {
    "LOW": "🟢",
    "MEDIUM": "🟡",
    "HIGH": "🟠",
    "VERY_HIGH": "🔴",
    "CRITICAL": "🟣",
}

# Recommended interventions per tier
TIER_INTERVENTIONS: Dict[str, List[str]] = {
    "LOW": ["Continue monitoring"],
    "MEDIUM": ["Send engagement check-in email", "Assign peer mentor"],
    "HIGH": ["Schedule one-to-one meeting", "Send warning letter"],
    "VERY_HIGH": ["Urgent导师 meeting", "Involve student welfare services"],
    "CRITICAL": ["Immediate welfare check", "Escalate to head of year"],
}


@dataclass
class RiskAlert:
    """Structured alert for a student at risk."""
    student_id: str
    risk_score: float
    tier: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    explanation: str = ""
    top_factors: List[Tuple[str, float]] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    shap_values: Optional[np.ndarray] = None
    model_name: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "student_id": self.student_id,
            "risk_score": round(self.risk_score, 1),
            "tier": self.tier,
            "emoji": RISK_EMOJI.get(self.tier, "❓"),
            "timestamp": self.timestamp,
            "explanation": self.explanation,
            "top_factors": [(k, round(v, 4)) for k, v in self.top_factors],
            "recommended_actions": self.recommended_actions,
            "model_name": self.model_name,
        }

    def to_display_string(self) -> str:
        emoji = RISK_EMOJI.get(self.tier, "❓")
        lines = [
            f"{emoji} Student: {self.student_id}",
            f"   Risk Score: {self.risk_score:.1f}/100 [{self.tier}]",
            f"   Time: {self.timestamp}",
            f"   Explanation: {self.explanation}",
        ]
        if self.top_factors:
            lines.append(f"   Top risk factors:")
            for factor, val in self.top_factors[:5]:
                lines.append(f"     • {factor}: {val:+.4f}")
        if self.recommended_actions:
            lines.append(f"   Actions: {', '.join(self.recommended_actions[:2])}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────


class RiskScorer:
    """
    Converts raw model outputs to interpretable 0-100 risk scores.

    Wraps a trained model and provides:
        - Score normalization
        - Tier stratification
        - SHAP explanations
        - Alert generation
        - Lead time estimation
    """

    def __init__(
        self,
        model: Any,
        thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
        feature_names: Optional[List[str]] = None,
        model_name: str = "model",
        shap_background: Optional[np.ndarray] = None,
        shap_background_size: int = 100,
    ):
        """
        Initialize the RiskScorer.

        Args:
            model: Trained model with a predict() or predict_proba() method
            thresholds: Custom risk tier thresholds (default: RISK_TIERS)
            feature_names: List of feature names for explanations
            model_name: Human-readable name for this model
            shap_background: Background samples for SHAP KernelExplainer
            shap_background_size: Max background samples to use
        """
        self.model = model
        self.thresholds = thresholds or RISK_TIERS
        self.feature_names = feature_names or []
        self.model_name = model_name
        self.shap_background = shap_background
        self.shap_background_size = shap_background_size
        self._shap_explainer: Optional[Any] = None

        # Fit a scaler on first predict call if needed
        self._calibrator_fitted = False

        logger.info(f"RiskScorer initialized for model: {model_name}")

    # ── Core prediction ────────────────────────────────────────────────────────

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """
        Convert model output to 0-100 risk scores.

        Args:
            X: Input features

        Returns:
            Array of risk scores in range [0, 100]
        """
        # Get raw model predictions
        raw = self._get_raw_predictions(X)

        # Normalize to 0-1 then scale to 0-100
        raw_min, raw_max = raw.min(), raw.max()
        if raw_max - raw_min > 1e-8:
            normalized = (raw - raw_min) / (raw_max - raw_min)
        else:
            normalized = np.full_like(raw, 0.5)

        # Map to 0-100
        risk_scores = np.clip(normalized * 100, 0, 100).astype(float)
        return risk_scores

    def _get_raw_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Get raw predictions from the underlying model.

        Handles various model interfaces (sklearn, xgboost, lgb, torch, etc.)
        """
        # Try sklearn-style predict_proba
        if hasattr(self.model, "predict_proba"):
            try:
                proba = self.model.predict_proba(X)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    return proba[:, 1]  # Probability of positive class
                return proba.ravel()
            except Exception:
                pass

        # Try sklearn-style predict
        if hasattr(self.model, "predict"):
            try:
                preds = self.model.predict(X)
                return preds.ravel()
            except Exception:
                pass

        # Try torch model
        if hasattr(self.model, "predict"):
            try:
                preds = self.model.predict(X)
                if isinstance(preds, np.ndarray):
                    return preds.ravel()
                return np.asarray(preds).ravel()
            except Exception:
                pass

        raise ValueError(
            f"Model {self.model_name} does not have predict() or predict_proba() method. "
            "Provide a model with one of these interfaces."
        )

    def calibrate_scores(
        self,
        X_calib: np.ndarray,
        y_true: np.ndarray,
        method: str = "isotonic",
    ) -> "RiskScorer":
        """
        Calibrate risk scores using a held-out calibration set.

        Uses sklearn's CalibratedClassifierCV internally.

        Args:
            X_calib: Calibration features
            y_true: True labels
            method: 'isotonic' or 'sigmoid'

        Returns:
            self (for chaining)
        """
        from sklearn.calibration import CalibratedClassifierCV

        if not hasattr(self.model, "predict_proba"):
            logger.warning("Model does not support predict_proba — calibration skipped")
            return self

        calibrator = CalibratedClassifierCV(self.model, method=method, cv=3)
        calibrator.fit(X_calib, y_true)
        self.model = calibrator  # replace with calibrated version
        self._calibrator_fitted = True

        logger.info(f"Risk scores calibrated using {method} method")
        return self

    # ── Stratification ─────────────────────────────────────────────────────────

    def stratify(self, risk_scores: np.ndarray) -> np.ndarray:
        """
        Assign risk tiers to an array of risk scores.

        Args:
            risk_scores: Array of scores in range [0, 100]

        Returns:
            Array of tier labels
        """
        tiers = np.empty(len(risk_scores), dtype=object)
        for label, (low, high) in self.thresholds.items():
            mask = (risk_scores > low) & (risk_scores <= high)
            tiers[mask] = label

        # Edge case: score == 0
        tiers[risk_scores <= 0] = "LOW"
        # Edge case: score == 100
        tiers[risk_scores > 100] = "CRITICAL"

        return tiers

    def stratify_df(self, df) -> Any:
        """
        Add tier and color columns to a DataFrame with risk scores.

        Args:
            df: DataFrame with a 'risk_score' column

        Returns:
            DataFrame with added 'tier', 'color', 'emoji' columns
        """
        import pandas as pd

        df = df.copy()
        df["tier"] = self.stratify(df["risk_score"].values)
        df["color"] = df["tier"].map(RISK_COLORS)
        df["emoji"] = df["tier"].map(RISK_EMOJI)
        return df

    # ── Explanations ──────────────────────────────────────────────────────────

    def explain(
        self,
        X: np.ndarray,
        student_idx: Union[int, List[int]],
        feature_names: Optional[List[str]] = None,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """
        Generate SHAP-based explanations for a student or set of students.

        Args:
            X: Feature matrix
            student_idx: Index of student(s) to explain
            feature_names: Override feature names
            top_n: Number of top factors to return

        Returns:
            Dictionary with explanation data
        """
        if not SHAP_AVAILABLE:
            return {"error": "SHAP not installed. Run: pip install shap"}

        if isinstance(student_idx, int):
            student_idx = [student_idx]

        X_sample = X[student_idx]
        names = feature_names or self.feature_names

        # Lazy-build the explainer
        if self._shap_explainer is None:
            self._shap_explainer = self._build_shap_explainer(X)

        # Compute SHAP values
        shap_values = self._shap_explainer.shap_values(X_sample)

        # shap_values shape: (n_samples, n_features)
        if np.ndim(shap_values) == 3:
            # Some explainers return per-class
            shap_values = shap_values[1] if shap_values.shape[2] > 1 else shap_values[0]

        # Aggregate across samples if multiple
        if len(student_idx) > 1:
            mean_shap = np.abs(shap_values).mean(axis=0)
            mean_shap = mean_shap.tolist()
            top_indices = np.argsort(mean_shap)[::-1][:top_n]
            top_factors = [
                ((names[i] if names else f"feature_{i}"), float(mean_shap[i]))
                for i in top_indices
            ]
            explanation_text = self._generate_explanation_text(
                np.mean(self.predict_risk(X_sample)),
                top_factors,
            )
        else:
            sv = shap_values[0] if np.ndim(shap_values) == 2 else shap_values
            top_indices = np.argsort(np.abs(sv))[::-1][:top_n]
            top_factors = [
                ((names[i] if names else f"feature_{i}"), float(sv[i]))
                for i in top_indices
            ]
            risk_score = float(self.predict_risk(X_sample.reshape(1, -1))[0])
            explanation_text = self._generate_explanation_text(risk_score, top_factors)

        return {
            "risk_score": float(self.predict_risk(X_sample)[0]) if len(student_idx) == 1 else float(np.mean(self.predict_risk(X_sample))),
            "top_factors": top_factors,
            "explanation": explanation_text,
            "shap_values": shap_values.tolist() if len(student_idx) == 1 else shap_values.tolist(),
            "n_features": X.shape[1],
        }

    def _build_shap_explainer(self, X_background: np.ndarray) -> Any:
        """Build a SHAP explainer from the model."""
        # Use a subset as background
        n_bg = min(self.shap_background_size, len(X_background))
        bg_indices = np.random.choice(len(X_background), n_bg, replace=False)
        background = X_background[bg_indices]

        if hasattr(self.model, "predict_proba"):
            def model_predict(x):
                p = self.model.predict_proba(x)
                return p[:, 1] if p.ndim == 2 else p
        elif hasattr(self.model, "predict"):
            def model_predict(x):
                return self.model.predict(x)
        else:
            model_predict = self.model.predict

        explainer = shap.KernelExplainer(model_predict, background)
        return explainer

    def _generate_explanation_text(
        self,
        risk_score: float,
        top_factors: List[Tuple[str, float]],
    ) -> str:
        """Generate a human-readable explanation."""
        tier = self.stratify(np.array([risk_score]))[0]
        emoji = RISK_EMOJI.get(tier, "❓")

        lines = [f"This student has a {risk_score:.1f}/100 risk score ({tier})."]

        if top_factors:
            positive_factors = [(f, v) for f, v in top_factors if v > 0]
            negative_factors = [(f, v) for f, v in top_factors if v <= 0]

            if positive_factors:
                factors_str = ", ".join([f"{f} (+{v:.3f})" for f, v in positive_factors[:3]])
                lines.append(f"Key risk factors: {factors_str}.")

            if negative_factors:
                factors_str = ", ".join([f"{f} ({v:.3f})" for f, v in negative_factors[:3]])
                lines.append(f"Protective factors: {factors_str}.")

        return " ".join(lines)

    # ── Lead time calculation ─────────────────────────────────────────────────

    def calculate_lead_time(
        self,
        predictions: np.ndarray,
        actual_events: np.ndarray,
        event_times: np.ndarray,
        prediction_times: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Calculate average lead time: how early the model detects risk before event.

        Args:
            predictions: Risk scores at prediction time
            actual_events: Bool array (True = event occurred)
            event_times: Time from start to event/censor
            prediction_times: Optional array of prediction times (default: same for all)

        Returns:
            Dictionary with mean/median/std lead times
        """
        if prediction_times is None:
            # Assume predictions were made at a fixed point (e.g., week 8)
            # Lead time = event_time - prediction_point
            prediction_time = 8  # weeks — reasonable default
            lead_times = event_times - prediction_time
        else:
            lead_times = event_times - prediction_times

        # Only consider students who actually experienced the event
        event_mask = actual_events.astype(bool)
        lead_times_events = lead_times[event_mask]

        if len(lead_times_events) == 0:
            return {"mean_lead_weeks": 0.0, "median_lead_weeks": 0.0, "n_events": 0}

        # Only count positive lead times (correct early detection)
        positive_leads = lead_times_events[lead_times_events > 0]

        return {
            "mean_lead_weeks": float(np.mean(lead_times_events)),
            "median_lead_weeks": float(np.median(lead_times_events)),
            "std_lead_weeks": float(np.std(lead_times_events)),
            "pct_early_detection": float(len(positive_leads) / len(lead_times_events) * 100)
            if len(lead_times_events) > 0
            else 0.0,
            "n_events": int(len(lead_times_events)),
            "n_early_detections": int(len(positive_leads)),
        }

    # ── Alert generation ──────────────────────────────────────────────────────

    def generate_alert(
        self,
        student_id: str,
        risk_score: float,
        explanation: Optional[Union[str, Dict[str, Any]]] = None,
        include_shap: bool = True,
    ) -> RiskAlert:
        """
        Generate a structured alert for a student.

        Args:
            student_id: Student identifier
            risk_score: Risk score (0-100)
            explanation: Pre-computed explanation dict or text
            include_shap: Whether to compute SHAP values for alert

        Returns:
            RiskAlert dataclass
        """
        tier = self.stratify(np.array([risk_score]))[0]
        actions = TIER_INTERVENTIONS.get(tier, [])

        # Build top factors
        top_factors: List[Tuple[str, float]] = []
        explanation_text = ""

        if explanation is not None:
            if isinstance(explanation, dict):
                top_factors = explanation.get("top_factors", [])
                explanation_text = explanation.get("explanation", "")
            else:
                explanation_text = str(explanation)

        alert = RiskAlert(
            student_id=str(student_id),
            risk_score=round(float(risk_score), 1),
            tier=str(tier),
            explanation=explanation_text,
            top_factors=top_factors,
            recommended_actions=actions[:3],
            model_name=self.model_name,
        )

        return alert

    def generate_alerts_batch(
        self,
        student_ids: List[str],
        X: np.ndarray,
        risk_scores: Optional[np.ndarray] = None,
        top_k: Optional[int] = None,
        min_tier: str = "MEDIUM",
    ) -> List[RiskAlert]:
        """
        Generate alerts for a batch of students.

        Args:
            student_ids: List of student identifiers
            X: Feature matrix
            risk_scores: Pre-computed risk scores (optional)
            top_k: Only return alerts for top K highest-risk students
            min_tier: Minimum tier to generate alerts for

        Returns:
            List of RiskAlert objects, sorted by risk score descending
        """
        if risk_scores is None:
            risk_scores = self.predict_risk(X)

        tiers = self.stratify(risk_scores)

        # Filter by minimum tier
        tier_order = list(RISK_TIERS.keys())
        min_idx = tier_order.index(min_tier) if min_tier in tier_order else 0

        alerts: List[RiskAlert] = []
        for i, (sid, score, tier) in enumerate(zip(student_ids, risk_scores, tiers)):
            tier_idx = tier_order.index(tier) if tier in tier_order else 0
            if tier_idx >= min_idx:
                alert = self.generate_alert(sid, score)
                alerts.append(alert)

        # Sort by risk score descending
        alerts.sort(key=lambda a: a.risk_score, reverse=True)

        if top_k is not None:
            alerts = alerts[:top_k]

        return alerts

    # ── Summary & utilities ──────────────────────────────────────────────────

    def get_tier_summary(self, risk_scores: np.ndarray) -> Dict[str, int]:
        """
        Count students in each risk tier.

        Args:
            risk_scores: Array of risk scores

        Returns:
            Dictionary of tier → count
        """
        tiers = self.stratify(risk_scores)
        from collections import Counter
        return dict(Counter(tiers.tolist()))

    def print_tier_summary(self, risk_scores: np.ndarray) -> None:
        """Print a formatted tier summary."""
        summary = self.get_tier_summary(risk_scores)
        total = sum(summary.values())

        print(f"\n{'─'*50}")
        print(f"  Risk Tier Summary (n={total})")
        print(f"{'─'*50}")
        for tier in ["LOW", "MEDIUM", "HIGH", "VERY_HIGH", "CRITICAL"]:
            count = summary.get(tier, 0)
            pct = count / total * 100 if total > 0 else 0
            emoji = RISK_EMOJI.get(tier, "❓")
            bar = "█" * int(pct / 5)
            print(f"  {emoji} {tier:<12}: {count:>4} ({pct:>5.1f}%) {bar}")
        print(f"{'─'*50}\n")

    def summary(self) -> str:
        """Return a string summary of the scorer."""
        return (
            f"RiskScorer(\n"
            f"  model={self.model_name},\n"
            f"  thresholds={list(self.thresholds.keys())},\n"
            f"  shap_available={SHAP_AVAILABLE}\n"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module exports
# ─────────────────────────────────────────────────────────────────────────────
__all__ = [
    "RiskScorer",
    "RiskAlert",
    "RISK_TIERS",
    "RISK_COLORS",
    "RISK_EMOJI",
    "TIER_INTERVENTIONS",
]
