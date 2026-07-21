"""
LSTM Early Warning Model for temporal pattern recognition in student engagement.

Architecture:
    Input: (batch_size=32, seq_len=12, features=14)
        LSTM(14 → 64, return_sequences=True, dropout=0.3)
        LSTM(64 → 32, return_sequences=False, dropout=0.3)
        Dense(32 → 16, ReLU, dropout=0.3)
        Dense(16 → 1, sigmoid)
    Output: dropout/failure probability
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import shap
except ImportError:
    shap = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMEarlyWarning(nn.Module):
    """
    LSTM model for early warning / dropout prediction.

    Uses bidirectional LSTM layers with dropout regularization,
    early stopping, and SHAP-based explainability.
    """

    def __init__(
        self,
        input_dim: int = 14,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        """
        Initialize the LSTM model.

        Args:
            input_dim: Number of input features (default: 14 engagement metrics)
            hidden_dim: Hidden state dimension (default: 64)
            n_layers: Number of LSTM layers (default: 2)
            dropout: Dropout probability (default: 0.3)
            bidirectional: Use bidirectional LSTM (default: False)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout
        self.bidirectional = bidirectional

        # Single LSTM with multiple layers — dropout is applied between layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_output_size = hidden_dim * 2 if bidirectional else hidden_dim

        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_size, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch_size, 1) — dropout/failure probability
        """
        # Single stacked LSTM — returns only last hidden state
        _, (hidden_n, _) = self.lstm(x)
        # hidden_n shape: (num_directions * num_layers, batch, hidden_dim)
        # Take the last layer's hidden state
        last_hidden = hidden_n[-1]  # (batch, hidden_dim)
        if self.bidirectional:
            # Concatenate forward and backward from last layer
            last_hidden = hidden_n.view(self.n_layers, 2, -1, self.hidden_dim)[-1]
            last_hidden = last_hidden.permute(1, 0, 2).contiguous().view(-1)
        else:
            last_hidden = hidden_n[-1]

        # Dense layers
        x = self.fc1(last_hidden)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_epochs: int = 50,
        early_stopping_patience: int = 10,
        batch_size: int = 32,
        lr: float = 0.001,
        checkpoint_dir: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the LSTM model with early stopping.

        Args:
            X_train: Training data (batch_size, seq_len, features)
            y_train: Training labels (batch_size,)
            X_val: Validation data
            y_val: Validation labels
            n_epochs: Maximum number of epochs
            early_stopping_patience: Epochs to wait before early stopping
            batch_size: Batch size for training
            lr: Learning rate
            checkpoint_dir: Directory to save model checkpoints
            device: Device to train on ('cuda' or 'cpu')

        Returns:
            Dictionary with training history (loss, val_loss, etc.)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(device)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state: Optional[Dict[str, Any]] = None

        for epoch in range(n_epochs):
            self.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_X.size(0)
                preds = (outputs >= 0.5).float()
                train_correct += (preds == batch_y).sum().item()
                train_total += batch_X.size(0)

            avg_train_loss = train_loss / train_total
            train_acc = train_correct / train_total

            # Validation
            self.eval()
            with torch.no_grad():
                val_outputs = self.forward(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
                val_preds = (val_outputs >= 0.5).float()
                val_acc = (val_preds == y_val_t).sum().item() / len(y_val_t)

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            logger.info(
                f"Epoch {epoch+1}/{n_epochs} — "
                f"train_loss: {avg_train_loss:.4f}, val_loss: {val_loss:.4f}, "
                f"train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                logger.info(f"  ↳ New best model (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch+1} "
                        f"(patience={early_stopping_patience})"
                    )
                    break

        # Restore best model
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            self.to(device)

        # Save checkpoint if directory provided
        if checkpoint_dir is not None:
            self.save_checkpoint(checkpoint_dir, best_val_loss)

        return history

    def predict(self, X: np.ndarray, device: Optional[str] = None) -> np.ndarray:
        """
        Run inference on input data.

        Args:
            X: Input data (batch_size, seq_len, features)
            device: Device to use for inference

        Returns:
            Array of probabilities (batch_size,)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        self.eval()

        X_t = torch.FloatTensor(X).to(device)

        with torch.no_grad():
            outputs = self.forward(X_t)

        return outputs.cpu().numpy().squeeze()

    def predict_with_uncertainty(
        self, X: np.ndarray, n_samples: int = 30, device: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference with Monte Carlo dropout for uncertainty estimation.

        Args:
            X: Input data
            n_samples: Number of forward passes with dropout enabled
            device: Device to use

        Returns:
            Tuple of (mean_probabilities, std_deviations)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        self.train()  # Enable dropout

        X_t = torch.FloatTensor(X).to(device)
        predictions: List[np.ndarray] = []

        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(X_t).cpu().numpy().squeeze()
                predictions.append(out)

        predictions = np.array(predictions)
        mean_prob = predictions.mean(axis=0)
        std_prob = predictions.std(axis=0)

        return mean_prob, std_prob

    def get_shap_values(
        self,
        X_sample: np.ndarray,
        background_size: int = 100,
        device: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        Compute SHAP values for model explainability.

        Args:
            X_sample: Sample input data to explain (N, seq_len, features)
            background_size: Number of background samples for kernel Explainer
            device: Device to use

        Returns:
            SHAP values array (N, seq_len, features) or None if shap not available
        """
        if shap is None:
            logger.warning("SHAP not installed. Run: pip install shap")
            return None

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        import torch.nn.functional as F

        # Create a wrapper that returns numpy output
        def model_forward(x: np.ndarray) -> np.ndarray:
            x_t = torch.FloatTensor(x).to(device)
            with torch.no_grad():
                out = self.forward(x_t).cpu().numpy().squeeze()
            if out.ndim == 0:
                out = out.reshape(1)
            return out

        # Use a subset as background
        n_samples = X_sample.shape[0]
        bg_size = min(background_size, n_samples)
        bg_indices = np.random.choice(n_samples, bg_size, replace=False)
        background = X_sample[bg_indices]

        # Wrap in PyTorch for compatibility
        explainer = shap.KernelExplainer(model_forward, background)
        shap_values = explainer.shap_values(X_sample)

        logger.info(f"Computed SHAP values for {n_samples} samples")
        return shap_values

    def save_checkpoint(self, checkpoint_dir: str, best_loss: float) -> Path:
        """
        Save model checkpoint to disk.

        Args:
            checkpoint_dir: Directory to save checkpoint
            best_loss: Best validation loss achieved

        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / "lstm_early_warning_best.pt"

        torch.save(
            {
                'model_state_dict': self.state_dict(),
                'config': {
                    'input_dim': self.input_dim,
                    'hidden_dim': self.hidden_dim,
                    'n_layers': self.n_layers,
                    'dropout': self.dropout_rate,
                    'bidirectional': self.bidirectional,
                },
                'best_val_loss': best_loss,
            },
            checkpoint_path,
        )

        logger.info(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str, device: Optional[str] = None) -> "LSTMEarlyWarning":
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model onto

        Returns:
            Loaded LSTMEarlyWarning model
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint['config']

        model = cls(
            input_dim=config['input_dim'],
            hidden_dim=config['hidden_dim'],
            n_layers=config['n_layers'],
            dropout=config['dropout'],
            bidirectional=config['bidirectional'],
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        logger.info(f"Model loaded from {checkpoint_path} (val_loss: {checkpoint['best_val_loss']:.4f})")
        return model

    def get_feature_importance_summary(
        self,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Get a simple feature importance summary based on input layer weights.

        This is an approximation — use SHAP for proper importance.

        Args:
            feature_names: Optional list of feature names

        Returns:
            Dictionary mapping feature names to importance scores
        """
        # Use absolute mean of input weight matrix
        weights = self.lstm.weight_ih_l0.detach().cpu().numpy()
        importance = np.abs(weights).mean(axis=0)

        if feature_names is not None and len(feature_names) == len(importance):
            return dict(zip(feature_names, importance.tolist()))
        else:
            return {f"feature_{i}": float(v) for i, v in enumerate(importance)}

    def summary(self) -> str:
        """Return a summary string of the model architecture."""
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return (
            f"LSTMEarlyWarning(\n"
            f"  input_dim={self.input_dim},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  n_layers={self.n_layers},\n"
            f"  dropout={self.dropout_rate},\n"
            f"  bidirectional={self.bidirectional},\n"
            f"  total_params={n_params:,},\n"
            f"  trainable_params={n_trainable:,}\n"
            f")"
        )


def generate_synthetic_data(
    n_samples: int = 1000,
    seq_len: int = 12,
    n_features: int = 14,
    dropout_rate: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic student engagement data for testing.

    Args:
        n_samples: Number of student sequences
        seq_len: Sequence length (weeks of data)
        n_features: Number of features per timestep
        dropout_rate: Fraction of positive (dropout) labels
        seed: Random seed

    Returns:
        Tuple of (X, y) arrays
    """
    np.random.seed(seed)

    # Generate sequences with temporal patterns
    t = np.linspace(0, 2 * np.pi, seq_len)
    X = np.zeros((n_samples, seq_len, n_features))

    for i in range(n_samples):
        trend = np.random.randn() * 0.1
        for j in range(seq_len):
            X[i, j, :] = (
                np.sin(t[j] + np.random.randn() * 0.5) * 0.3
                + np.random.randn(n_features) * 0.2
                + trend * j
            )

    # Labels: students with declining engagement are more likely to drop out
    engagement_trends = np.diff(X[:, :, 0], axis=1).mean(axis=1)
    probs = 1 / (1 + np.exp(-(engagement_trends * 5 + np.random.randn(n_samples) * 0.5)))
    probs = np.clip(probs, 0, 1)

    # Force roughly dropout_rate fraction to be positive
    threshold = np.percentile(probs, (1 - dropout_rate) * 100)
    y = (probs >= threshold).astype(float)

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Module-level import alias for clean imports
# ─────────────────────────────────────────────────────────────────────────────
__all__ = ["LSTMEarlyWarning", "generate_synthetic_data"]
