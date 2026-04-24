"""Training callbacks for PyTorch model training.

Provides utility classes for monitoring training progress and implementing
training control mechanisms such as early stopping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import nn


class EarlyStopping:
    """Monitor a metric and stop training if no improvement after patience epochs.

    Saves model checkpoints when the monitored metric improves. Supports both
    minimization (e.g., loss) and maximization (e.g., accuracy) modes.

    Attributes:
        patience: Number of epochs to wait for improvement before stopping.
        mode: Whether to minimize or maximize the monitored metric.
        save_path: Directory path where checkpoint files are saved.
        counter: Current count of epochs without improvement.
        best_score: Best metric value seen so far.
    """

    def __init__(
        self,
        patience: int,
        mode: Literal["min", "max"] = "min",
        save_path: str = "outputs/models",
    ) -> None:
        """Initialize the EarlyStopping callback.

        Args:
            patience: Number of epochs to wait for improvement before stopping.
            mode: Direction of improvement - "min" for metrics to minimize
                (e.g., loss), "max" for metrics to maximize (e.g., accuracy).
            save_path: Directory path for saving model checkpoints.

        Raises:
            ValueError: If patience is not positive.
            ValueError: If mode is not "min" or "max".

        """
        if patience <= 0:
            raise ValueError(f"patience must be positive, got {patience}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        self.patience = patience
        self.mode = mode
        self.save_path = Path(save_path)
        self.counter = 0
        self.best_score: float | None = None
        self._is_better = self._get_comparison_fn()

    def _get_comparison_fn(self) -> callable:
        """Return the appropriate comparison function based on mode.

        Returns:
            Callable that takes (current, best) and returns True if current
            is better than best according to the mode.

        """
        if self.mode == "min":
            return lambda current, best: current < best
        return lambda current, best: current > best

    def step(
        self,
        metric: float,
        model: nn.Module,
        model_name: str,
    ) -> bool:
        """Process one epoch's metric and determine if training should stop.

        Saves a checkpoint when the metric improves and increments the
        patience counter when it does not.

        Args:
            metric: The metric value for the current epoch.
            model: The model to save if the metric improves.
            model_name: Base name for the checkpoint file (e.g., "resnet").

        Returns:
            bool: True if training should stop (no improvement for patience
                epochs), False otherwise.

        Raises:
            TypeError: If model is not an nn.Module.

        """
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")

        if self.best_score is None:
            self.best_score = metric
            self._save_checkpoint(model, model_name)
            return False

        if self._is_better(metric, self.best_score):
            self.best_score = metric
            self.counter = 0
            self._save_checkpoint(model, model_name)
        else:
            self.counter += 1

        return self.counter >= self.patience

    def _save_checkpoint(self, model: nn.Module, model_name: str) -> None:
        """Save the model state dict to a checkpoint file.

        Args:
            model: The model whose state should be saved.
            model_name: Base name for the checkpoint file.

        """
        self.save_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.save_path / f"{model_name}_best.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"EarlyStopping: Saved new best model to {checkpoint_path}")
