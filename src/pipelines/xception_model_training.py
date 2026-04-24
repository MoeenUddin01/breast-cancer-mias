"""Xception model training pipeline for MIAS breast cancer detection.

Orchestrates loading the Xception model, training, evaluation, and visualization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from src.evaluation.evaluator import evaluate
from src.evaluation.visualizer import plot_confusion_matrix, plot_training_history
from src.models.xception_model import get_xception_model
from src.training.trainer import train

if TYPE_CHECKING:
    from types import SimpleNamespace

    from torch.utils.data import DataLoader


def _collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Collect true labels and predictions from a dataloader.

    Args:
        model: The trained model.
        dataloader: DataLoader providing batches.
        device: Device to run inference on.

    Returns:
        Tuple containing:
            - y_true: List of true labels.
            - y_pred: List of predicted labels.

    """
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs >= 0.5).int()

            y_true.extend(labels.cpu().numpy().astype(int).tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    return y_true, y_pred


def run_xception_pipeline(
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: SimpleNamespace,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Run the complete Xception training pipeline.

    Loads the Xception model, trains it on the training data, evaluates on
    the test data, generates visualizations, and returns metrics and history.

    Args:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for test data (used as validation).
        config: Configuration object containing:
            - DEVICE: torch.device to run training on.
            - LEARNING_RATE: float, optimizer learning rate.
            - EPOCHS: int, maximum number of training epochs.
            - PATIENCE: int, epochs to wait before early stopping.

    Returns:
        Tuple containing:
            - metrics: Dictionary of evaluation metrics from evaluator.
            - history: Dictionary of training history per epoch.

    Raises:
        AttributeError: If config is missing required attributes.
        RuntimeError: If model loading or training fails.

    """
    model_name = "xception"

    # Validate config has required attributes
    required_attrs = ["DEVICE", "LEARNING_RATE", "EPOCHS", "PATIENCE"]
    for attr in required_attrs:
        if not hasattr(config, attr):
            raise AttributeError(f"config missing required attribute: {attr}")

    # Load Xception model
    try:
        model = get_xception_model()
    except ImportError as e:
        raise RuntimeError(f"Failed to import timm for Xception model: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load Xception model: {e}") from e

    # Move model to device
    try:
        device = config.DEVICE
        model.to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to move model to device {config.DEVICE}: {e}") from e

    # Train the model (use test_loader as validation)
    try:
        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            model_name=model_name,
            config=config,
        )
    except Exception as e:
        raise RuntimeError(f"Training failed for {model_name}: {e}") from e

    # Evaluate the model on test set
    try:
        metrics = evaluate(
            model=model,
            dataloader=test_loader,
            device=device,
            model_name=model_name,
        )
    except Exception as e:
        raise RuntimeError(f"Evaluation failed for {model_name}: {e}") from e

    # Plot training history
    try:
        plot_training_history(history, model_name)
    except Exception as e:
        print(f"[WARNING] Failed to plot training history: {e}")

    # Collect predictions for confusion matrix
    try:
        y_true, y_pred = _collect_predictions(model, test_loader, device)
    except Exception as e:
        raise RuntimeError(f"Failed to collect predictions for confusion matrix: {e}") from e

    # Plot confusion matrix
    try:
        plot_confusion_matrix(y_true, y_pred, model_name)
    except Exception as e:
        print(f"[WARNING] Failed to plot confusion matrix: {e}")

    return metrics, history
