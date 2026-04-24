"""Validation utilities for PyTorch model training.

Provides functions for evaluating model performance on validation/test sets,
computing loss, accuracy, and AUC metrics.
"""

from __future__ import annotations

import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate a model on a validation dataset.

    Computes average loss, accuracy, and AUC over all batches.
    Uses sigmoid activation on logits before thresholding at 0.5 for accuracy.

    Args:
        model: The neural network model to evaluate.
        dataloader: DataLoader providing validation batches of (images, labels).
        criterion: Loss function (e.g., BCEWithLogitsLoss).
        device: Device to run evaluation on (cpu or cuda).

    Returns:
        Tuple containing:
            - float: Average validation loss.
            - float: Validation accuracy (percentage).
            - float: Validation AUC score.

    Raises:
        TypeError: If model is not an nn.Module.
        TypeError: If dataloader is not a DataLoader.
        TypeError: If criterion is not an nn.Module.
        TypeError: If device is not a torch.device.

    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if not isinstance(dataloader, DataLoader):
        raise TypeError(
            f"dataloader must be a DataLoader, got {type(dataloader).__name__}"
        )
    if not isinstance(criterion, nn.Module):
        raise TypeError(
            f"criterion must be an nn.Module, got {type(criterion).__name__}"
        )
    if not isinstance(device, torch.device):
        raise TypeError(f"device must be a torch.device, got {type(device).__name__}")

    model.eval()

    total_loss = 0.0
    all_predictions: list[float] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))
            total_loss += loss.item()

            probs = torch.sigmoid(outputs).squeeze()
            all_predictions.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().astype(int).tolist())

    num_batches = len(dataloader)
    val_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # Accuracy: threshold at 0.5
    predicted_labels = [1 if p >= 0.5 else 0 for p in all_predictions]
    correct = sum(1 for pred, true in zip(predicted_labels, all_labels) if pred == true)
    val_accuracy = 100.0 * correct / len(all_labels) if all_labels else 0.0

    # AUC
    if len(set(all_labels)) > 1:
        val_auc = roc_auc_score(all_labels, all_predictions)
    else:
        val_auc = 0.0

    return val_loss, val_accuracy, val_auc
