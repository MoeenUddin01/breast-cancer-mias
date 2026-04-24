"""Training loop for PyTorch breast cancer detection models.

Provides the main training function that orchestrates model training,
validation, early stopping, and metric logging.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.training.callbacks import EarlyStopping
from src.training.validator import validate

if TYPE_CHECKING:
    from types import SimpleNamespace


ARTIFACTS_DIR = Path(
    "/home/moeenuddin/Desktop/Deep_learning/breast-cancer-mias/artifacts"
)


def _save_model_and_results(
    model: nn.Module,
    model_name: str,
    history: dict[str, list[float]],
    config: SimpleNamespace,
) -> None:
    """Save the final model and training results to the artifacts directory.

    Creates a model-specific folder containing:
        - {model_name}_final.pth: Final model checkpoint.
        - results.json: Training metrics and configuration.

    Args:
        model: The trained model to save.
        model_name: Name identifier for the model.
        history: Dictionary containing training history metrics.
        config: Configuration object with training parameters.

    """
    model_dir = ARTIFACTS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save final model
    model_path = model_dir / f"{model_name}_final.pth"
    torch.save(model.state_dict(), model_path)

    # Compute final metrics
    final_epoch = len(history["train_loss"])
    results = {
        "model_name": model_name,
        "epochs_trained": final_epoch,
        "final_train_loss": history["train_loss"][-1]
        if history["train_loss"]
        else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "final_val_accuracy": history["val_accuracy"][-1]
        if history["val_accuracy"]
        else None,
        "final_val_auc": history["val_auc"][-1] if history["val_auc"] else None,
        "best_val_auc": max(history["val_auc"]) if history["val_auc"] else None,
        "best_val_accuracy": max(history["val_accuracy"])
        if history["val_accuracy"]
        else None,
        "config": {
            "learning_rate": getattr(config, "LEARNING_RATE", None),
            "epochs": getattr(config, "EPOCHS", None),
            "patience": getattr(config, "PATIENCE", None),
            "device": str(getattr(config, "DEVICE", None)),
        },
        "history": history,
    }

    # Save results
    results_path = model_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Model saved to {model_path}")
    print(f"Results saved to {results_path}")


def _compute_pos_weight(train_loader: DataLoader, device: torch.device) -> torch.Tensor:
    """Compute positive weight for BCEWithLogitsLoss from training labels.

    Calculates pos_weight = num_negatives / num_positives to handle class imbalance.

    Args:
        train_loader: DataLoader providing training batches.
        device: Device to place the weight tensor on.

    Returns:
        torch.Tensor: Scalar tensor containing the positive weight.

    """
    num_positives = 0
    num_negatives = 0

    for _, labels in train_loader:
        labels_int = labels.int()
        num_positives += labels_int.sum().item()
        num_negatives += (labels_int == 0).sum().item()

    if num_positives == 0:
        pos_weight = 1.0
    else:
        pos_weight = num_negatives / num_positives

    return torch.tensor([pos_weight], dtype=torch.float32, device=device)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_name: str,
    config: SimpleNamespace,
) -> dict[str, list[float]]:
    """Train a model with validation and early stopping.

    Uses Adam optimizer and BCEWithLogitsLoss with class weighting.
    Validates after each epoch and applies early stopping on validation AUC.

    Args:
        model: The neural network model to train.
        train_loader: DataLoader providing training batches.
        val_loader: DataLoader providing validation batches.
        model_name: Name identifier for the model (used for checkpointing).
        config: Configuration object with attributes:
            - LEARNING_RATE: float, optimizer learning rate.
            - EPOCHS: int, maximum number of training epochs.
            - PATIENCE: int, epochs to wait before early stopping.
            - DEVICE: torch.device, device to run training on.

    Returns:
        Dictionary containing per-epoch history lists:
            - "train_loss": List of average training losses.
            - "val_loss": List of average validation losses.
            - "val_accuracy": List of validation accuracies.
            - "val_auc": List of validation AUC scores.

    Raises:
        TypeError: If model is not an nn.Module.
        TypeError: If train_loader or val_loader is not a DataLoader.
        TypeError: If model_name is not a string.
        AttributeError: If config is missing required attributes.

    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if not isinstance(train_loader, DataLoader):
        raise TypeError(
            f"train_loader must be a DataLoader, got {type(train_loader).__name__}"
        )
    if not isinstance(val_loader, DataLoader):
        raise TypeError(
            f"val_loader must be a DataLoader, got {type(val_loader).__name__}"
        )
    if not isinstance(model_name, str):
        raise TypeError(f"model_name must be a string, got {type(model_name).__name__}")

    required_attrs = ["LEARNING_RATE", "EPOCHS", "PATIENCE", "DEVICE"]
    for attr in required_attrs:
        if not hasattr(config, attr):
            raise AttributeError(f"config missing required attribute: {attr}")

    device = config.DEVICE
    model.to(device)

    # Compute positive weight and initialize loss
    pos_weight = _compute_pos_weight(train_loader, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)

    early_stopping = EarlyStopping(
        patience=config.PATIENCE,
        mode="max",
        save_path=str(ARTIFACTS_DIR / model_name),
    )

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_auc": [],
    }

    for epoch in range(1, config.EPOCHS + 1):
        # Training phase
        model.train()
        epoch_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation phase
        val_loss, val_accuracy, val_auc = validate(model, val_loader, criterion, device)

        # Log metrics
        print(
            f"Epoch {epoch}/{config.EPOCHS} - "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_accuracy:.2f}%, "
            f"Val AUC: {val_auc:.4f}"
        )

        # Store history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_auc"].append(val_auc)

        # Early stopping check
        if early_stopping.step(val_auc, model, model_name):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # Save final model and results
    _save_model_and_results(model, model_name, history, config)

    return history
