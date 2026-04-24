"""Visualization utilities for model evaluation and comparison.

Provides plotting functions for training history, confusion matrices,
and model performance comparisons.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_training_history(
    history: dict[str, list[float]],
    model_name: str,
) -> None:
    """Plot training and validation metrics over epochs.

    Creates a 2-subplot figure with:
        - Left: Train loss and validation loss curves.
        - Right: Validation accuracy curve.

    Args:
        history: Dictionary containing training history lists.
            Expected keys: "train_loss", "val_loss", "val_accuracy".
        model_name: Name identifier for the model (used in filename).

    Raises:
        TypeError: If history is not a dictionary.
        TypeError: If model_name is not a string.
        ValueError: If required keys are missing from history.

    """
    if not isinstance(history, dict):
        raise TypeError(f"history must be a dict, got {type(history).__name__}")
    if not isinstance(model_name, str):
        raise TypeError(f"model_name must be a string, got {type(model_name).__name__}")

    required_keys = ["train_loss", "val_loss", "val_accuracy"]
    for key in required_keys:
        if key not in history:
            raise ValueError(f"history missing required key: {key}")

    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Plot losses
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_name} - Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot validation accuracy
    axes[1].plot(epochs, history["val_accuracy"], "g-", label="Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title(f"{model_name} - Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = plots_dir / f"{model_name}_history.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Training history plot saved to: {save_path}")
    mlflow.log_artifact(str(save_path))


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    model_name: str,
) -> None:
    """Plot a confusion matrix heatmap.

    Args:
        y_true: List of true labels.
        y_pred: List of predicted labels.
        model_name: Name identifier for the model (used in filename).

    Raises:
        TypeError: If y_true or y_pred is not a list.
        TypeError: If model_name is not a string.
        ValueError: If y_true and y_pred have different lengths.

    """
    if not isinstance(y_true, list):
        raise TypeError(f"y_true must be a list, got {type(y_true).__name__}")
    if not isinstance(y_pred, list):
        raise TypeError(f"y_pred must be a list, got {type(y_pred).__name__}")
    if not isinstance(model_name, str):
        raise TypeError(f"model_name must be a string, got {type(model_name).__name__}")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length, got {len(y_true)} and {len(y_pred)}"
        )

    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    labels = ["No Cancer", "Cancer"]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} - Confusion Matrix")

    save_path = plots_dir / f"{model_name}_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix plot saved to: {save_path}")
    mlflow.log_artifact(str(save_path))


def plot_model_comparison(
    model_names: list[str],
    accuracies: list[float],
    aucs: list[float],
) -> None:
    """Plot a grouped bar chart comparing model performance.

    Args:
        model_names: List of model names for the x-axis.
        accuracies: List of accuracy scores for each model.
        aucs: List of AUC-ROC scores for each model.

    Raises:
        TypeError: If model_names, accuracies, or aucs is not a list.
        ValueError: If the three lists have different lengths.

    """
    if not isinstance(model_names, list):
        raise TypeError(f"model_names must be a list, got {type(model_names).__name__}")
    if not isinstance(accuracies, list):
        raise TypeError(f"accuracies must be a list, got {type(accuracies).__name__}")
    if not isinstance(aucs, list):
        raise TypeError(f"aucs must be a list, got {type(aucs).__name__}")
    if not (len(model_names) == len(accuracies) == len(aucs)):
        raise ValueError("model_names, accuracies, and aucs must have the same length")

    plots_dir = Path("outputs/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="skyblue")
    bars2 = ax.bar(x + width / 2, aucs, width, label="AUC-ROC", color="lightcoral")

    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    save_path = plots_dir / "model_comparison.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Model comparison plot saved to: {save_path}")
    mlflow.log_artifact(str(save_path))
