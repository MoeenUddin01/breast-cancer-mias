"""Model evaluation utilities for PyTorch breast cancer detection.

Provides comprehensive evaluation metrics including accuracy, AUC-ROC,
F1-score, precision, recall, and detailed classification reports.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    model_name: str,
) -> dict[str, float]:
    """Evaluate a trained model and generate comprehensive metrics.

    Computes accuracy, AUC-ROC, F1-score, precision, and recall.
    Prints a classification report and saves it to a text file.

    Args:
        model: The trained model to evaluate.
        dataloader: DataLoader providing test/validation batches.
        device: Device to run evaluation on (cpu or cuda).
        model_name: Name identifier for the model (used for report filename).

    Returns:
        Dictionary containing:
            - "accuracy": Classification accuracy.
            - "auc_roc": AUC-ROC score.
            - "f1_score": F1-score.
            - "precision": Precision score.
            - "recall": Recall score.

    Raises:
        TypeError: If model is not an nn.Module.
        TypeError: If dataloader is not a DataLoader.
        TypeError: If device is not a torch.device.
        TypeError: If model_name is not a string.

    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if not isinstance(dataloader, DataLoader):
        raise TypeError(
            f"dataloader must be a DataLoader, got {type(dataloader).__name__}"
        )
    if not isinstance(device, torch.device):
        raise TypeError(f"device must be a torch.device, got {type(device).__name__}")
    if not isinstance(model_name, str):
        raise TypeError(f"model_name must be a string, got {type(model_name).__name__}")

    model.eval()

    all_predictions: list[int] = []
    all_probabilities: list[float] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs >= 0.5).int()

            all_predictions.extend(preds.cpu().numpy().tolist())
            all_probabilities.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().astype(int).tolist())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    auc = (
        roc_auc_score(all_labels, all_probabilities)
        if len(set(all_labels)) > 1
        else 0.0
    )

    results = {
        "accuracy": accuracy,
        "auc_roc": auc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
    }

    # Log metrics to MLflow
    mlflow.log_metrics({
        "test_accuracy": accuracy,
        "test_auc": auc,
        "test_f1": f1,
        "test_precision": precision,
        "test_recall": recall,
    })

    # Print metrics
    print(f"\n{'=' * 50}")
    print(f"Evaluation Results: {model_name}")
    print(f"{'=' * 50}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"{'=' * 50}")

    # Generate and print classification report
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=["No Cancer", "Cancer"],
        digits=4,
    )
    print("\nClassification Report:")
    print(report)

    # Save report to file
    reports_dir = Path("outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"{model_name}_report.txt"

    with open(report_path, "w") as f:
        f.write(f"Evaluation Results: {model_name}\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"AUC-ROC:   {auc:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"{'=' * 50}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"\nReport saved to: {report_path}")

    # Log report artifact to MLflow
    mlflow.log_artifact(f"outputs/reports/{model_name}_report.txt")

    return results
