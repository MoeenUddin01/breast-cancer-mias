"""Utility helper functions for the MIAS breast cancer detection project."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

from src.utils import config


def seed_everything(seed: int | None = None) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: The seed value to use. If None, uses the SEED from config.

    Returns:
        None

    Raises:
        TypeError: If seed is not an integer.

    """
    if seed is None:
        seed = config.SEED
    if not isinstance(seed, int):
        raise TypeError(f"Seed must be an integer, got {type(seed)}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Get the appropriate device for computation.

    Returns:
        torch.device: CUDA device if available, otherwise CPU.

    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_directories() -> None:
    """Create all necessary output directories if they do not exist.

    Creates the following directories:
        - outputs/
        - outputs/models/
        - outputs/plots/
        - outputs/reports/

    Returns:
        None

    """
    dirs_to_create = [
        Path(config.OUTPUT_DIR),
        Path(config.MODELS_DIR),
        Path(config.PLOTS_DIR),
        Path(config.REPORTS_DIR),
    ]

    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)


def save_report(
    report_dict: dict,
    model_name: str,
    reports_dir: str | None = None,
) -> Path:
    """Save a classification report dictionary as a formatted text file.

    Args:
        report_dict: Dictionary containing classification report metrics.
            Expected to have keys like 'accuracy', 'macro avg', 'weighted avg',
            and per-class metrics.
        model_name: Name of the model for the report filename.
        reports_dir: Directory to save the report. If None, uses
            config.REPORTS_DIR.

    Returns:
        Path: Path to the saved report file.

    Raises:
        ValueError: If report_dict is empty or invalid.
        IOError: If the report cannot be written.

    """
    if not report_dict:
        raise ValueError("Report dictionary cannot be empty")

    if reports_dir is None:
        reports_dir = config.REPORTS_DIR

    reports_path = Path(reports_dir)
    reports_path.mkdir(parents=True, exist_ok=True)

    report_file = reports_path / f"{model_name}_report.txt"

    try:
        with open(report_file, "w") as f:
            f.write(f"Classification Report - {model_name}\n")
            f.write("=" * 50 + "\n\n")

            # Write overall metrics
            if "accuracy" in report_dict:
                f.write(f"Accuracy: {report_dict['accuracy']:.4f}\n\n")

            # Write per-class and average metrics
            for key, value in report_dict.items():
                if key == "accuracy":
                    continue
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for metric, score in value.items():
                        if isinstance(score, (int, float)):
                            f.write(f"  {metric}: {score:.4f}\n")
                        else:
                            f.write(f"  {metric}: {score}\n")
                    f.write("\n")
                else:
                    f.write(f"{key}: {value}\n")

    except Exception as e:
        raise IOError(f"Failed to write report to {report_file}: {e}")

    return report_file
