"""Configuration constants for the MIAS breast cancer detection project."""

from __future__ import annotations

import torch

# Data paths
DATA_DIR: str = "data/all-mias/"
PROCESSED_DIR: str = "data/all-mias/processed/"
OUTPUT_DIR: str = "outputs/"
MODELS_DIR: str = "outputs/models/"
PLOTS_DIR: str = "outputs/plots/"
REPORTS_DIR: str = "outputs/reports/"

# Image and model parameters
IMAGE_SIZE: tuple[int, int] = (224, 224)
BATCH_SIZE: int = 32
EPOCHS: int = 40
LEARNING_RATE: float = 1e-4
PATIENCE: int = 6
TEST_SIZE: float = 0.15
SEED: int = 42
NUM_CLASSES: int = 1
NUM_WORKERS: int = 4

# Device configuration
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model names for comparison
MODEL_NAMES: list[str] = ["xception", "resnet152", "efficientnet_b2"]
