"""Configuration constants for BreakHis dataset and MLflow integration."""

from __future__ import annotations

DAGSHUB_REPO_OWNER = "your_dagshub_username"
DAGSHUB_REPO_NAME = "breast-cancer-mias"
EXPERIMENT_NAME = "BreakHis_Breast_Cancer_Detection"

# Dataset configuration
DATA_DIR = "/kaggle/input/breakhis/BreaKHis_v1/histology_slides/breast/"
MAGNIFICATIONS = ["40X", "100X", "200X", "400X"]

# Training configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
TEST_SIZE = 0.15
SEED = 42
NUM_CLASSES = 1
