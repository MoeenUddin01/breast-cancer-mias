"""Configuration loader for YAML config files.

Loads and provides access to project configuration from config.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml


def _load_config() -> dict[str, Any]:
    """Load configuration from YAML file.

    Returns:
        Dict containing all configuration values.

    Raises:
        FileNotFoundError: If config.yaml does not exist.
        yaml.YAMLError: If config.yaml is invalid.

    """
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Load config once at module import
_CONFIG = _load_config()

# Data paths
DATA_DIR: str = _CONFIG["data"]["dir"]
PROCESSED_DIR: str = _CONFIG["data"]["processed_dir"]

# Output paths
OUTPUT_DIR: str = _CONFIG["output"]["dir"]
MODELS_DIR: str = _CONFIG["output"]["models_dir"]
PLOTS_DIR: str = _CONFIG["output"]["plots_dir"]
REPORTS_DIR: str = _CONFIG["output"]["reports_dir"]

# Image parameters
IMAGE_SIZE: tuple[int, int] = tuple(_CONFIG["image"]["size"])

# Training parameters
BATCH_SIZE: int = _CONFIG["training"]["batch_size"]
EPOCHS: int = _CONFIG["training"]["epochs"]
LEARNING_RATE: float = _CONFIG["training"]["learning_rate"]
PATIENCE: int = _CONFIG["training"]["patience"]
TEST_SIZE: float = _CONFIG["training"]["test_size"]
SEED: int = _CONFIG["training"]["seed"]
NUM_WORKERS: int = _CONFIG["training"]["num_workers"]

# Model parameters
NUM_CLASSES: int = _CONFIG["model"]["num_classes"]
MODEL_NAMES: list[str] = _CONFIG["model"]["names"]

# Device configuration
if "device" in _CONFIG:
    DEVICE: torch.device = torch.device(_CONFIG["device"])
else:
    DEVICE: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
