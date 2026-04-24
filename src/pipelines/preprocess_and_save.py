"""Preprocess MIAS images and save to disk.

Loads raw PGM images, splits into train/test, applies CLAHE + preprocessing,
saves processed images to processed/train/ and processed/test/ directories.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.data.loader import load_data
from src.data.preprocessor import apply_clahe, preprocess_image
from src.data.splitter import split_by_image_id
from src.utils import config_loader as config
from src.utils.helpers import create_directories


def preprocess_and_save(
    raw_data_dir: str | None = None,
    processed_dir: str | None = None,
    image_size: tuple[int, int] | None = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> None:
    """Load raw MIAS images, split, preprocess, and save to disk.

    Args:
        raw_data_dir: Directory with raw PGM images and Info.txt.
            If None, uses config.DATA_DIR.
        processed_dir: Directory to save processed images.
            If None, uses config.DATA_DIR/processed/.
        image_size: Target size for preprocessing. If None,
            uses config.IMAGE_SIZE.
        test_size: Fraction of data for test set (default 0.2 for 80/20).
        seed: Random seed for reproducible train/test split.

    Returns:
        None

    """
    if raw_data_dir is None:
        raw_data_dir = config.DATA_DIR
    if processed_dir is None:
        processed_dir = str(Path(raw_data_dir) / "processed")
    if image_size is None:
        image_size = config.IMAGE_SIZE

    raw_path = Path(raw_data_dir)
    proc_path = Path(processed_dir)

    # Create train and test subdirectories
    train_path = proc_path / "train"
    test_path = proc_path / "test"
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    # Load raw data
    data = load_data(raw_data_dir)
    print(f"Loaded {len(data)} raw images")

    # Split into train/test before any preprocessing
    train_data, test_data = split_by_image_id(data, test_size=test_size, seed=seed)

    # Process and save train split
    print(f"\nProcessing {len(train_data)} train images...")
    for image_id, image_array, label in tqdm(train_data, desc="Train"):
        # Apply CLAHE
        enhanced = apply_clahe(image_array)

        # Preprocess (3-channel, resize, normalize)
        processed = preprocess_image(enhanced, image_size)

        # Convert from float32 [0,1] to uint8 [0,255] for saving
        save_array = (processed * 255).astype(np.uint8)

        # Save as PNG (3-channel)
        output_file = train_path / f"{image_id}.png"
        cv2.imwrite(str(output_file), cv2.cvtColor(save_array, cv2.COLOR_RGB2BGR))

    # Process and save test split
    print(f"\nProcessing {len(test_data)} test images...")
    for image_id, image_array, label in tqdm(test_data, desc="Test"):
        # Apply CLAHE
        enhanced = apply_clahe(image_array)

        # Preprocess (3-channel, resize, normalize)
        processed = preprocess_image(enhanced, image_size)

        # Convert from float32 [0,1] to uint8 [0,255] for saving
        save_array = (processed * 255).astype(np.uint8)

        # Save as PNG (3-channel)
        output_file = test_path / f"{image_id}.png"
        cv2.imwrite(str(output_file), cv2.cvtColor(save_array, cv2.COLOR_RGB2BGR))

    print(f"\nSaved processed images:")
    print(f"  - Train: {train_path} ({len(train_data)} images)")
    print(f"  - Test: {test_path} ({len(test_data)} images)")


if __name__ == "__main__":
    create_directories()
    preprocess_and_save()
