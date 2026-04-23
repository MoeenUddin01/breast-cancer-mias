"""Preprocess MIAS images and save to disk.

Loads raw PGM images, applies CLAHE + preprocessing, saves processed
images to processed/ directory for faster training.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.data.loader import load_data
from src.data.preprocessor import apply_clahe, preprocess_image
from src.utils import config
from src.utils.helpers import create_directories


def preprocess_and_save(
    raw_data_dir: str | None = None,
    processed_dir: str | None = None,
    image_size: tuple[int, int] | None = None,
) -> None:
    """Load raw MIAS images, preprocess, and save to disk.

    Args:
        raw_data_dir: Directory with raw PGM images and Info.txt.
            If None, uses config.DATA_DIR.
        processed_dir: Directory to save processed images.
            If None, uses config.DATA_DIR/processed/.
        image_size: Target size for preprocessing. If None,
            uses config.IMAGE_SIZE.

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
    proc_path.mkdir(parents=True, exist_ok=True)

    # Load raw data
    data = load_data(raw_data_dir)
    print(f"Processing {len(data)} images...")

    for image_id, image_array, label in tqdm(data):
        # Apply CLAHE
        enhanced = apply_clahe(image_array)

        # Preprocess (3-channel, resize, normalize)
        processed = preprocess_image(enhanced, image_size)

        # Convert from float32 [0,1] to uint8 [0,255] for saving
        save_array = (processed * 255).astype(np.uint8)

        # Save as PNG (3-channel)
        output_file = proc_path / f"{image_id}.png"
        cv2.imwrite(str(output_file), cv2.cvtColor(save_array, cv2.COLOR_RGB2BGR))

    print(f"Saved {len(data)} processed images to {proc_path}")


if __name__ == "__main__":
    create_directories()
    preprocess_and_save()
