"""Data loader for MIAS breast cancer dataset.

Handles parsing of Info.txt metadata and loading of PGM mammogram images
using OpenCV.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.utils import config


def load_data(data_dir: str | None = None) -> list[tuple[str, np.ndarray, int]]:
    """Load MIAS dataset from Info.txt and corresponding PGM images.

    Parses Info.txt to extract image filenames and labels, then loads
    each corresponding PGM image using OpenCV.

    Args:
        data_dir: Directory containing Info.txt and PGM images.
            If None, uses config.DATA_DIR.

    Returns:
        List of tuples containing:
            - image_id (str): The image filename (e.g., 'mdb001')
            - image_array (np.ndarray): Grayscale image loaded with OpenCV
            - label (int): Binary label (0 for benign/normal, 1 for malignant)

    Raises:
        FileNotFoundError: If Info.txt or a referenced image does not exist.

    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    data_path = Path(data_dir)
    info_path = data_path / "Info.txt"

    if not info_path.exists():
        raise FileNotFoundError(f"Info.txt not found at: {info_path}")

    data = []
    benign_count = 0
    malignant_count = 0

    try:
        with open(info_path, "r") as f:
            for line in f:
                try:
                    line = line.strip()

                    # Skip header line
                    if line == "Truth-Data:":
                        continue

                    tokens = line.split()

                    # Skip lines with fewer than 4 tokens
                    if len(tokens) < 4:
                        continue

                    image_id = tokens[0]
                    severity = tokens[3]

                    # Map label: B = 0, M = 1
                    if severity == "B":
                        label = 0
                        benign_count += 1
                    elif severity == "M":
                        label = 1
                        malignant_count += 1
                    else:
                        # Skip normal/unknown labels
                        continue

                    # Load the image using OpenCV
                    image_file = data_path / f"{image_id}.pgm"
                    try:
                        image_array = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
                    except Exception as e:
                        print(f"[ERROR] cv2.imread failed for {image_file}: {e}")
                        continue

                    if image_array is None:
                        print(f"[WARNING] Failed to load image: {image_file}")
                        continue

                    data.append((image_id, image_array, label))

                except Exception as e:
                    print(f"[WARNING] Error processing line '{line}': {e}")
                    continue

    except FileNotFoundError:
        print(f"[ERROR] Info.txt not found at: {info_path}")
        raise
    except Exception as e:
        print(f"[ERROR] Failed to read Info.txt: {e}")
        raise

    print(f"Loaded {benign_count} benign and {malignant_count} malignant samples")

    return data
