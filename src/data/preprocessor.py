"""Image preprocessing for MIAS breast cancer dataset.

Provides CLAHE enhancement, resizing, normalization, and conversion to
3-channel format for pretrained CNN models.
"""

from __future__ import annotations

import cv2
import numpy as np


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE to a grayscale image.

    Applies Contrast Limited Adaptive Histogram Equalization to enhance
    local contrast in mammogram images.

    Args:
        image: Input grayscale image array with shape (H, W).

    Returns:
        np.ndarray: CLAHE-enhanced grayscale image array.

    Raises:
        ValueError: If image is not 2D (grayscale).

    """
    if image.ndim != 2:
        raise ValueError(
            f"CLAHE requires 2D grayscale image, got shape {image.shape}"
        )

    try:
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        # Ensure uint8 format for CLAHE
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

        enhanced = clahe.apply(image)
        return enhanced
    except Exception as e:
        print(f"[ERROR] CLAHE failed for image with shape {image.shape}, dtype {image.dtype}: {e}")
        raise


def preprocess_image(image: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Preprocess a grayscale image for model input.

    Converts to 3-channel by stacking, resizes to target size, and
    normalizes pixel values to [0, 1] range.

    Args:
        image: Input grayscale image array with shape (H, W).
        image_size: Target size (height, width) for resizing.

    Returns:
        np.ndarray: Preprocessed 3-channel image with shape (H, W, 3)
            and dtype float32, pixel values in [0, 1] range.

    Raises:
        ValueError: If image is not 2D.

    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale image, got shape {image.shape}")

    try:
        # Convert grayscale to 3-channel by stacking
        image_3ch = np.stack([image, image, image], axis=-1)

        # Resize to target size (OpenCV uses (width, height) ordering)
        resized = cv2.resize(image_3ch, (image_size[1], image_size[0]))

        # Normalize pixel values to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        return normalized
    except Exception as e:
        print(f"[ERROR] Preprocessing failed for image with shape {image.shape}: {e}")
        raise
