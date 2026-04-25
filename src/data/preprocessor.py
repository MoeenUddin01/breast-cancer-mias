"""Image preprocessing for BreakHis breast cancer dataset.

Provides CLAHE enhancement for RGB images, resizing, and normalization
for pretrained CNN models.
"""

from __future__ import annotations

import cv2
import numpy as np


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE to an RGB image.

    Converts RGB to LAB color space, applies Contrast Limited Adaptive
    Histogram Equalization to the L (luminance) channel, then converts
    back to RGB.

    Args:
        image: Input RGB image array with shape (H, W, 3).

    Returns:
        np.ndarray: CLAHE-enhanced RGB image array.

    Raises:
        ValueError: If image is not 3D RGB.

    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"CLAHE for RGB requires 3D image with 3 channels, got shape {image.shape}"
        )

    try:
        # Convert RGB to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # Split channels
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )
        l_enhanced = clahe.apply(l)

        # Merge channels and convert back to RGB
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

        return enhanced
    except Exception as e:
        print(f"[ERROR] CLAHE failed for image with shape {image.shape}, dtype {image.dtype}: {e}")
        raise


def preprocess_image(image: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Preprocess an RGB image for model input.

    Resizes to target size and normalizes pixel values to [0, 1] range.

    Args:
        image: Input RGB image array with shape (H, W, 3).
        image_size: Target size (height, width) for resizing.

    Returns:
        np.ndarray: Preprocessed RGB image with shape (H, W, 3)
            and dtype float32, pixel values in [0, 1] range.

    Raises:
        ValueError: If image is not 3D with 3 channels.

    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected 3D RGB image with 3 channels, got shape {image.shape}")

    try:
        # Resize to target size (OpenCV uses (width, height) ordering)
        resized = cv2.resize(image, (image_size[1], image_size[0]))

        # Normalize pixel values to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        return normalized
    except Exception as e:
        print(f"[ERROR] Preprocessing failed for image with shape {image.shape}: {e}")
        raise
