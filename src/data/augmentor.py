"""Data augmentation for BreakHis breast cancer training data.

Applies torchvision transforms ONLY on the training split.
Test data receives only normalization and resizing without augmentation.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


def get_train_transforms(image_size: tuple[int, int]) -> Callable:
    """Get data augmentation transforms for training.

    Applies random augmentations suitable for mammogram images including
    flips, rotation, and color jitter. Training transforms are ONLY ever
    used on training split data.

    Args:
        image_size: Target size (H, W) for resized images.

    Returns:
        Callable: Composed torchvision transform pipeline.

    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_test_transforms(image_size: tuple[int, int]) -> Callable:
    """Get transforms for test/validation data (no augmentation).

    Applies only resizing and normalization without random augmentations.

    Args:
        image_size: Target size (H, W) for resized images.

    Returns:
        Callable: Composed torchvision transform pipeline.

    """
    return transforms.Compose(
        [
            # PIL Image is already passed from dataset, no ToPILImage needed
            # Resize to target size
            transforms.Resize(image_size),
            # Convert to tensor
            transforms.ToTensor(),
            # Normalize using ImageNet statistics
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def augment_training_data(
    train_data: list[tuple[str, np.ndarray, int]],
) -> list[tuple[str, np.ndarray, int]]:
    """Expand the training set with offline augmentations.

    Applies deterministic augmentations to each training image and appends
    the resulting samples to the original list. Test data is NEVER passed
    here — call this ONLY on the training split after split_by_image_id().

    Augmentations applied per image:
        - Horizontal flip
        - Vertical flip
        - 90-degree rotation
        - 180-degree rotation
        - 270-degree rotation
        - Brightness boost (+20 % via numpy clip)
        - Contrast adjustment (histogram-stretch)

    Args:
        train_data: List of (image_id, image_array, label) tuples where
            image_array is a 2D uint8 grayscale numpy array.

    Returns:
        list[tuple[str, np.ndarray, int]]: Expanded list containing the
            original samples plus all augmented variants.

    Raises:
        ValueError: If train_data is empty.

    """
    if not train_data:
        raise ValueError("train_data cannot be empty")

    augmented: list[tuple[str, np.ndarray, int]] = list(train_data)

    for image_id, image_array, label in train_data:
        img = image_array.astype(np.uint8)

        # Horizontal flip
        augmented.append((f"{image_id}_hflip", np.fliplr(img).copy(), label))

        # Vertical flip
        augmented.append((f"{image_id}_vflip", np.flipud(img).copy(), label))

        # Rotations at 90, 180, 270 degrees
        augmented.append((f"{image_id}_rot90", np.rot90(img, k=1).copy(), label))
        augmented.append((f"{image_id}_rot180", np.rot90(img, k=2).copy(), label))
        augmented.append((f"{image_id}_rot270", np.rot90(img, k=3).copy(), label))

        # Brightness boost (+20 %, clamped to [0, 255])
        bright = np.clip(img.astype(np.int32) + 51, 0, 255).astype(np.uint8)
        augmented.append((f"{image_id}_bright", bright, label))

        # Contrast stretch (linear min-max normalisation to full uint8 range)
        lo, hi = int(img.min()), int(img.max())
        if hi > lo:
            contrast = (
                (img.astype(np.float32) - lo) / (hi - lo) * 255
            ).astype(np.uint8)
        else:
            contrast = img.copy()
        augmented.append((f"{image_id}_contrast", contrast, label))

        # Brightness + contrast combined
        bright_contrast = cv2.convertScaleAbs(img, alpha=1.3, beta=15)
        augmented.append((f"{image_id}_bc", bright_contrast, label))

    original_count = len(train_data)
    total_count = len(augmented)
    print(
        f"✓ Augmentation complete: {original_count} → {total_count} training samples "
        f"({total_count // original_count}× expansion)"
    )
    return augmented
