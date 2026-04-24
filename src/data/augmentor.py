"""Data augmentation for MIAS breast cancer training data.

Applies torchvision transforms ONLY on the training split.
Test data receives only normalization and resizing without augmentation.
"""

from __future__ import annotations

from typing import Callable

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
    return transforms.Compose(
        [
            # PIL Image is passed from dataset; no ToPILImage needed
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


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
