"""Preprocessing pipeline for MIAS breast cancer detection.

Loads raw data, splits into train/test, applies CLAHE enhancement,
creates PyTorch Datasets with augmentations, and returns DataLoaders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import DataLoader

from src.data.augmentor import get_test_transforms, get_train_transforms
from src.data.dataset import MIASDataset
from src.data.loader import load_data
from src.data.preprocessor import apply_clahe
from src.data.splitter import split_by_image_id

if TYPE_CHECKING:
    from types import SimpleNamespace


def _apply_clahe_to_split(
    data: list[tuple[str, object, int]],
) -> list[tuple[str, object, int]]:
    """Apply CLAHE enhancement to all images in a data split.

    Args:
        data: List of (image_id, image_array, label) tuples.

    Returns:
        List of tuples with CLAHE-enhanced images.

    Raises:
        RuntimeError: If CLAHE fails on any image.

    """
    enhanced_data = []
    for image_id, image_array, label in data:
        try:
            enhanced_image = apply_clahe(image_array)
            enhanced_data.append((image_id, enhanced_image, label))
        except Exception as e:
            raise RuntimeError(
                f"CLAHE failed for image {image_id}: {e}"
            ) from e
    return enhanced_data


def run_preprocessing_pipeline(config: SimpleNamespace) -> tuple[DataLoader, DataLoader]:
    """Run the full preprocessing pipeline and return DataLoaders.

    Loads raw MIAS data, splits by image ID, applies CLAHE enhancement,
    creates PyTorch Datasets with appropriate transforms, and returns
    DataLoaders for training and testing.

    Args:
        config: Configuration object containing:
            - DATA_DIR: Path to raw data directory with Info.txt and PGM images.
            - TEST_SIZE: Fraction of data for test set (0.0 to 1.0).
            - SEED: Random seed for reproducible train/test split.
            - BATCH_SIZE: Batch size for DataLoaders.
            - NUM_WORKERS: Number of worker processes for data loading.
            - IMAGE_SIZE: Target size (H, W) for resized images.

    Returns:
        Tuple containing:
            - train_loader: DataLoader for training data with augmentations.
            - test_loader: DataLoader for test data without augmentations.

    Raises:
        ValueError: If data loading or splitting fails.
        RuntimeError: If DataLoader creation fails.

    """
    # Validate config has required attributes
    required_attrs = ["DATA_DIR", "TEST_SIZE", "SEED", "BATCH_SIZE", "NUM_WORKERS", "IMAGE_SIZE"]
    for attr in required_attrs:
        if not hasattr(config, attr):
            raise AttributeError(f"config missing required attribute: {attr}")

    # Load raw data
    try:
        data = load_data(config.DATA_DIR)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Failed to load data from {config.DATA_DIR}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}") from e

    if not data:
        raise ValueError("No data loaded from DATA_DIR")

    # Split into train and test by image ID
    try:
        train_data, test_data = split_by_image_id(
            data,
            config.TEST_SIZE,
            config.SEED,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to split data: {e}") from e

    if not train_data or not test_data:
        raise ValueError(
            f"Empty split: train={len(train_data)}, test={len(test_data)}"
        )

    # Apply CLAHE to all images in both splits
    try:
        train_data = _apply_clahe_to_split(train_data)
        test_data = _apply_clahe_to_split(test_data)
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"CLAHE enhancement failed: {e}") from e

    # Get transforms for train and test
    try:
        train_transforms = get_train_transforms(config.IMAGE_SIZE)
        test_transforms = get_test_transforms(config.IMAGE_SIZE)
    except Exception as e:
        raise RuntimeError(f"Failed to create transforms: {e}") from e

    # Create datasets
    try:
        train_dataset = MIASDataset(
            data=train_data,
            transform=train_transforms,
            image_size=config.IMAGE_SIZE,
        )
        test_dataset = MIASDataset(
            data=test_data,
            transform=test_transforms,
            image_size=config.IMAGE_SIZE,
        )
    except ValueError as e:
        raise ValueError(f"Failed to create dataset: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to create dataset: {e}") from e

    # Create DataLoaders
    try:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create DataLoader: {e}") from e

    # Print final dataset sizes
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Test loader batches: {len(test_loader)}")

    return train_loader, test_loader
