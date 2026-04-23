"""Dataset splitter for MIAS breast cancer data.

Splits data by unique image ID BEFORE augmentation to prevent
any data leakage between train and test sets.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def split_by_image_id(
    data: list[tuple[str, np.ndarray, int]],
    test_size: float,
    seed: int,
) -> tuple[list[tuple[str, np.ndarray, int]], list[tuple[str, np.ndarray, int]]]:
    """Split data by unique image IDs with stratification.

    Performs a stratified train/test split at the image ID level to
    ensure no image_id appears in both train and test sets.

    Args:
        data: List of (image_id, image_array, label) tuples from loader.py.
        test_size: Fraction of data to use for the test set (0.0 to 1.0).
        seed: Random seed for reproducibility, passed to train_test_split
            as random_state.

    Returns:
        Tuple containing:
            - train_data: List of (image_id, image_array, label) for training.
            - test_data: List of (image_id, image_array, label) for testing.

    Raises:
        ValueError: If data is empty or if split results in leakage.

    """
    if not data:
        raise ValueError("Data cannot be empty")

    try:
        # Extract unique image IDs and their labels
        unique_ids = []
        unique_labels = []

        seen_ids = set()
        for image_id, _, label in data:
            if image_id not in seen_ids:
                unique_ids.append(image_id)
                unique_labels.append(label)
                seen_ids.add(image_id)

        if len(unique_ids) == 0:
            raise ValueError("No unique image IDs found in data")

        if len(set(unique_labels)) < 2:
            print(f"[WARNING] Only one class found ({set(unique_labels)}), stratification disabled")
            stratify = None
        else:
            stratify = unique_labels

        # Perform stratified split on unique image IDs
        train_ids, test_ids = train_test_split(
            unique_ids,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except Exception as e:
        print(f"[ERROR] Failed to split data: {e}")
        raise

    try:
        # Convert to sets for O(1) lookup
        train_id_set = set(train_ids)
        test_id_set = set(test_ids)

        # Verify no overlap
        overlap = train_id_set & test_id_set
        if overlap:
            raise ValueError(f"Data leakage detected: {overlap} appear in both sets")

        # Split original data based on image_id
        train_data = []
        test_data = []

        for image_id, image_array, label in data:
            if image_id in train_id_set:
                train_data.append((image_id, image_array, label))
            elif image_id in test_id_set:
                test_data.append((image_id, image_array, label))
            else:
                print(f"[WARNING] Image ID {image_id} not in train or test split")

    except Exception as e:
        print(f"[ERROR] Failed to assign data to train/test sets: {e}")
        raise

    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

    return train_data, test_data
