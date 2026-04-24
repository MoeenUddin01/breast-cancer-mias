"""Custom PyTorch Dataset for MIAS breast cancer classification.

Integrates data loading, preprocessing, and augmentation into a
PyTorch-compatible Dataset class for use with DataLoader.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.preprocessor import preprocess_image


class MIASDataset(Dataset):
    """PyTorch Dataset for MIAS mammogram images.

    Handles preprocessing and augmentation of mammogram images
    for binary classification.

    Attributes:
        data: List of (image_id, image_array, label) tuples.
        transform: Transform pipeline for augmentation/normalization.
        image_size: Target size for resizing images.

    """

    def __init__(
        self,
        data: list[tuple[str, np.ndarray, int]],
        transform: Callable,
        image_size: tuple[int, int],
    ) -> None:
        """Initialize the MIAS Dataset.

        Args:
            data: List of (image_id, image_array, label) tuples from loader.
            transform: Torchvision transform pipeline for augmentation.
            image_size: Target size (H, W) for resized images.

        Raises:
            ValueError: If data list is empty.

        """
        if not data:
            raise ValueError("Data list cannot be empty")

        try:
            self.data = data
            self.transform = transform
            self.image_size = image_size
        except Exception as e:
            print(f"[ERROR] Failed to initialize MIASDataset: {e}")
            raise

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: Total number of samples.

        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single preprocessed image and its label by index.

        Applies preprocess_image, converts to PIL, applies transform,
        and returns the tensor with label.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple containing:
                - torch.Tensor: Preprocessed and augmented image tensor.
                - torch.Tensor: Binary label as torch.float32.

        Raises:
            IndexError: If idx is out of range.

        """
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        try:
            image_id, image_array, label = self.data[idx]
        except Exception as e:
            print(f"[ERROR] Failed to retrieve data at index {idx}: {e}")
            raise

        try:
            # Apply preprocessing (CLAHE, 3-channel conversion, resize, normalize)
            processed = preprocess_image(image_array, self.image_size)
        except Exception as e:
            print(f"[ERROR] Preprocessing failed for image {image_id} at index {idx}: {e}")
            raise

        try:
            # Convert float32 [0, 1] → uint8 [0, 255] numpy array (H, W, C).
            # transforms.ToTensor() accepts numpy (H, W, C) uint8 arrays directly,
            # so no PIL conversion is needed and ToPILImage conflicts are avoided.
            image_uint8 = (processed * 255).astype(np.uint8)
        except Exception as e:
            print(f"[ERROR] Failed to convert to uint8 for {image_id}: {e}")
            raise

        try:
            # Apply transform pipeline (expects numpy uint8 or PIL Image)
            tensor = self.transform(image_uint8)
        except Exception as e:
            print(f"[ERROR] Transform failed for image {image_id}: {e}")
            raise

        try:
            # Return tensor and label as float32
            label_tensor = torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"[ERROR] Failed to create label tensor for image {image_id}, label={label}: {e}")
            raise

        return tensor, label_tensor
