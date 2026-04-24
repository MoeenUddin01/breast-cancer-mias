"""EfficientNet-B2 model for MIAS breast cancer classification.

Provides a simple function to load EfficientNet-B2 with pretrained weights
and a custom classification head for binary breast cancer detection.
"""

from __future__ import annotations

from torch import nn
from torchvision import models

from src.models.base import build_head, freeze_backbone


def get_efficientnet_b2_model() -> models.EfficientNet:
    """Load an EfficientNet-B2 model with pretrained ImageNet weights.

    The base model layers are frozen, and the classifier is replaced
    with a custom classification head for binary breast cancer detection.

    Returns:
        models.EfficientNet: EfficientNet-B2 model with custom classification head.

    Raises:
        RuntimeError: If pretrained weights fail to download or load.

    """
    try:
        weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1
        model = models.efficientnet_b2(weights=weights)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load EfficientNet-B2 pretrained weights: {e}"
        ) from e

    freeze_backbone(model)

    # Unfreeze the last 3 feature blocks for fine-tuning.
    # This allows the model to adapt ImageNet features to mammography.
    for param in model.features[-3:].parameters():
        param.requires_grad = True

    # EfficientNet-B2 classifier has in_features=1408 at classifier[1]
    model.classifier = build_head(in_features=1408)

    return model
