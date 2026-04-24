"""ResNet152 model for MIAS breast cancer classification.

Provides a simple function to load ResNet-152 with pretrained weights
and a custom classification head for binary breast cancer detection.
"""

from __future__ import annotations

from torch import nn
from torchvision import models

from src.models.base import build_head, freeze_backbone


def get_resnet152_model() -> models.ResNet:
    """Load a ResNet-152 model with pretrained ImageNet weights.

    The base model layers are frozen, and the final fc layer is replaced
    with a custom classification head for binary breast cancer detection.

    Returns:
        models.ResNet: ResNet-152 model with custom classification head.

    Raises:
        RuntimeError: If pretrained weights fail to download or load.

    """
    try:
        weights = models.ResNet152_Weights.IMAGENET1K_V1
        model = models.resnet152(weights=weights)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load ResNet-152 pretrained weights: {e}"
        ) from e

    freeze_backbone(model)

    # Unfreeze the last residual block (layer4) for fine-tuning.
    # This allows the model to adapt ImageNet features to mammography.
    for param in model.layer4.parameters():
        param.requires_grad = True

    model.fc = build_head(in_features=2048)

    return model
