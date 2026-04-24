"""XceptionNet model for MIAS breast cancer classification.

Implements Xception architecture with a custom classification head
using the timm library for pretrained weights.
"""

from __future__ import annotations

import torch
from torch import nn

try:
    import timm
except ImportError:
    timm = None

from src.models.base import (
    CustomClassificationHead,
    build_custom_head,
    freeze_backbone,
    unfreeze_backbone,
)
from src.utils import config_loader as config


class XceptionModel(nn.Module):
    """XceptionNet with custom classification head for binary classification.

    Attributes:
        backbone: Xception feature extractor from timm.
        classifier: Custom classification head with dropout and batch norm.
        num_classes: Number of output classes (1 for binary).

    """

    def __init__(
        self,
        num_classes: int | None = None,
        dropout_rate: float = 0.5,
        hidden_dim: int = 512,
        pretrained: bool = True,
    ) -> None:
        """Initialize the Xception model with custom head.

        Args:
            num_classes: Number of output classes. If None, uses config.NUM_CLASSES.
            dropout_rate: Dropout probability for classification head.
            hidden_dim: Hidden layer dimension for classification head.
            pretrained: Whether to load pretrained ImageNet weights.

        Raises:
            ImportError: If timm library is not installed.
            ValueError: If num_classes is not positive.

        """
        if timm is None:
            raise ImportError(
                "timm library is required for Xception model. "
                "Install with: pip install timm"
            )

        super().__init__()

        if num_classes is None:
            num_classes = config.NUM_CLASSES

        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim

        # Load Xception backbone from timm
        self.backbone = timm.create_model(
            "xception",
            pretrained=pretrained,
            num_classes=0,  # Remove default classifier
            global_pool="avg",
        )

        # Get the number of features from the backbone
        in_features = self.backbone.num_features

        # Build custom classification head
        self.classifier = build_custom_head(
            in_features=in_features,
            dropout_rate=dropout_rate,
            hidden_dim=hidden_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor with shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Output probabilities with shape (batch_size, num_classes).

        """
        # Extract features from backbone
        features = self.backbone(x)

        # Pass through custom classifier
        output = self.classifier(features)

        return output

    def freeze_backbone(self) -> None:
        """Freeze the Xception backbone for transfer learning.

        Returns:
            None

        """
        freeze_backbone(self.backbone)

    def unfreeze_backbone(self) -> None:
        """Unfreeze the Xception backbone for fine-tuning.

        Returns:
            None

        """
        unfreeze_backbone(self.backbone)


def create_xception_model(
    pretrained: bool = True,
    freeze_backbone: bool = True,
    dropout_rate: float = 0.5,
    hidden_dim: int = 512,
) -> XceptionModel:
    """Factory function to create an Xception model.

    Args:
        pretrained: Whether to load pretrained ImageNet weights. Default is True.
        freeze_backbone: Whether to freeze the backbone initially. Default is True.
        dropout_rate: Dropout probability for classification head. Default is 0.5.
        hidden_dim: Hidden layer dimension for classification head. Default is 512.

    Returns:
        XceptionModel: Configured Xception model with custom head.

    """
    model = XceptionModel(
        num_classes=config.NUM_CLASSES,
        dropout_rate=dropout_rate,
        hidden_dim=hidden_dim,
        pretrained=pretrained,
    )

    if freeze_backbone:
        model.freeze_backbone()

    return model
