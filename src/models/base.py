"""Base model components for MIAS breast cancer classification.

Provides shared building blocks for custom classification heads
used across all model architectures (Xception, ResNet, EfficientNet).
"""

from __future__ import annotations

import torch
from torch import nn

from src.utils import config_loader as config


class CustomClassificationHead(nn.Module):
    """Custom classification head for binary breast cancer detection.

    Features a fully connected architecture with dropout, batch normalization,
    and sigmoid activation for binary classification output.

    Attributes:
        in_features: Number of input features from the backbone.
        dropout_rate: Dropout probability for regularization.
        hidden_dim: Dimension of the hidden layer.

    """

    def __init__(
        self,
        in_features: int,
        dropout_rate: float = 0.5,
        hidden_dim: int = 512,
    ) -> None:
        """Initialize the custom classification head.

        Args:
            in_features: Number of input features from the backbone model.
            dropout_rate: Dropout probability. Default is 0.5.
            hidden_dim: Size of the hidden fully connected layer. Default is 512.

        Raises:
            ValueError: If in_features or hidden_dim is not positive.

        """
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

        super().__init__()

        self.in_features = in_features
        self.dropout_rate = dropout_rate
        self.hidden_dim = hidden_dim

        self.classifier = nn.Sequential(
            # First dropout layer for regularization
            nn.Dropout(p=dropout_rate),
            # Hidden fully connected layer
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            # Second dropout layer
            nn.Dropout(p=dropout_rate),
            # Output layer for binary classification
            nn.Linear(hidden_dim, config.NUM_CLASSES),
            nn.BatchNorm1d(config.NUM_CLASSES),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classification head.

        Args:
            x: Input tensor with shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output probabilities with shape (batch_size, NUM_CLASSES).

        """
        return self.classifier(x)


def build_custom_head(
    in_features: int,
    dropout_rate: float = 0.5,
    hidden_dim: int = 512,
) -> CustomClassificationHead:
    """Factory function to create a custom classification head.

    Args:
        in_features: Number of input features from the backbone model.
        dropout_rate: Dropout probability. Default is 0.5.
        hidden_dim: Size of the hidden fully connected layer. Default is 512.

    Returns:
        CustomClassificationHead: Configured classification head module.

    """
    return CustomClassificationHead(
        in_features=in_features,
        dropout_rate=dropout_rate,
        hidden_dim=hidden_dim,
    )


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters of a model's backbone (feature extractor).

    Useful for transfer learning when only training the classification head.

    Args:
        model: The model whose backbone parameters should be frozen.

    Returns:
        None

    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all parameters of a model for fine-tuning.

    Args:
        model: The model whose parameters should be unfrozen.

    Returns:
        None

    """
    for param in model.parameters():
        param.requires_grad = True


def get_trainable_params(model: nn.Module) -> list[nn.Parameter]:
    """Get list of trainable parameters from a model.

    Args:
        model: The model to extract trainable parameters from.

    Returns:
        List[nn.Parameter]: List of parameters with requires_grad=True.

    """
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters in a model.

    Args:
        model: The model to count parameters for.

    Returns:
        Tuple containing:
            - int: Total number of parameters.
            - int: Number of trainable parameters.

    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
