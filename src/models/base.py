"""Base model components for MIAS breast cancer classification.

Provides shared building blocks for custom classification heads
used across all model architectures (Xception, ResNet, EfficientNet).
"""

from __future__ import annotations

from torch import nn


def build_head(in_features: int) -> nn.Sequential:
    """Build a custom classification head for binary breast cancer detection.

    Architecture designed for use with BCEWithLogitsLoss (no sigmoid output).

    Args:
        in_features: Number of input features from the backbone model.

    Returns:
        nn.Sequential: Classification head with dropout, batch norm, and ReLU.

    Raises:
        ValueError: If in_features is not positive.

    """
    if in_features <= 0:
        raise ValueError(f"in_features must be positive, got {in_features}")

    return nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(1024, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(1024, 1),
    )


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all parameters of a model's backbone (feature extractor).

    Useful for transfer learning when only training the classification head.

    Args:
        model: The model whose backbone parameters should be frozen.

    Returns:
        None

    Raises:
        TypeError: If model is not an nn.Module.

    """
    if model is None:
        raise TypeError("model cannot be None")
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze all parameters of a model for fine-tuning.

    Args:
        model: The model whose parameters should be unfrozen.

    Returns:
        None

    Raises:
        TypeError: If model is not an nn.Module.

    """
    if model is None:
        raise TypeError("model cannot be None")
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    for param in model.parameters():
        param.requires_grad = True


def get_trainable_params(model: nn.Module) -> list[nn.Parameter]:
    """Get list of trainable parameters from a model.

    Args:
        model: The model to extract trainable parameters from.

    Returns:
        List[nn.Parameter]: List of parameters with requires_grad=True.

    Raises:
        TypeError: If model is not an nn.Module.

    """
    if model is None:
        raise TypeError("model cannot be None")
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters in a model.

    Args:
        model: The model to count parameters for.

    Returns:
        Tuple containing:
            - int: Total number of parameters.
            - int: Number of trainable parameters.

    Raises:
        TypeError: If model is not an nn.Module.

    """
    if model is None:
        raise TypeError("model cannot be None")
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
