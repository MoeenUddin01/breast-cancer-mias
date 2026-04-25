"""XceptionNet model for MIAS breast cancer classification.

Provides a simple function to load Xception with pretrained weights
and a custom classification head for binary breast cancer detection.
"""

from __future__ import annotations

from torch import nn

try:
    import timm
except ImportError as e:
    raise ImportError(
        "timm library is required for Xception model. "
        "Install with: pip install timm"
    ) from e

from src.models.base import build_head, freeze_backbone


def get_xception_model() -> nn.Module:
    """Load an Xception model with pretrained ImageNet weights.

    The base model layers are frozen, and the classifier is replaced
    with a custom classification head for binary breast cancer detection.

    Returns:
        nn.Module: Xception model with custom classification head.

    Raises:
        ImportError: If timm library is not installed.
        RuntimeError: If pretrained weights fail to download or load.

    """
    try:
        model = timm.create_model(
            "xception",
            pretrained=True,
            num_classes=0,  # Remove default classifier
            global_pool="avg",
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load Xception pretrained weights: {e}"
        ) from e

    freeze_backbone(model)

    # Debug: print Xception structure
    print("Xception children:")
    for name, _ in model.named_children():
        print(f"  - {name}")

    # timm Xception uses named children not .blocks
    # Unfreeze last 2 named children of the backbone
    # (everything except the custom fc head we just added)
    children = list(model.named_children())
    backbone_children = [(name, module) for name, module
                         in children if name != "fc"]

    # Unfreeze last 2 backbone children
    for name, module in backbone_children[-2:]:
        print(f"Unfreezing: {name}")
        for param in module.parameters():
            param.requires_grad = True

    # Xception has 2048 features
    model.fc = build_head(in_features=2048)

    return model
