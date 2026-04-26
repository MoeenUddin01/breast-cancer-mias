from __future__ import annotations

import torch.nn as nn


def build_vit_head(in_features: int = 768) -> nn.Sequential:
    """Custom head for ViT-B/16.

    Designed for BCEWithLogitsLoss — NO sigmoid at end.
    in_features=768 for ViT-B/16

    Args:
        in_features: Input feature dimension from ViT backbone.

    Returns:
        Sequential neural network head with LayerNorm and GELU activations.

    """
    return nn.Sequential(
        nn.LayerNorm(in_features),  # ViT benefits from LayerNorm
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 512),
        nn.GELU(),  # ViT uses GELU not ReLU
        nn.Dropout(p=0.5),
        nn.Linear(512, 256),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 1),
    )
