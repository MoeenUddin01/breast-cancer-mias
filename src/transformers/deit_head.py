from __future__ import annotations

import torch.nn as nn


def build_deit_head(in_features: int = 768) -> nn.Sequential:
    """Custom head for DeiT-B Distilled.

    Designed for BCEWithLogitsLoss — NO sigmoid at end.
    Uses LayerNorm + GELU activation pattern suitable for transformers.

    Args:
        in_features: Input feature dimension from DeiT backbone.

    Returns:
        Sequential neural network head with LayerNorm and GELU activations.

    """
    return nn.Sequential(
        nn.LayerNorm(in_features),
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 768),
        nn.GELU(),
        nn.LayerNorm(768),
        nn.Dropout(p=0.5),
        nn.Linear(768, 384),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(384, 1),
    )
