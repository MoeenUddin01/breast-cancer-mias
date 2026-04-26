"""ViT-B/16 model for breast cancer detection.

Pretrained on ImageNet-21k (14 million images).
Fine-tuned on BreakHis histology dataset.
"""

from __future__ import annotations

import timm
import torch.nn as nn

from src.transformers.vit_config import (
    VIT_IN_FEATURES,
    VIT_MODEL_NAME,
    VIT_PRETRAINED,
    VIT_UNFREEZE_BLOCKS,
)
from src.transformers.vit_head import build_vit_head


def get_vit_model() -> nn.Module:
    """Load pretrained ViT-B/16 and attach custom head.

    Architecture:
    - Backbone: ViT-B/16 pretrained on ImageNet-21k
    - Head: LayerNorm → Dropout → Linear(768→512) →
            GELU → Dropout → Linear(512→256) →
            GELU → Dropout → Linear(256→1)
    - Loss: BCEWithLogitsLoss (no sigmoid in head)

    Returns:
        ViT-B/16 model with custom classification head attached.

    """
    # Load pretrained ViT-B/16
    # num_classes=0 removes the original ImageNet head
    # This gives us raw 768-dim features
    model = timm.create_model(
        VIT_MODEL_NAME, pretrained=VIT_PRETRAINED, num_classes=0  # remove ImageNet head
    )

    # Step 1: Freeze ALL backbone layers
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: Add custom classification head
    # ViT-B/16 outputs 768-dim CLS token features
    model.head = build_vit_head(in_features=VIT_IN_FEATURES)

    # Step 3: Make sure head is trainable
    for param in model.head.parameters():
        param.requires_grad = True

    # Print structure confirmation
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: ViT-B/16 (ImageNet-21k pretrained)")
    print(f"  Total parameters    : {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Frozen parameters   : {total - trainable:,}")

    return model


def unfreeze_vit_blocks(model: nn.Module, num_blocks: int = 4) -> None:
    """Unfreeze last N transformer blocks for Phase 2 fine-tuning.

    ViT-B/16 has 12 transformer blocks total.
    We unfreeze last 4 for fine-tuning.

    Args:
        model: The ViT model to unfreeze blocks in.
        num_blocks: Number of transformer blocks to unfreeze from the end.

    """
    blocks = list(model.blocks.children())
    total_blocks = len(blocks)

    for block in blocks[-num_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    # Also unfreeze the final norm layer
    if hasattr(model, "norm"):
        for param in model.norm.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Unfrozen last {num_blocks}/{total_blocks} " f"transformer blocks")
    print(f"  Trainable parameters now: {trainable:,}")
