"""DeiT-B Distilled model for breast cancer detection.

Pretrained on ImageNet-1k with distillation.
Fine-tuned on BreakHis histology dataset.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from src.transformers.deit_config import (
    DEIT_DROP_PATH_RATE,
    DEIT_IN_FEATURES,
    DEIT_MODEL_NAME,
    DEIT_PRETRAINED,
    DEIT_UNFREEZE_BLOCKS,
)
from src.transformers.deit_head import build_deit_head


def get_deit_model() -> nn.Module:
    """Load pretrained DeiT-B Distilled and attach custom head.

    Architecture:
    - Backbone: DeiT-B Distilled pretrained on ImageNet-1k
    - Head: LayerNorm → Dropout → Linear → GELU → LayerNorm →
            Dropout → Linear → GELU → Dropout → Linear(1)
    - Loss: BCEWithLogitsLoss (no sigmoid in head)

    DeiT has TWO output tokens: CLS token and DIST token.
    timm DeiT with num_classes=0 returns concatenated
    [CLS, DIST] = 768*2 = 1536 features depending on version.

    Returns:
        DeiT-B Distilled model with custom classification head attached.

    """
    model = timm.create_model(
        DEIT_MODEL_NAME,
        pretrained=DEIT_PRETRAINED,
        num_classes=0,
        drop_path_rate=DEIT_DROP_PATH_RATE,
    )

    # Check actual output size
    dummy = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    actual_features = out.shape[-1]
    print(f"  DeiT output features: {actual_features}")

    # Step 1: Freeze ALL backbone layers
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: Add custom classification head
    model.head = build_deit_head(in_features=actual_features)
    model.head_dist = nn.Identity()

    # Step 3: Make sure head and head_dist are trainable
    for param in model.head.parameters():
        param.requires_grad = True
    for param in model.head_dist.parameters():
        param.requires_grad = True

    # Print structure confirmation
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"  Model: DeiT-B Distilled (ImageNet-1k+distillation)")
    print(f"  Total parameters    : {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Frozen parameters   : {frozen:,}")

    return model


def unfreeze_deit_blocks(model: nn.Module, num_blocks: int = 8) -> None:
    """Unfreeze last N transformer blocks for Phase 2 fine-tuning.

    DeiT-B Distilled has 12 transformer blocks total.
    We unfreeze last 8 for fine-tuning (more than ViT for better adaptation).

    Args:
        model: The DeiT model to unfreeze blocks in.
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
    print(f"  Unfrozen last {num_blocks}/{total_blocks} transformer blocks")
    print(f"  Trainable parameters now: {trainable:,}")
