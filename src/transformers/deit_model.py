"""DeiT-Small Distilled model for breast cancer detection.

Pretrained on ImageNet-1k with distillation.
Fine-tuned on BreakHis histology dataset.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn

from src.transformers.deit_config import (
    DEIT_DROP_PATH_RATE,
    DEIT_MODEL_NAME,
    DEIT_PRETRAINED,
)
from src.transformers.deit_head import build_deit_head


class DeiTBinaryClassifier(nn.Module):
    """Wrap a DeiT backbone and force binary-logit output."""

    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits with shape [B, 1] for BCEWithLogitsLoss."""
        features = self.backbone(x)

        if isinstance(features, (tuple, list)):
            features = features[0]

        if features.dim() == 3:
            # Some DeiT/timm variants may return token sequence [B, N, C].
            features = features[:, 0]

        logits = self.head(features)
        if logits.dim() == 1:
            logits = logits.unsqueeze(1)
        return logits


def get_deit_model() -> nn.Module:
    """Load pretrained DeiT-Small Distilled and attach custom head.

    Architecture:
    - Backbone: DeiT-Small Distilled pretrained on ImageNet-1k
    - Head: LayerNorm → Dropout → Linear → GELU → LayerNorm →
            Dropout → Linear → GELU → Dropout → Linear(1)
    - Loss: BCEWithLogitsLoss (no sigmoid in head)

    DeiT has TWO output tokens: CLS token and DIST token.
    timm DeiT with num_classes=0 returns concatenated
    [CLS, DIST] = 384*2 = 768 features depending on version.

    Returns:
        DeiT-Small Distilled model with custom classification head attached.

    """
    backbone = timm.create_model(
        DEIT_MODEL_NAME,
        pretrained=DEIT_PRETRAINED,
        num_classes=0,
        img_size=224,
        drop_path_rate=DEIT_DROP_PATH_RATE,
    )

    # Check actual output size
    dummy = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        out = backbone(dummy)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.dim() == 3:
        out = out[:, 0]
    actual_features = out.shape[-1]
    print(f"  DeiT output features: {actual_features}")

    # Step 1: Freeze ALL backbone layers
    for param in backbone.parameters():
        param.requires_grad = False

    # Step 2: Add custom classification head
    model = DeiTBinaryClassifier(
        backbone=backbone,
        head=build_deit_head(in_features=actual_features),
    )

    # Step 3: Make sure custom head is trainable
    for param in model.head.parameters():
        param.requires_grad = True

    # Print structure confirmation
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print("  Model: DeiT-Small Distilled (ImageNet-1k+distillation)")
    print(f"  Total parameters    : {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Frozen parameters   : {frozen:,}")

    return model


def unfreeze_deit_blocks(model: nn.Module, num_blocks: int = 8) -> None:
    """Unfreeze last N transformer blocks for Phase 2 fine-tuning.

    DeiT-Small Distilled has 12 transformer blocks total.
    We unfreeze last 8 for fine-tuning (balanced speed and performance).

    Args:
        model: The DeiT model to unfreeze blocks in.
        num_blocks: Number of transformer blocks to unfreeze from the end.

    """
    base_model = model.backbone if hasattr(model, "backbone") else model
    blocks = list(base_model.blocks.children())
    total_blocks = len(blocks)

    for block in blocks[-num_blocks:]:
        for param in block.parameters():
            param.requires_grad = True

    # Also unfreeze the final norm layer
    if hasattr(base_model, "norm"):
        for param in base_model.norm.parameters():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Unfrozen last {num_blocks}/{total_blocks} transformer blocks")
    print(f"  Trainable parameters now: {trainable:,}")
