# Models Architecture Diagram

This diagram illustrates the structure and relationships between the model components in `src/models/`.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        src/models/                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────┐                                              │
│  │   base.py     │◄────────────────────────────────────┐         │
│  │               │                                    │         │
│  │  • build_head │                                    │         │
│  │  • freeze_    │                                    │         │
│  │    backbone   │                                    │         │
│  │  • unfreeze_  │                                    │         │
│  │    backbone   │                                    │         │
│  │  • get_       │                                    │         │
│  │    trainable_ │                                    │         │
│  │    params     │                                    │         │
│  │  • count_     │                                    │         │
│  │    parameters │                                    │         │
│  └───────┬───────┘                                    │         │
│          │                                            │         │
│          │ build_head(in_features)                     │         │
│          │ freeze_backbone(model)                      │         │
│          ▼                                            │         │
│  ┌──────────────────────────────────────────────────┐│         │
│  │                                                  ││         │
│  │  ┌──────────────────┐   ┌──────────────────┐     ││         │
│  │  │ resnet_model.py  │   │ efficientnet_    │     ││         │
│  │  │                  │   │ model.py         │     ││         │
│  │  │ get_resnet152_   │   │                  │     ││         │
│  │  │ model()          │   │ get_efficientnet_│     ││         │
│  │  │                  │   │ b2_model()       │     ││         │
│  │  │ • ResNet-152     │   │                  │     ││         │
│  │  │ • in_features:   │   │ • EfficientNet-  │     ││         │
│  │  │   2048           │   │   B2             │     ││         │
│  │  │ • torchvision    │   │ • in_features:   │     ││         │
│  │  │   pretrained     │   │   1408           │     ││         │
│  │  └──────────────────┘   └──────────────────┘     ││         │
│  │                                                  ││         │
│  │  ┌──────────────────┐                            ││         │
│  │  │ xception_model.py│                            ││         │
│  │  │                  │                            ││         │
│  │  │ get_xception_    │                            ││         │
│  │  │ model()          │                            ││         │
│  │  │                  │                            ││         │
│  │  │ • Xception       │                            ││         │
│  │  │ • in_features:   │                            ││         │
│  │  │   2048           │                            ││         │
│  │  │ • timm library   │                            ││         │
│  │  └──────────────────┘                            ││         │
│  │                                                  ││         │
│  └──────────────────────────────────────────────────┘│         │
│                                                      │         │
└──────────────────────────────────────────────────────┴─────────┘
```

## Module Details

### base.py (12-130)
Core building blocks shared across all model architectures.

| Function | Line | Purpose |
|----------|------|---------|
| `build_head(in_features)` | 12-42 | Creates classification head with dropout(0.2), Linear(1024), BatchNorm1d, ReLU — returns raw logits for BCEWithLogitsLoss |
| `freeze_backbone(model)` | 44-65 | Sets requires_grad=False on all model parameters |
| `unfreeze_backbone(model)` | 67-86 | Sets requires_grad=True on all model parameters |
| `get_trainable_params(model)` | 88-106 | Returns list of trainable parameters |
| `count_parameters(model)` | 108-130 | Returns (total, trainable) parameter counts |

### Model Functions

Each model function follows the same pattern:
1. Load pretrained ImageNet weights
2. Freeze all base model layers
3. Replace classifier/fc with `build_head(in_features)`

| File | Function | Architecture | Library | in_features |
|------|----------|--------------|---------|-------------|
| `resnet_model.py` | `get_resnet152_model()` | ResNet-152 | torchvision | 2048 |
| `efficientnet_model.py` | `get_efficientnet_b2_model()` | EfficientNet-B2 | torchvision | 1408 |
| `xception_model.py` | `get_xception_model()` | Xception | timm | 2048 |

## Classification Head Architecture

```
Input (batch, in_features)
         │
         ▼
    ┌──────────┐
    │ Dropout  │ p=0.2
    │  (0.2)   │
    └────┬─────┘
         ▼
    ┌──────────┐
    │ Linear   │ in_features → 1024
    └────┬─────┘
         ▼
    ┌──────────┐
    │BatchNorm1│ 1024
    └────┬─────┘
         ▼
    ┌──────────┐
    │  ReLU    │
    └────┬─────┘
         ▼
    ┌──────────┐
    │ Dropout  │ p=0.2
    │  (0.2)   │
    └────┬─────┘
         ▼
    ┌──────────┐
    │ Linear   │ 1024 → 1024
    └────┬─────┘
         ▼
    ┌──────────┐
    │BatchNorm1│ 1024
    └────┬─────┘
         ▼
    ┌──────────┐
    │  ReLU    │
    └────┬─────┘
         ▼
    ┌──────────┐
    │ Dropout  │ p=0.2
    │  (0.2)   │
    └────┬─────┘
         ▼
    ┌──────────┐
    │ Linear   │ 1024 → 1
    └────┬─────┘
         ▼
    Raw Logits (batch, 1)
         │
    BCEWithLogitsLoss
```

## Usage Example

```python
from src.models.resnet_model import get_resnet152_model
from src.models.efficientnet_model import get_efficientnet_b2_model
from src.models.xception_model import get_xception_model

# Load any model
model = get_resnet152_model()  # or get_efficientnet_b2_model(), get_xception_model()

# Only classification head is trainable
# (backbone is frozen by default)
```
