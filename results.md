# Model Results & Comparison

## Overview

This document presents comprehensive performance results for CNN and Vision Transformer models trained on the BreakHis breast cancer histology dataset (7,909 images, 324 patients).

## Dataset Information

- **Dataset**: BreakHis Breast Cancer Histology
- **Total Images**: 7,909
- **Patients**: 324
- **Split**: 85% train / 15% test (by patient ID)
- **Classes**: Benign vs Malignant
- **GPU**: Tesla T4

## CNN Models Performance

| Model | AUC-ROC | Accuracy | F1 Score | Training Time |
|-------|---------|----------|----------|---------------|
| **EfficientNetB2** | 0.9862 | 94.93% | 0.9629 | 12.3 min |
| ResNet152 | 0.9780 | 94.04% | 0.9571 | 25.3 min |
| Xception | 0.8827 | 78.18% | 0.8259 | 12.0 min |

**Best CNN Model**: EfficientNetB2 achieves the highest AUC-ROC (0.9862) with the fastest training time (12.3 minutes).

## Vision Transformer (DeiT) Performance

| Model | AUC-ROC | Accuracy | F1 Score | Training Time | Parameters |
|-------|---------|----------|----------|---------------|------------|
| **DeiT-Base Distilled** | 0.9721 | 95.01% | 0.9640 | 108.4 min | 86.7M |
| DeiT-Small Distilled | 0.9811 | 93.32% | 0.9510 | 68.0 min | 22.0M |

## Key Findings

- DeiT-Base achieves the **highest accuracy (95.01%)** and **highest F1 score (0.9640)** among all models
- DeiT-Base does NOT beat EfficientNetB2 in AUC-ROC (0.9721 vs 0.9862)
- Transformers require significantly more training time (9× slower than EfficientNetB2)
- Transformers need larger datasets to fully outperform CNNs due to lower inductive bias
- DeiT-Small achieved AUC 0.9811 (closer to CNNs) with faster training (68 min)

## Comparison Analysis

### DeiT-Base vs EfficientNetB2

| Metric | DeiT-Base | EfficientNetB2 | Difference |
|--------|-----------|----------------|------------|
| AUC-ROC | 0.9721 | 0.9862 | -0.0141 (❌) |
| Accuracy | 95.01% | 94.93% | +0.08% (✅) |
| F1 Score | 0.9640 | 0.9629 | +0.0011 (✅) |
| Training Time | 108.4 min | 12.3 min | +96.1 min (❌) |

### DeiT-Base vs ResNet152

| Metric | DeiT-Base | ResNet152 | Difference |
|--------|-----------|-----------|------------|
| AUC-ROC | 0.9721 | 0.9780 | -0.0059 (❌) |
| Accuracy | 95.01% | 94.04% | +0.97% (✅) |
| F1 Score | 0.9640 | 0.9571 | +0.0069 (✅) |
| Training Time | 108.4 min | 25.3 min | +83.1 min (❌) |

### DeiT-Base vs Xception

| Metric | DeiT-Base | Xception | Difference |
|--------|-----------|----------|------------|
| AUC-ROC | 0.9721 | 0.8827 | +0.0894 (✅) |
| Accuracy | 95.01% | 78.18% | +16.83% (✅) |
| F1 Score | 0.9640 | 0.8259 | +0.1381 (✅) |
| Training Time | 108.4 min | 12.0 min | +96.4 min (❌) |

## Conclusion

For this small medical imaging dataset (7,909 images), **CNNs remain superior** due to:

1. **Stronger inductive bias** - Convolutional layers have built-in spatial structure understanding
2. **Faster training** - 9× faster than DeiT-Base
3. **Better AUC-ROC** - Primary metric for imbalanced classification

**Transformers show promise** but require:
- Larger datasets to fully leverage their capacity
- Ensemble techniques (TTA, 5-fold cross-validation)
- Higher resolution inputs (384×384)
- Stronger augmentation strategies

## Research Visualizations

All comparison charts and analysis are available in `artifacts/`:

### DeiT vs CNN Comparison
- **`artifacts/deit_vs_cnn_comparison/`**
  - `deit_vs_cnn_comparison.png` - 4-panel comparison chart
  - `all_metrics_comparison.png` - Combined metrics chart
  - `summary_of_comparison.txt` - Comprehensive analysis

### Research-Ready Charts
- **`artifacts/research/`**
  - `model_comparison_accuracy_f1.png` - Accuracy vs F1 bar chart
  - `model_comparison_accuracy_f1_line.png` - Line graph version
  - `model_comparison_all_metrics.png` - All metrics comparison
  - `radar_chart_comparison.png` - Radar/spider chart
  - `scatter_accuracy_f1_auc.png` - Scatter with AUC as point size
  - `efficiency_analysis.png` - Time vs AUC efficiency
  - `performance_tier_analysis.png` - Performance tier analysis
  - `training_time_comparison.png` - Training time comparison

### Individual Metric Charts
- **`artifacts/research/onebyone/`**
  - `01_auc_roc_comparison.png` - AUC-ROC bar chart
  - `02_accuracy_comparison.png` - Accuracy bar chart
  - `03_f1_score_comparison.png` - F1 Score bar chart
  - `04_training_time_comparison.png` - Training Time bar chart
  - `05_auc_roc_trend.png` - AUC-ROC line chart
  - `06_accuracy_trend.png` - Accuracy line chart
  - `07_f1_score_trend.png` - F1 Score line chart
  - `08_training_time_trend.png` - Training Time line chart

## Recommendations to Beat CNNs

To improve DeiT performance beyond CNNs:

1. **Test-Time Augmentation (TTA)** - Expected +0.005-0.01 AUC gain
2. **Increase Resolution to 384×384** - Expected +0.003-0.008 AUC gain
3. **Stratified 5-Fold Ensemble** - Expected +0.005-0.015 AUC gain
4. **Stronger Augmentation (MixUp, CutMix)** - Expected +0.002-0.005 AUC gain
5. **Reduce Overfitting** - Lower dropout, higher weight decay

## Training Details

### DeiT-Base Configuration
- **Model**: DeiT-Base Distilled (patch16_224)
- **Parameters**: 86.7M total
- **Epochs**: 44 total (4 frozen + 40 fine-tune)
- **Batch Size**: 16
- **Learning Rate**: Phase 1: 1e-3, Phase 2: 5e-6 (backbone), 5e-5 (head)
- **Optimizer**: AdamW
- **Loss**: BCEWithLogitsLoss with pos_weight
- **Augmentation**: 5× offline augmentation enabled

### CNN Configuration
- **Models**: Pretrained on ImageNet-1k
- **Epochs**: 50 max with early stopping
- **Batch Size**: 32
- **Learning Rate**: 1e-4
- **Optimizer**: Adam
- **Loss**: BCEWithLogitsLoss with pos_weight

---

🚨 **Research prototype. Not for clinical use.**
