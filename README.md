# Breast Cancer Detection (BreakHis Histology)

PyTorch-based deep learning project for binary breast cancer detection from BreakHis histopathology images. Implements a complete ML pipeline with data preprocessing, patient-aware train/test splitting, class imbalance handling, model training (ResNet, EfficientNet, Xception), MLflow experiment tracking, and visualization tools.

## Features

- **Data Pipeline**: BreakHis histology RGB images (all magnifications: 40X, 100X, 200X, 400X), train on single magnification or combine all, patient-aware splitting (prevents data leakage), CLAHE-enhanced preprocessing
- **Class Imbalance Handling**: WeightedRandomSampler for balanced training batches + BCEWithLogitsLoss with computed pos_weight
- **Models**: Pretrained CNN backbones (ResNet-152, EfficientNet-B2, Xception) with custom classification heads
- **Training**: Adam optimizer, early stopping on validation AUC-ROC (primary metric for imbalanced data)
- **Evaluation**: AUC-ROC and F1 as primary metrics (accuracy is misleading with imbalanced data), classification reports, confusion matrices
- **Artifacts**: Automatic model checkpointing and results saving to per-model folders
- **Experiment Tracking**: MLflow integration with DagHub for metrics, parameters, artifacts, and model registry
- **Kaggle Support**: Ready-to-use `kaggle_train.py` for training on Kaggle GPUs

## Class Imbalance Handling

The BreakHis dataset has a significant class imbalance (~74% malignant, ~26% benign). Two approaches are combined:

### 1. WeightedRandomSampler

Balances training batches by oversampling the minority class:

```python
from torch.utils.data import WeightedRandomSampler

# Compute class weights
class_counts = torch.bincount(torch.tensor(train_labels))
class_weights = 1.0 / class_counts.float()
sample_weights = [class_weights[label] for label in train_labels]

# Create sampler
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# Use in DataLoader (replaces shuffle=True)
train_loader = DataLoader(..., sampler=sampler, shuffle=False)
```

### 2. BCEWithLogitsLoss with pos_weight

Down-weights the majority class loss contribution:

```python
# pos_weight = num_negatives / num_positives (~0.34 for BreakHis)
pos_weight = torch.tensor([num_negatives / num_positives]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

## Project Structure

```text
.
├── artifacts/                       # Trained models and results
│   ├── resnet/
│   │   ├── resnet_final.pth
│   │   ├── resnet_best.pth
│   │   └── results.json
│   ├── efficientnet/
│   └── xception/
├── data/
│   └── raw/                         # BreakHis dataset (not tracked by git)
│       └── BreaKHis_v1/             # Kaggle BreakHis dataset structure
│           └── histology_slides/
│               └── breast/
│                   ├── benign/SOB/  # Benign histology images (400X)
│                   └── malignant/SOB/  # Malignant histology images (400X)
├── outputs/
│   ├── models/                      # Additional model checkpoints
│   ├── plots/                       # Training curves, confusion matrices
│   └── reports/                     # Evaluation reports (.txt)
├── docs/                            # Architecture diagrams
│   ├── models_architecture_diagram.md
│   ├── preprocess_pipeline_diagram.md
│   └── training_evaluation_pipeline_diagram.md
├── verify_breakhis.py               # Dataset verification and statistics
├── src/
│   ├── data/                        # Data loading and preprocessing
│   │   ├── loader.py                # Load BreakHis PNG images, extract patient_id
│   │   ├── splitter.py              # Train/test split by patient_id (prevents leakage)
│   │   ├── preprocessor.py          # RGB CLAHE (LAB color space) + preprocessing
│   │   ├── augmentor.py             # Torchvision transforms
│   │   └── dataset.py               # PyTorch Dataset wrapper
│   ├── models/                      # Model architectures
│   │   ├── base.py                  # Classification head + utilities
│   │   ├── resnet_model.py
│   │   ├── efficientnet_model.py
│   │   └── xception_model.py
│   ├── training/                    # Training loop and utilities
│   │   ├── trainer.py               # Main training with pos_weight computation
│   │   ├── validator.py             # Validation metrics (AUC-ROC focus)
│   │   └── callbacks.py             # Early stopping with monitor parameter
│   ├── evaluation/                  # Evaluation and visualization
│   │   ├── evaluator.py             # Metrics computation
│   │   └── visualizer.py            # Plotting functions
│   ├── pipelines/                   # End-to-end training pipelines
│   │   ├── preprocessing.py         # Data preprocessing pipeline
│   │   ├── xception_model_training.py
│   │   ├── resnet_model_training.py
│   │   ├── efficientnet_model_training.py
│   │   └── preprocess_and_save.py   # Generate processed PNG dataset
│   ├── utils/                       # Utilities and configuration
│   │   ├── helpers.py               # Seed, device, directory helpers
│   │   └── config_loader.py         # YAML configuration loader
│   └── main.py                      # Main entry point for full pipeline
└── README.md
```

## Setup

Python: 3.12+ (see `.python-version`).

```bash
python3 -m venv .venv
source .venv/bin/activate

# Core dependencies
pip install numpy opencv-python pillow pyyaml scikit-learn tqdm matplotlib seaborn

# PyTorch (adjust for your CUDA version)
pip install torch torchvision

# Model backbones
pip install timm  # Required for Xception
```

## Data Layout (BreakHis)

Download the BreakHis dataset from Kaggle and place it at:

```text
/kaggle/input/breakhis/BreaKHis_v1/histology_slides/breast/
  ├── benign/
  │   └── SOB/
  │       └── adenosis/
  │           ├── SOB_B_A_14-4659-40X/    # 40X magnification
  │           │   └── 001.png
  │           ├── SOB_B_A_14-4659-100X/   # 100X magnification
  │           ├── SOB_B_A_14-4659-200X/   # 200X magnification
  │           └── SOB_B_A_14-4659-400X/   # 400X magnification
  │               └── 001.png
  └── malignant/
      └── SOB/
          └── ductal_carcinoma/
              └── SOB_M_DC_14-2985-400X/
                  └── 001.png
                  └── ...
```

**Dataset Characteristics:**
- **Images**: RGB PNG format (700x460 pixels)
- **Magnifications**: 40X, 100X, 200X, 400X (selectable or combine all)
- **Classes**: Benign (label 0) / Malignant (label 1)
- **Total images**: ~8,000+ images across all magnifications
- **Patients**: ~82 unique patients (all magnifications combined)
- **Patient IDs**: Extracted from filenames (e.g., `SOB_B_TA-14-4659-400X-001.png` → patient `14-4659`)
- **Splitting**: By patient ID (not image ID) to prevent data leakage
- **Class Imbalance**: ~2.4:1 ratio (malignant outnumber benign)

## Usage

### Quick Start: Run Full Pipeline

Run all preprocessing and model training with a single command:

```bash
python3 -m src.main
```

This will:

1. Set random seeds for reproducibility
2. Create output directories
3. Preprocess BreakHis data (load images from all magnifications, patient-aware split, CLAHE, create DataLoaders)
4. Train Xception, ResNet-152, and EfficientNet-B2 models
5. Evaluate each model and generate individual reports/plots
6. Create a comparison plot across all models
7. Print a final comparison table to console

### Verify Dataset

Check your BreakHis dataset structure and counts:

```bash
python verify_breakhis.py
```

Output shows image counts per magnification, total images, unique patients, and dataset projections after augmentation.

### Kaggle Notebook Training

Train all 3 models on Kaggle GPUs with MLflow tracking:

**Cell 1:** Install dependencies
```python
!pip install -q torch torchvision timm mlflow dagshub scikit-learn opencv-python tqdm matplotlib seaborn
```

**Cell 2:** Setup secrets and clone repo
```python
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
os.environ["DAGSHUB_TOKEN"] = secrets.get_secret("DAGSHUB_TOKEN")
os.environ["DAGSHUB_USER_TOKEN"] = secrets.get_secret("DAGSHUB_TOKEN")
os.environ["MLFLOW_TRACKING_USERNAME"] = secrets.get_secret("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = secrets.get_secret("MLFLOW_TRACKING_PASSWORD")

import os
os.chdir("/tmp")
!rm -rf /kaggle/working/breast-cancer-mias
!git clone https://github.com/MoeenUddin01/breast-cancer-mias.git /kaggle/working/breast-cancer-mias

import sys
os.chdir("/kaggle/working/breast-cancer-mias")
sys.path.insert(0, "/kaggle/working/breast-cancer-mias")
```

**Cell 3:** Run training
```python
exec(open("kaggle_train.py").read(), globals())
```

**Required Kaggle Secrets:**
- `DAGSHUB_TOKEN` - Your DagHub personal access token
- `MLFLOW_TRACKING_USERNAME` - Your DagHub username
- `MLFLOW_TRACKING_PASSWORD` - Same as DAGSHUB_TOKEN

### Step-by-Step Usage

#### 1. Preprocess and Save Data

Generate processed PNG dataset:

```bash
python3 -m src.pipelines.preprocess_and_save
```

Outputs:

- `dataset/all-mias/processed/train/*.png`
- `dataset/all-mias/processed/test/*.png`

#### 2. Preprocessing Pipeline (In-Memory)

Use the preprocessing pipeline to get DataLoaders directly:

```python
from types import SimpleNamespace
from src.pipelines.preprocessing import run_preprocessing_pipeline

config = SimpleNamespace(
    DATA_DIR="dataset/all-mias",
    TEST_SIZE=0.2,
    SEED=42,
    BATCH_SIZE=16,
    NUM_WORKERS=4,
    IMAGE_SIZE=(224, 224),
)

train_loader, test_loader = run_preprocessing_pipeline(config)
```

#### 3. Train Individual Models

Train a specific model using its dedicated pipeline:

```python
from types import SimpleNamespace
from src.pipelines.resnet_model_training import run_resnet_pipeline
from src.pipelines.xception_model_training import run_xception_pipeline
from src.pipelines.efficientnet_model_training import run_efficientnet_pipeline

config = SimpleNamespace(
    DEVICE="cuda",
    LEARNING_RATE=1e-4,
    EPOCHS=50,
    PATIENCE=10,
)

# ResNet-152
metrics, history = run_resnet_pipeline(train_loader, test_loader, config)

# Xception
metrics, history = run_xception_pipeline(train_loader, test_loader, config)

# EfficientNet-B2
metrics, history = run_efficientnet_pipeline(train_loader, test_loader, config)
```

#### 4. Evaluate a Model

```python
from src.evaluation.evaluator import evaluate
from src.evaluation.visualizer import plot_confusion_matrix, plot_training_history

# Compute metrics and save report
results = evaluate(model, test_loader, device, "resnet")

# Generate plots
plot_training_history(history, "resnet")
plot_confusion_matrix(y_true, y_pred, "resnet")
```

Outputs:
- `outputs/reports/resnet_report.txt`: Classification report
- `outputs/plots/resnet_history.png`: Training curves
- `outputs/plots/resnet_confusion_matrix.png`: Confusion matrix

#### 5. Compare Models

```python
from src.evaluation.visualizer import plot_model_comparison

plot_model_comparison(
    model_names=["ResNet", "EfficientNet", "Xception"],
    accuracies=[0.85, 0.88, 0.82],
    aucs=[0.91, 0.93, 0.89],
)
```

Output: `outputs/plots/model_comparison.png`

#### 6. MLflow Experiment Tracking

All training runs are logged to MLflow with DagHub integration:

```python
import mlflow
import dagshub

# Initialize DagHub MLflow tracking
dagshub.init(
    repo_owner="MoeenUddin01",
    repo_name="breast-cancer-mias",
    mlflow=True,
)

# Training metrics are automatically logged:
# - train_loss, val_loss, accuracy, auc, f1, recall
# - confusion matrices, training curves
# - model artifacts and checkpoints
```

View experiments at: https://dagshub.com/MoeenUddin01/breast-cancer-mias/experiments

#### 7. Streamlit Web Demo

Run the interactive web app for image upload and prediction:

```bash
streamlit run app/streamlit.py
```

Features:
- Upload mammogram images (PNG, JPG, PGM)
- Select between trained models (ResNet, EfficientNet, Xception)
- Real-time prediction with confidence scores
- Visualize predictions with class probabilities

#### 8. FastAPI REST API

Start the REST API server for programmatic inference:

```bash
uvicorn app.main:app --reload
```

Endpoints:
- `POST /v1/predict` - Predict on uploaded image
- `GET /v1/health` - Health check

Example usage:
```bash
curl -X POST "http://localhost:8000/v1/predict" \
  -H "accept: application/json" \
  -F "file=@mdb001.pgm" \
  -F "model_name=resnet"
```

## Configuration

Configuration is loaded from `src/utils/config.yaml`. Key parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `DATA_DIR` | str | Path to BreakHis dataset directory |
| `TEST_SIZE` | float | Fraction of data for test set (0.0-1.0) |
| `SEED` | int | Random seed for reproducibility |
| `BATCH_SIZE` | int | DataLoader batch size |
| `NUM_WORKERS` | int | DataLoader worker processes |
| `IMAGE_SIZE` | tuple | Target image size (H, W) |
| `LEARNING_RATE` | float | Adam optimizer learning rate |
| `EPOCHS` | int | Maximum training epochs |
| `PATIENCE` | int | Early stopping patience (epochs) |
| `DEVICE` | str | Training device (cuda/cpu) |

### Training-Specific Config

The `train()` function expects a config object with:

| Parameter | Type | Description |
|-----------|------|-------------|
| `LEARNING_RATE` | float | Adam optimizer learning rate |
| `EPOCHS` | int | Maximum training epochs |
| `PATIENCE` | int | Early stopping patience (epochs) |
| `DEVICE` | torch.device | Training device (cuda/cpu) |

## Key Modules

| Module | Purpose |
|--------|---------|
| `verify_breakhis.py` | **Dataset verification** - counts images per magnification, validates patient counts |
| `src/main.py` | **Entry point** - runs full pipeline with all models |
| `src/pipelines/preprocessing.py` | End-to-end data preprocessing pipeline |
| `src/pipelines/resnet_model_training.py` | ResNet-152 training pipeline |
| `src/pipelines/xception_model_training.py` | Xception training pipeline |
| `src/pipelines/efficientnet_model_training.py` | EfficientNet-B2 training pipeline |
| `src/training/trainer.py` | Main training loop with validation |
| `src/training/validator.py` | Validation metrics (loss, accuracy, AUC) |
| `src/training/callbacks.py` | EarlyStopping with model checkpointing |
| `src/evaluation/evaluator.py` | Comprehensive evaluation metrics |
| `src/evaluation/visualizer.py` | Training curves, confusion matrices, comparisons |
| `src/models/base.py` | Shared classification head and utilities |
| `src/utils/helpers.py` | Seed setting, device selection, directory creation |

## Output Locations

| Type | Path |
|------|------|
| Trained models | `artifacts/{model_name}/` |
| Model checkpoints | `artifacts/{model_name}/{model_name}_best.pth` |
| Training results | `artifacts/{model_name}/results.json` |
| Evaluation reports | `outputs/reports/{model_name}_report.txt` |
| Plots | `outputs/plots/{model_name}_*.png` |
| Model comparison | `outputs/plots/model_comparison.png` |

## Documentation

Architecture diagrams with file references and line numbers:

| Document | Content |
|----------|---------|
| `docs/preprocess_pipeline_diagram.md` | Data loading, CLAHE preprocessing, train/test split flow |
| `docs/training_evaluation_pipeline_diagram.md` | Training loop, validation, early stopping, evaluation |
| `docs/models_architecture_diagram.md` | Model structure, classification head, base utilities |
