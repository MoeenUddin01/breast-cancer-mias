# Breast Cancer Detection with MIAS Dataset

PyTorch implementation for binary classification of breast cancer from mammogram images using the MIAS dataset. Supports multiple pretrained CNN architectures: Xception, ResNet-152, and EfficientNet-B2.

## Project Structure

```
.
├── data/
│   └── all-mias/              # MIAS dataset (PGM images + Info.txt)
│       └── processed/         # Preprocessed images (optional)
├── outputs/
│   ├── models/                # Saved model checkpoints
│   ├── plots/                 # Training curves and visualizations
│   └── reports/               # Classification reports
├── src/
│   ├── data/
│   │   ├── loader.py          # Load MIAS data from Info.txt
│   │   ├── splitter.py        # Train/test split by image ID
│   │   ├── preprocessor.py    # CLAHE + preprocessing pipeline
│   │   ├── augmentor.py       # Torchvision transforms
│   │   └── dataset.py         # PyTorch Dataset class
│   ├── pipelines/
│   │   ├── preprocess_and_save.py  # Optional: save preprocessed images
│   │   ├── xception_model_training.py
│   │   ├── resnet_model_training.py
│   │   └── efficientnet_model_training.py
│   └── utils/
│       ├── config.py          # Configuration constants
│       └── helpers.py         # Utility functions
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Preparation

1. Download MIAS dataset and place in `data/all-mias/`:
   - All `.pgm` mammogram images
   - `Info.txt` metadata file

2. (Optional) Preprocess and save images to disk for faster loading:
   ```bash
   python3 -m src.pipelines.preprocess_and_save
   ```

## Usage

### Train a Model

```bash
# Xception
python3 -m src.pipelines.xception_model_training

# ResNet-152
python3 -m src.pipelines.resnet_model_training

# EfficientNet-B2
python3 -m src.pipelines.efficientnet_model_training
```

### Model Comparison

Trained models, plots, and reports are saved to `outputs/`:
- Models: `outputs/models/{model_name}_best.pth`
- Plots: `outputs/plots/`
- Reports: `outputs/reports/`

## Data Pipeline

| Stage | File | Purpose |
|-------|------|---------|
| Load | `loader.py` | Parse Info.txt, load PGM images as numpy arrays |
| Split | `splitter.py` | Stratified train/test split by unique image ID (no leakage) |
| Preprocess | `preprocessor.py` | CLAHE → 3-channel → resize → normalize |
| Augment | `augmentor.py` | Training: flips, rotation, color jitter. Test: resize only |
| Dataset | `dataset.py` | PyTorch Dataset applying preprocessor + transforms |

## Key Components

### Configuration (`src/utils/config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMAGE_SIZE` | (224, 224) | Input image dimensions |
| `BATCH_SIZE` | 32 | Training batch size |
| `EPOCHS` | 40 | Maximum training epochs |
| `LEARNING_RATE` | 1e-4 | Optimizer learning rate |
| `PATIENCE` | 6 | Early stopping patience |
| `TEST_SIZE` | 0.15 | Fraction for test split |
| `SEED` | 42 | Random seed for reproducibility |

### Helpers (`src/utils/helpers.py`)

- `seed_everything(seed)` - Set random seeds for reproducibility
- `create_directories()` - Create output directories
- `get_device()` - Return CUDA if available, else CPU
- `save_report(report_dict, model_name)` - Save classification report

## Preprocessing Pipeline

1. **CLAHE**: Contrast Limited Adaptive Histogram Equalization (`clipLimit=2.0`, `tileGridSize=(8,8)`)
2. **3-Channel Conversion**: Stack grayscale to RGB for pretrained models
3. **Resize**: 224×224 using OpenCV
4. **Normalize**: Scale to [0, 1] range
5. **ImageNet Normalization**: Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225]

## Data Augmentation (Training Only)

- Random Horizontal Flip (p=0.5)
- Random Vertical Flip (p=0.5)
- Random Rotation (±15°)
- Color Jitter (brightness, contrast, saturation, hue)

## Code Style

- PEP 8 compliant (enforced by `ruff`)
- Type annotations on all functions
- Google-style docstrings
- Maximum 88 character line length

## Linting

```bash
ruff check src/ app/
ruff format src/ app/
```
