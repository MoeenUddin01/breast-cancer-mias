# Breast Cancer Detection (MIAS Mammograms)

Codebase for working with the MIAS mammography dataset, focused on a clean data loading + preprocessing pipeline (CLAHE → resize → normalize) and scaffolding for training pretrained CNN backbones in PyTorch.

## Status

- ✅ Data loading from `Info.txt` + `.pgm` files (`src/data/loader.py`)
- ✅ Leakage-safe train/test split by image ID (`src/data/splitter.py`)
- ✅ Preprocess-and-save pipeline to generate PNGs on disk (`src/pipelines/preprocess_and_save.py`)
- 🚧 Training / evaluation pipelines are not implemented yet (several files are currently placeholders)

## Project Structure

```
.
├── dataset/
│   └── all-mias/                    # Put MIAS here (Info.txt + *.pgm)
│       └── processed/               # Created by preprocess_and_save.py
├── docs/
│   └── preprocess_pipeline_diagram.md
├── outputs/
│   ├── models/
│   ├── plots/
│   └── reports/
├── src/
│   ├── data/
│   │   ├── loader.py                # Parse Info.txt + load PGM images
│   │   ├── splitter.py              # Train/test split by image_id
│   │   ├── preprocessor.py          # CLAHE + preprocessing helpers
│   │   ├── augmentor.py             # Torchvision transform factories
│   │   └── dataset.py               # PyTorch Dataset wrapper
│   ├── models/                      # Backbone wrappers (WIP)
│   ├── pipelines/
│   │   └── preprocess_and_save.py   # Generate processed PNG dataset
│   └── utils/
│       ├── config.yaml              # Central configuration
│       ├── config_loader.py         # Loads config.yaml into constants
│       └── helpers.py               # Seeding, output dirs, reports
└── README.md
```

## Setup

Python: 3.12+ (see `.python-version`).

This repository currently does not pin dependencies in `requirements.txt` or `pyproject.toml`. Install the core runtime deps manually:

```bash
python3 -m venv .venv
source .venv/bin/activate

# Core deps used by the implemented pipeline
pip install numpy opencv-python pillow pyyaml scikit-learn tqdm

# If/when you use the model code:
pip install torch torchvision
pip install timm  # required for Xception
```

## Data Layout (MIAS)

Place the MIAS files in:

```
dataset/all-mias/
  Info.txt
  mdb001.pgm
  mdb002.pgm
  ...
```

Notes:
- `src/data/loader.py` currently keeps only samples labeled `B` (benign → `0`) and `M` (malignant → `1`). Other labels are skipped.

## Preprocess + Save (Recommended First Step)

This generates a disk-backed processed dataset:

```bash
python3 -m src.pipelines.preprocess_and_save
```

Outputs:
- `dataset/all-mias/processed/train/*.png`
- `dataset/all-mias/processed/test/*.png`

Pipeline details: `docs/preprocess_pipeline_diagram.md`.

## Configuration

Edit `src/utils/config.yaml` to change:
- data locations (`data.dir`, `data.processed_dir`)
- output locations (`output.*`)
- image size (`image.size`)
- split/training knobs (`training.*`)

The values are loaded at import-time via `src/utils/config_loader.py`.
