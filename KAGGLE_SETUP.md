# Kaggle Setup Guide

This guide explains how to train your breast cancer detection models on Kaggle using GPUs.

## Prerequisites

1. Kaggle account (free tier has 30 hours/week of GPU)
2. MIAS dataset uploaded to Kaggle or use Kaggle's MIAS dataset
3. DagsHub account (for MLflow experiment tracking)

## Step 1: Upload This Repository as a Kaggle Dataset

1. Zip your project folder:
   ```bash
   zip -r breast-cancer-mias.zip breast-cancer-mias/ -x "*.git*" -x "*.pyc" -x "__pycache__/*" -x "outputs/*" -x "data/*"
   ```

2. Go to [Kaggle Datasets](https://www.kaggle.com/datasets)

3. Click **"New Dataset"**

4. Upload the zip file

5. Edit `kaggle.json` to update:
   - `id`: Your Kaggle username + dataset name
   - `title`: Display title for the dataset

6. **IMPORTANT**: Enable GPU in dataset settings

## Step 2: Create Kaggle Notebooks (One Per Model)

Create 3 separate notebooks - one for each model. This allows parallel training.

### Notebook 1: Xception Model

```python
import os
import sys

# Set model name
os.environ["MODEL_NAME"] = "xception"

# Add the dataset to path
sys.path.insert(0, "/kaggle/input/breast-cancer-mias")

# Run training
exec(open("/kaggle/input/breast-cancer-mias/kaggle_train.py").read())
```

### Notebook 2: ResNet-152 Model

```python
import os
import sys

os.environ["MODEL_NAME"] = "resnet"
sys.path.insert(0, "/kaggle/input/breast-cancer-mias")

exec(open("/kaggle/input/breast-cancer-mias/kaggle_train.py").read())
```

### Notebook 3: EfficientNet-B2 Model

```python
import os
import sys

os.environ["MODEL_NAME"] = "efficientnet"
sys.path.insert(0, "/kaggle/input/breast-cancer-mias")

exec(open("/kaggle/input/breast-cancer-mias/kaggle_train.py").read())
```

## Step 3: Configure DagsHub (MLflow Tracking)

Before running notebooks, update `src/utils/config.py`:

```python
DAGSHUB_REPO_OWNER = "your-dagshub-username"  # Replace with yours
DAGSHUB_REPO_NAME = "breast-cancer-mias"
EXPERIMENT_NAME = "MIAS_Breast_Cancer_Detection"
```

## Step 4: Run Training

1. Go to your notebook
2. In right panel → **Notebook Options**:
   - **Accelerator**: Select GPU T4 or P100
   - **Internet**: ON (required for MLflow tracking)
3. Click **Run All**
4. Repeat for all 3 notebooks

## Step 5: Monitor Results

- **Kaggle**: View training logs in the notebook output
- **DagsHub**: Go to your repository → Experiments tab to see all metrics

## Troubleshooting

### Import Errors
If you see `ModuleNotFoundError`, ensure the dataset path is correct:
```python
import os
print(os.listdir("/kaggle/input/"))  # Check your dataset name
```

### MLflow Connection Issues
- Verify DagsHub credentials in `src/utils/config.py`
- Check internet is enabled in notebook settings
- Ensure DagsHub repo is public or you have access

### Out of Memory
Reduce batch size in `src/utils/config_loader.py`:
```python
BATCH_SIZE = 16  # Try 8 if still OOM
```

### Dataset Not Found
The MIAS dataset should be uploaded separately or use Kaggle's existing MIAS dataset. Add it as a data source to your notebook:
1. Click **"Add Data"** in the notebook
2. Search for "MIAS mammography" or upload your own

## File Structure on Kaggle

```
/kaggle/input/
├── breast-cancer-mias/          # Your uploaded code
│   ├── kaggle_train.py
│   ├── src/
│   ├── requirements.txt
│   └── ...
└── mias-dataset/                # The MIAS images
    └── ...
```

## Results

After training completes:
- Models saved to: `outputs/models/`
- Plots saved to: `outputs/plots/`
- Reports saved to: `outputs/reports/`
- All metrics logged to DagsHub MLflow

## Tips

1. **Version Control**: Save notebook versions after successful runs
2. **Checkpoints**: Training automatically saves best model checkpoints
3. **Early Stopping**: Configured in `config.PATIENCE` (default: 10 epochs)
4. **GPU Hours**: Monitor usage at [Kaggle Account](https://www.kaggle.com/settings/account)

## Alternative: Sequential Training

If you prefer to train all models in one notebook:

```python
import os
import sys

sys.path.insert(0, "/kaggle/input/breast-cancer-mias")

models = ["xception", "resnet", "efficientnet"]

for model in models:
    print(f"\n{'='*50}")
    print(f"Training {model.upper()}")
    print(f"{'='*50}\n")
    
    os.environ["MODEL_NAME"] = model
    exec(open("/kaggle/input/breast-cancer-mias/kaggle_train.py").read())
```
