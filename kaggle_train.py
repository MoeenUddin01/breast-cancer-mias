"""Kaggle training script for MIAS breast cancer detection models.

Usage on Kaggle:
    1. Upload this repository as a Kaggle Dataset
    2. Create a new notebook and add the dataset
    3. Set the MODEL_NAME environment variable in the Kaggle UI:
       - "xception" for Xception model
       - "resnet" for ResNet-152 model
       - "efficientnet" for EfficientNet-B2 model
    4. Run the notebook

Example:
    import os
    os.environ["MODEL_NAME"] = "resnet"
    %run /kaggle/input/breast-cancer-mias/kaggle_train.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add src to Python path for imports
# Adjust this path based on where you upload the dataset in Kaggle
KAGGLE_DATASET_PATH = "/kaggle/input/breast-cancer-mias"
sys.path.insert(0, str(KAGGLE_DATASET_PATH))

from src.pipelines.efficientnet_model_training import run_efficientnet_pipeline
from src.pipelines.preprocessing import run_preprocessing_pipeline
from src.pipelines.resnet_model_training import run_resnet_pipeline
from src.pipelines.xception_model_training import run_xception_pipeline
from src.utils import config_loader as config
from src.utils.helpers import seed_everything, setup_mlflow_dagshub


def main() -> int:
    """Run training for a single model on Kaggle.

    Returns:
        Exit code: 0 for success, 1 for failure.

    """
    # Get model name from environment variable (set in Kaggle UI)
    model_name = os.environ.get("MODEL_NAME", "resnet").lower()

    print("=" * 70)
    print(f"KAGGLE TRAINING - MODEL: {model_name.upper()}")
    print("=" * 70)

    # Validate model name
    valid_models = ["xception", "resnet", "efficientnet"]
    if model_name not in valid_models:
        print(f"[ERROR] Invalid MODEL_NAME: {model_name}")
        print(f"Valid options: {', '.join(valid_models)}")
        return 1

    try:
        # Set random seeds for reproducibility
        print("\nSetting random seeds for reproducibility...")
        seed_everything(config.SEED)

        # Initialize MLflow tracking on DagsHub
        print("Initializing MLflow tracking on DagsHub...")
        setup_mlflow_dagshub()

        # Run preprocessing pipeline
        print("\n" + "=" * 70)
        print("RUNNING PREPROCESSING PIPELINE")
        print("=" * 70)
        train_loader, test_loader = run_preprocessing_pipeline(config)

        # Map model names to pipeline functions
        pipelines = {
            "xception": run_xception_pipeline,
            "resnet": run_resnet_pipeline,
            "efficientnet": run_efficientnet_pipeline,
        }

        # Run the selected model pipeline
        print("\n" + "=" * 70)
        print(f"TRAINING {model_name.upper()} MODEL")
        print("=" * 70)

        run_pipeline = pipelines[model_name]
        metrics, _ = run_pipeline(train_loader, test_loader, config)

        # Print final results
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Accuracy:  {metrics.get('accuracy', 0.0):.4f}")
        print(f"AUC-ROC:   {metrics.get('auc_roc', 0.0):.4f}")
        print(f"F1-Score:  {metrics.get('f1_score', 0.0):.4f}")
        print(f"Precision: {metrics.get('precision', 0.0):.4f}")
        print(f"Recall:    {metrics.get('recall', 0.0):.4f}")
        print("=" * 70)

        return 0

    except KeyboardInterrupt:
        print("\n[WARNING] Training interrupted by user.")
        return 130
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
