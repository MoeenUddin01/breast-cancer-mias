"""Main entry point for MIAS breast cancer detection training pipeline.

Orchestrates data preprocessing, model training for multiple architectures,
evaluation, and comparison of results across all models.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from src.evaluation.visualizer import plot_model_comparison
from src.pipelines.efficientnet_model_training import run_efficientnet_pipeline
from src.pipelines.preprocessing import run_preprocessing_pipeline
from src.pipelines.resnet_model_training import run_resnet_pipeline
from src.pipelines.xception_model_training import run_xception_pipeline
from src.utils import config_loader as config
from src.utils.helpers import create_directories, seed_everything, setup_mlflow_dagshub

if TYPE_CHECKING:
    pass


def _print_comparison_table(results: dict[str, dict[str, float]]) -> None:
    """Print a formatted comparison table of model results.

    Args:
        results: Dictionary mapping model names to their metrics dictionaries.

    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Model':<20} {'Accuracy':>12} {'AUC-ROC':>12} {'F1-Score':>12}")
    print("-" * 70)

    for model_name, metrics in results.items():
        accuracy = metrics.get("accuracy", 0.0)
        auc = metrics.get("auc_roc", 0.0)
        f1 = metrics.get("f1_score", 0.0)
        print(
            f"{model_name:<20} {accuracy:>12.4f} {auc:>12.4f} {f1:>12.4f}"
        )

    print("=" * 70)

    # Highlight best model by AUC
    if results:
        best_model = max(results.items(), key=lambda x: x[1].get("auc_roc", 0.0))
        print(f"\nBest Model (by AUC-ROC): {best_model[0]}")
        print(f"  AUC-ROC: {best_model[1].get('auc_roc', 0.0):.4f}")
        print(f"  Accuracy: {best_model[1].get('accuracy', 0.0):.4f}")
        print(f"  F1-Score: {best_model[1].get('f1_score', 0.0):.4f}")


def main() -> int:
    """Run the complete breast cancer detection training pipeline.

    Returns:
        Exit code: 0 for success, 1 for failure.

    """
    try:
        # Set random seeds for reproducibility
        print("Setting random seeds for reproducibility...")
        seed_everything(config.SEED)

        # Initialize MLflow tracking on DagsHub
        print("Initializing MLflow tracking on DagsHub...")
        setup_mlflow_dagshub()

        # Create output directories
        print("Creating output directories...")
        create_directories()

        # Run preprocessing pipeline
        print("\n" + "=" * 70)
        print("RUNNING PREPROCESSING PIPELINE")
        print("=" * 70)
        try:
            train_loader, test_loader = run_preprocessing_pipeline(config)
        except Exception as e:
            print(f"[ERROR] Preprocessing pipeline failed: {e}")
            return 1

        # Dictionary to store results from all models
        all_results: dict[str, dict[str, float]] = {}

        # Run Xception pipeline
        print("\n" + "=" * 70)
        print("TRAINING XCEPTION MODEL")
        print("=" * 70)
        try:
            xception_metrics, _ = run_xception_pipeline(
                train_loader, test_loader, config
            )
            all_results["Xception"] = xception_metrics
        except Exception as e:
            print(f"[ERROR] Xception pipeline failed: {e}")
            all_results["Xception"] = {
                "accuracy": 0.0,
                "auc_roc": 0.0,
                "f1_score": 0.0,
            }

        # Run ResNet-152 pipeline
        print("\n" + "=" * 70)
        print("TRAINING RESNET-152 MODEL")
        print("=" * 70)
        try:
            resnet_metrics, _ = run_resnet_pipeline(
                train_loader, test_loader, config
            )
            all_results["ResNet-152"] = resnet_metrics
        except Exception as e:
            print(f"[ERROR] ResNet-152 pipeline failed: {e}")
            all_results["ResNet-152"] = {
                "accuracy": 0.0,
                "auc_roc": 0.0,
                "f1_score": 0.0,
            }

        # Run EfficientNet-B2 pipeline
        print("\n" + "=" * 70)
        print("TRAINING EFFICIENTNET-B2 MODEL")
        print("=" * 70)
        try:
            efficientnet_metrics, _ = run_efficientnet_pipeline(
                train_loader, test_loader, config
            )
            all_results["EfficientNet-B2"] = efficientnet_metrics
        except Exception as e:
            print(f"[ERROR] EfficientNet-B2 pipeline failed: {e}")
            all_results["EfficientNet-B2"] = {
                "accuracy": 0.0,
                "auc_roc": 0.0,
                "f1_score": 0.0,
            }

        # Plot model comparison
        print("\n" + "=" * 70)
        print("GENERATING MODEL COMPARISON PLOT")
        print("=" * 70)
        try:
            model_names = list(all_results.keys())
            accuracies = [all_results[m]["accuracy"] for m in model_names]
            aucs = [all_results[m]["auc_roc"] for m in model_names]
            plot_model_comparison(model_names, accuracies, aucs)
        except Exception as e:
            print(f"[WARNING] Failed to generate model comparison plot: {e}")

        # Print final comparison table
        _print_comparison_table(all_results)

        print("\n" + "=" * 70)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)
        return 0

    except KeyboardInterrupt:
        print("\n[WARNING] Training interrupted by user.")
        return 130
    except Exception as e:
        print(f"\n[ERROR] Training pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
