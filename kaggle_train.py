from __future__ import annotations

"""Kaggle notebook for MIAS breast cancer detection.

Trains Xception, ResNet152, and EfficientNetB2 on MIAS Mammography dataset.
Logs everything to DagHub via MLflow.

Run this in Kaggle with: exec(open("kaggle_train.py").read())
Prerequisites: pip install torch torchvision timm mlflow dagshub scikit-learn opencv-python tqdm matplotlib seaborn
"""

# ═══════════════════════════════════════════════════════
# CELL 1: Install dependencies (run manually in Kaggle first)
# ═══════════════════════════════════════════════════════
# !pip install -q torch torchvision timm mlflow dagshub scikit-learn opencv-python matplotlib seaborn

# ═══════════════════════════════════════════════════════
# CELL 2: Load secrets, clone repo, setup paths
# ═══════════════════════════════════════════════════════

import os
import subprocess
import sys
from pathlib import Path

# Clone the project if not already cloned
repo_path = Path("/kaggle/working/breast-cancer-mias")
if not repo_path.exists():
    subprocess.run(
        ["git", "clone", "https://github.com/MoeenUddin01/breast-cancer-mias.git", str(repo_path)],
        check=True,
        capture_output=True,
    )

os.chdir(str(repo_path))
sys.path.insert(0, str(repo_path))

print("\n✅ Repo ready at:", str(repo_path))

# ═══════════════════════════════════════════════════════
# CELL 3: BreakHis dataset configuration
# ═══════════════════════════════════════════════════════

import glob
import pathlib

print("🔍 Verifying BreakHis dataset...")

# BreakHis dataset paths
DATA_DIR = "/kaggle/input/datasets/ambarish/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/"
MAGNIFICATIONS = ["40X", "100X", "200X", "400X"]

# Verify the directory structure
benign_path_str = os.path.join(DATA_DIR, "benign/SOB/")
malignant_path_str = os.path.join(DATA_DIR, "malignant/SOB/")

print(f"\n📁 Dataset location: {DATA_DIR}")
print(f"Benign path exists: {os.path.exists(benign_path_str)}")
print(f"Malignant path exists: {os.path.exists(malignant_path_str)}")

# Convert to pathlib for rglob operations
benign_path = pathlib.Path(benign_path_str)
malignant_path = pathlib.Path(malignant_path_str)

if not benign_path.exists() or not malignant_path.exists():
    print("❌ Available inputs:")
    input_dirs = glob.glob("/kaggle/input/*")
    for d in input_dirs:
        print(f"   - {d}")
        contents = glob.glob(f"{d}/**", recursive=True)[:10]
        for c in contents:
            print(f"      {c}")
    raise RuntimeError(
        "Could not find BreakHis dataset. "
        "Please ensure the dataset is added to the notebook inputs."
    )

# Count images for all magnifications
print("\n" + "=" * 50)
print("  BREAKHIS DATASET COUNTS BY MAGNIFICATION")
print("=" * 50)

total_benign = 0
total_malignant = 0
for mag in MAGNIFICATIONS:
    benign_images = list(benign_path.rglob(f"{mag}/*.png"))
    malignant_images = list(malignant_path.rglob(f"{mag}/*.png"))
    total_benign += len(benign_images)
    total_malignant += len(malignant_images)
    print(f"  {mag:4s} → benign: {len(benign_images):4d}  malignant: {len(malignant_images):4d}")

print("-" * 50)
print(f"  Total → benign: {total_benign:4d}  malignant: {total_malignant:4d}")
print("=" * 50)
OUTPUT_DIR = "/kaggle/working/outputs/"
MODELS_DIR = "/kaggle/working/outputs/models/"
PLOTS_DIR = "/kaggle/working/outputs/plots/"
REPORTS_DIR = "/kaggle/working/outputs/reports/"

# Create output directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

print(f"\n✅ Output directories ready at: {OUTPUT_DIR}")

# ═══════════════════════════════════════════════════════
# CELL 4: Imports
# ═══════════════════════════════════════════════════════

import sys
from pathlib import Path

# Ensure src is in path (for exec() compatibility)
repo_root = Path(__file__).parent if "__file__" in dir() else Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import time
from datetime import datetime
from types import SimpleNamespace

import cv2
import dagshub
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import psutil
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc, accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve
)
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler

# Data imports
from src.data.augmentor import augment_training_data, get_test_transforms, get_train_transforms
from src.data.dataset import MIASDataset
from src.data.loader import load_data
from src.data.splitter import split_by_patient_id

# Model imports
from src.models.base import count_parameters
from src.models.efficientnet_model import get_efficientnet_b2_model
from src.models.resnet_model import get_resnet152_model
from src.models.xception_model import get_xception_model

# Training imports
from src.training.callbacks import EarlyStopping
from src.training.trainer import _compute_pos_weight
from src.training.validator import validate

# Evaluation imports
from src.evaluation import evaluator
from src.evaluation.evaluator import evaluate
from src.evaluation.visualizer import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_training_history,
)

# Utils imports
import importlib
from src.utils import config as dagshub_config
importlib.reload(dagshub_config)
from src.utils import config_loader as config
from src.utils import helpers

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print("=" * 40)
print("  TRAINING CONFIGURATION")
print("=" * 40)
for key in [
    "TRAIN_RESNET",
    "TRAIN_EFFICIENTNET",
    "TRAIN_XCEPTION",
    "EVALUATE_MODELS",
    "GENERATE_PLOTS",
    "SHOW_SUMMARY",
]:
    val = os.environ.get(key, "false")
    status = "✅ ON " if val == "true" else "⭕ OFF"
    print(f"  {status}  {key}")
print("=" * 40)

# ═══════════════════════════════════════════════════════
# CELL 5: Configuration
# ═══════════════════════════════════════════════════════

# DagHub configuration (from secrets)
DAGSHUB_REPO_OWNER = secrets.get_secret("DAGSHUB_REPO_OWNER") or "MoeenUddin01"
DAGSHUB_REPO_NAME = "breast-cancer-mias"

# Training hyperparameters
EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
PATIENCE = 6
SEED = dagshub_config.SEED

# Model training flags (set via environment variables)
TRAIN_RESNET = os.environ.get("TRAIN_RESNET", "false").lower() == "true"
TRAIN_EFFICIENTNET = os.environ.get("TRAIN_EFFICIENTNET", "false").lower() == "true"
TRAIN_XCEPTION = os.environ.get("TRAIN_XCEPTION", "false").lower() == "true"
EVALUATE_MODELS = os.environ.get("EVALUATE_MODELS", "false").lower() == "true"
GENERATE_PLOTS = os.environ.get("GENERATE_PLOTS", "false").lower() == "true"
SHOW_SUMMARY = os.environ.get("SHOW_SUMMARY", "false").lower() == "true"

print("  TRAIN_RESNET       :", TRAIN_RESNET)
print("  TRAIN_EFFICIENTNET :", TRAIN_EFFICIENTNET)
print("  TRAIN_XCEPTION     :", TRAIN_XCEPTION)
print("  EVALUATE_MODELS    :", EVALUATE_MODELS)
print("  GENERATE_PLOTS     :", GENERATE_PLOTS)
print("  SHOW_SUMMARY       :", SHOW_SUMMARY)

# Other constants
TEST_SIZE = dagshub_config.TEST_SIZE
NUM_WORKERS = 4
IMAGE_SIZE = (224, 224)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Configuration:")
print(f"  DATA_DIR: {DATA_DIR}")
print(f"  OUTPUT_DIR: {OUTPUT_DIR}")
print(f"  MODELS_DIR: {MODELS_DIR}")
print(f"  PLOTS_DIR: {PLOTS_DIR}")
print(f"  REPORTS_DIR: {REPORTS_DIR}")
print(f"  DAGSHUB_REPO_OWNER: {DAGSHUB_REPO_OWNER}")
print(f"  EPOCHS: {EPOCHS}")
print(f"  BATCH_SIZE: {BATCH_SIZE}")
print(f"  LEARNING_RATE: {LEARNING_RATE}")
print(f"  PATIENCE: {PATIENCE}")
print(f"  SEED: {SEED}")
print(f"  DEVICE: {DEVICE}")

# ═══════════════════════════════════════════════════════
# CELL 4: Seed and directory setup
# ═══════════════════════════════════════════════════════

helpers.seed_everything(SEED)

# Create directories
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)
Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)

print("✓ Random seeds set")
print("✓ Output directories created")

# ═══════════════════════════════════════════════════════
# CELL 6: DagHub + MLflow initialization
# ═══════════════════════════════════════════════════════

# Initialize DagHub (env vars already set in Cell 2 via secrets)
dagshub.init(
    repo_owner=DAGSHUB_REPO_OWNER,
    repo_name=DAGSHUB_REPO_NAME,
    mlflow=True,
)

# Set experiment
mlflow.set_experiment("MIAS_Breast_Cancer_Detection")

print(f"✓ DagHub initialized")
print(f"✓ MLflow tracking URI: {mlflow.get_tracking_uri()}")

# ═══════════════════════════════════════════════════════
# CELL 6: Load and preview data
# ═══════════════════════════════════════════════════════

# Load the dataset with all magnifications
data = load_data(DATA_DIR, magnifications=MAGNIFICATIONS)
ram = psutil.virtual_memory().used / 1e9
print(f"RAM after load_data()    : {ram:.1f} GB")

# Count classes and patients
benign_count = sum(1 for _, _, label in data if label == 0)
malignant_count = sum(1 for _, _, label in data if label == 1)
unique_patients = len({pid.split("_")[0] for pid, _, _ in data})

print(f"\nTotal samples: {len(data)}")
print(f"Benign samples: {benign_count}")
print(f"Malignant samples: {malignant_count}")
print(f"Unique patients: {unique_patients}")

# Preview 8 sample images (benign/malignant at 40X, 100X, 200X, 400X)
# Get one sample per magnification for each class
preview_samples = []
for mag in MAGNIFICATIONS:
    # Find one benign and one malignant sample for this magnification
    benign_sample = next((pid, path, lbl) for pid, path, lbl in data if lbl == 0 and mag in pid)
    malignant_sample = next((pid, path, lbl) for pid, path, lbl in data if lbl == 1 and mag in pid)
    preview_samples.extend([benign_sample, malignant_sample])

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, (patient_id, image_path, label) in enumerate(preview_samples):
    row = i % 2  # Row 0 = benign, Row 1 = malignant
    col = i // 2  # Columns 0-3 = 40X, 100X, 200X, 400X
    image_array = cv2.imread(image_path)
    if image_array is None:
        raise FileNotFoundError(f"Preview image not found: {image_path}")
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    axes[row, col].imshow(image_array)
    mag_label = MAGNIFICATIONS[col]
    label_name = "Benign" if label == 0 else "Malignant"
    axes[row, col].set_title(f"{label_name} {mag_label}\nPatient: {patient_id.split('_')[0]}")
    axes[row, col].axis("off")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/data_preview.png", dpi=150)
plt.show()
print(f"✓ Preview saved to {PLOTS_DIR}/data_preview.png")

# ═══════════════════════════════════════════════════════
# CELL 7: Split data (no patient leakage)
# ═══════════════════════════════════════════════════════

# Split by patient ID with stratification to prevent data leakage
train_data, test_data = split_by_patient_id(data, TEST_SIZE, SEED)
ram = psutil.virtual_memory().used / 1e9
print(f"RAM after split()        : {ram:.1f} GB")

print(f"\nTrain size: {len(train_data)}")
print(f"Test size: {len(test_data)}")

# Class distribution
train_benign = sum(1 for _, _, label in train_data if label == 0)
train_malignant = sum(1 for _, _, label in train_data if label == 1)
test_benign = sum(1 for _, _, label in test_data if label == 0)
test_malignant = sum(1 for _, _, label in test_data if label == 1)

print(f"\nTrain class distribution:")
print(f"  Benign: {train_benign} ({100*train_benign/len(train_data):.1f}%)")
print(f"  Malignant: {train_malignant} ({100*train_malignant/len(train_data):.1f}%)")

print(f"\nTest class distribution:")
print(f"  Benign: {test_benign} ({100*test_benign/len(test_data):.1f}%)")
print(f"  Malignant: {test_malignant} ({100*test_malignant/len(test_data):.1f}%)")

# Verify no patient leakage
train_patient_ids = {pid for pid, _, _ in train_data}
test_patient_ids = {pid for pid, _, _ in test_data}
patient_overlap = train_patient_ids & test_patient_ids
print(f"\n✓ No patient_id overlap: {len(patient_overlap) == 0}")
if patient_overlap:
    print(f"  WARNING: Found {len(patient_overlap)} overlapping patient IDs: {patient_overlap}")

# Print unique patient counts
print(f"Unique patients in train: {len(train_patient_ids)}")
print(f"Unique patients in test: {len(test_patient_ids)}")

# ═══════════════════════════════════════════════════════
# CELL 8: Preprocessing and DataLoaders
# ═══════════════════════════════════════════════════════

# Get transforms
train_transforms = get_train_transforms(IMAGE_SIZE)
test_transforms = get_test_transforms(IMAGE_SIZE)

# Create datasets
train_dataset = MIASDataset(
    data=train_data,
    transform=train_transforms,
    image_size=IMAGE_SIZE,
)
test_dataset = MIASDataset(
    data=test_data,
    transform=test_transforms,
    image_size=IMAGE_SIZE,
)

# Create DataLoaders with WeightedRandomSampler for class imbalance
# Count classes in training data for weighted sampling
train_labels = [item[2] for item in train_data]
class_counts = torch.bincount(torch.tensor(train_labels))
class_weights = 1.0 / class_counts.float()
sample_weights = [class_weights[label] for label in train_labels]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

print(f"Class counts: {class_counts.tolist()}")
print(f"Class weights: {class_weights.tolist()}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,      # Replaces shuffle=True for class balancing
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True,   # Drop last incomplete batch to avoid BatchNorm1d errors
    persistent_workers=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)

print(f"✓ Train samples  : {len(train_data)} (online aug each epoch)")
print(f"✓ Test samples   : {len(test_data)}")
print(f"✓ Train batches  : {len(train_loader)}")
print(f"✓ Test batches   : {len(test_loader)}")

# ═══════════════════════════════════════════════════════
# CELL 9: Model initialization function
# ═══════════════════════════════════════════════════════


def init_model(model_name: str) -> nn.Module:
    """Initialize a model by name.

    Args:
        model_name: One of "resnet152", "efficientnet_b2", "xception".

    Returns:
        Initialized model moved to DEVICE.

    """
    print(f"\nInitializing {model_name}...")

    if model_name == "resnet152":
        model = get_resnet152_model()
    elif model_name == "efficientnet_b2":
        model = get_efficientnet_b2_model()
    elif model_name == "xception":
        model = get_xception_model()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(DEVICE)

    total_params, trainable_params = count_parameters(model)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model


# Test model initialization
print("Model initialization function ready")

# ═══════════════════════════════════════════════════════
# CELL 10: Training loop function
# ═══════════════════════════════════════════════════════


def verbose_train(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
) -> dict[str, list[float]]:
    """Train a model with verbose logging.

    Args:
        model: The model to train.
        model_name: Name identifier for the model.
        train_loader: Training data loader.
        test_loader: Validation/test data loader.

    Returns:
        Dictionary with training history.

    """
    start_time = time.time()

    # End any active run first (for exec() re-runs)
    if mlflow.active_run():
        mlflow.end_run()

    # Start MLflow run
    mlflow.start_run(run_name=model_name)

    # Log hyperparameters
    mlflow.log_params({
        "model_name": model_name,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "patience": PATIENCE,
        "seed": SEED,
        "optimizer": "Adam",
        "loss": "BCEWithLogitsLoss",
    })

    # Setup
    model.to(DEVICE)
    pos_weight = _compute_pos_weight(train_loader, DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # Early stopping - monitor val_auc (primary metric for imbalanced data)
    early_stopping = EarlyStopping(
        patience=PATIENCE,
        mode="max",
        monitor="val_auc",
        save_path=MODELS_DIR,
    )

    # Training history
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_auc": [],
    }

    best_auc = 0.0
    best_epoch = 0
    total_epochs_trained = 0

    print(f"\n{'='*60}")
    print(f"TRAINING {model_name.upper()}")
    print(f"{'='*60}")

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # Training phase
        model.train()
        epoch_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_times = []

        for batch_idx, (images, labels) in enumerate(train_loader):
            batch_start = time.time()

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Compute batch accuracy
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predictions == labels.unsqueeze(1)).sum().item()
            train_total += labels.size(0)

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            # Calculate timing for custom progress bar
            avg_batch_time = sum(batch_times) / len(batch_times)
            elapsed = batch_idx * avg_batch_time
            remaining = (len(train_loader) - batch_idx) * avg_batch_time

            # Print detailed progress every 50 batches
            if (batch_idx + 1) % 50 == 0 or batch_idx == len(train_loader) - 1:
                progress = 100.0 * (batch_idx + 1) / len(train_loader)
                bar_len = 20
                filled = int(bar_len * (batch_idx + 1) / len(train_loader))
                bar = "█" * filled + "░" * (bar_len - filled)

                elapsed_str = f"{int(elapsed)//60}m{int(elapsed)%60}s"
                remaining_str = f"{int(remaining)//60}m{int(remaining)%60}s"

                print(
                    f"[{bar}] {progress:.1f}%  "
                    f"batch {batch_idx + 1}/{len(train_loader)}  │  "
                    f"loss={loss.item():.4f}  "
                    f"acc={100.0 * train_correct / train_total:.4f}  │  "
                    f"elapsed={elapsed_str}  eta={remaining_str}"
                )

        avg_train_loss = epoch_loss / len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total if train_total > 0 else 0.0

        # Validation phase
        val_loss, val_accuracy, val_auc = validate(model, test_loader, criterion, DEVICE)

        # Log to MLflow
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
        mlflow.log_metric("val_auc", val_auc, step=epoch)

        # Store history
        history["train_loss"].append(avg_train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_auc"].append(val_auc)

        # Track best AUC
        is_new_best = val_auc > best_auc
        if is_new_best:
            best_auc = val_auc
            best_epoch = epoch
            # Save best model
            best_path = f"{MODELS_DIR}/{model_name}_best.pth"
            torch.save(model.state_dict(), best_path)
            mlflow.log_artifact(best_path)

        total_epochs_trained = epoch

        # Print epoch results box
        epoch_time = time.time() - epoch_start
        best_marker = "   ⭐ NEW BEST!" if is_new_best else ""

        print()
        print("╔" + "═" * 39 + "╗")
        print(f"║  EPOCH {epoch}/{EPOCHS} RESULTS{best_marker:>16} ║")
        print("╠" + "═" * 39 + "╣")
        print("║  METRIC        TRAIN        VAL       ║")
        print(f"║  Loss          {avg_train_loss:.4f}       {val_loss:.4f}    ║")
        print(f"║  Accuracy      {train_accuracy/100:.4f}       {val_accuracy/100:.4f}    ║")
        print(f"║  AUC           —            {val_auc:.4f}    ║")
        print("╚" + "═" * 39 + "╝")
        print(f"Epoch time: {epoch_time:.1f}s")

        # Early stopping check
        if early_stopping.step(val_auc, model, model_name):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    # Final logging
    mlflow.log_metric("best_val_auc", best_auc)
    mlflow.log_metric("total_epochs_trained", total_epochs_trained)

    # Save final checkpoint
    final_path = f"{MODELS_DIR}/{model_name}_final.pth"
    torch.save(model.state_dict(), final_path)
    mlflow.log_artifact(final_path)

    # Log model as artifact
    mlflow.log_artifact(f"{MODELS_DIR}/{model_name}_best.pth")

    mlflow.end_run()

    total_time = time.time() - start_time
    print(f"\n✓ Training completed in {total_time/60:.1f} minutes")
    print(f"✓ Best validation AUC: {best_auc:.4f} (epoch {best_epoch})")

    return history, best_epoch, best_auc, total_time


print("Training function ready")


def save_model_report(model_name, model, history,
                      best_epoch, train_time, test_loader, device):
    """Lightweight report generation for a trained model.

    Args:
        model_name: Name of the model (e.g., "resnet152")
        model: Trained PyTorch model
        history: Training history dict with train_loss, val_loss, val_auc
        best_epoch: Epoch with best validation AUC
        train_time: Training time in minutes
        test_loader: DataLoader for test set
        device: torch.device for computation

    Returns:
        Dictionary containing all computed metrics

    """
    print(f"\n📊 Saving {model_name} report...")

    # Get predictions (fast)
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze()
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    auc_score = roc_auc_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    # Simple 1-row visualization
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"{model_name.upper()}  |  "
        f"AUC: {auc_score:.4f}  |  "
        f"Acc: {accuracy*100:.1f}%  |  "
        f"F1: {f1:.4f}  |  "
        f"Best Epoch: {best_epoch}  |  "
        f"Time: {train_time:.1f} min",
        fontsize=13, fontweight="bold",
        color="gold", y=1.02
    )

    # Plot 1: Training curves
    ax1 = axes[0]
    epochs_range = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs_range, history["train_loss"],
             "b-", linewidth=2, label="Train Loss")
    ax1.plot(epochs_range, history["val_loss"],
             "r-", linewidth=2, label="Val Loss")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs_range, history["val_auc"],
                  "g--", linewidth=2, label="Val AUC")
    ax1_twin.set_ylabel("AUC", color="green")
    ax1.axvline(x=best_epoch, color="gold",
                linestyle="--", linewidth=1.5,
                label=f"Best: ep{best_epoch}")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2,
               fontsize=7, loc="upper right")
    ax1.set_title("Training Curves", color="white")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Confusion matrix
    ax2 = axes[1]
    sns.heatmap(cm, annot=True, fmt="d",
                cmap="Blues", ax=ax2,
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"],
                annot_kws={"size": 14, "weight": "bold"})
    ax2.set_title("Confusion Matrix", color="white")
    ax2.set_ylabel("True")
    ax2.set_xlabel("Predicted")

    # Plot 3: Metrics bars
    ax3 = axes[2]
    metric_names = ["AUC", "Accuracy", "F1",
                     "Precision", "Recall", "Specificity"]
    metric_values = [auc_score, accuracy, f1,
                     precision, recall, specificity]
    colors = ["#e74c3c", "#3498db", "#2ecc71",
              "#f39c12", "#9b59b6", "#1abc9c"]
    bars = ax3.bar(metric_names, metric_values,
                   color=colors, alpha=0.85)
    for bar, val in zip(bars, metric_values):
        ax3.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f"{val:.3f}", ha="center",
            va="bottom", fontsize=9,
            color="white", fontweight="bold"
        )
    ax3.set_ylim(0, 1.15)
    ax3.axhline(y=0.9, color="gold",
                linestyle="--", alpha=0.5)
    ax3.set_title("Metrics", color="white")
    ax3.set_ylabel("Score")
    ax3.tick_params(axis="x", rotation=30)
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save plot
    plot_path = f"outputs/plots/{model_name}_report.png"
    plt.savefig(plot_path, dpi=100,
                bbox_inches="tight",
                facecolor="black")
    plt.show()
    plt.close()

    # Save txt
    txt_path = f"outputs/reports/{model_name}_report.txt"
    with open(txt_path, "w") as f:
        f.write(f"{'='*50}\n")
        f.write(f"  {model_name.upper()} RESULTS\n")
        f.write(f"{'='*50}\n")
        f.write(f"  AUC-ROC     : {auc_score:.4f}\n")
        f.write(f"  Accuracy    : {accuracy*100:.2f}%\n")
        f.write(f"  F1 Score    : {f1:.4f}\n")
        f.write(f"  Precision   : {precision:.4f}\n")
        f.write(f"  Recall      : {recall:.4f}\n")
        f.write(f"  Specificity : {specificity:.4f}\n")
        f.write(f"  Best Epoch  : {best_epoch}\n")
        f.write(f"  Train Time  : {train_time:.1f} min\n")
        f.write(f"  TN={tn} FP={fp} FN={fn} TP={tp}\n")
        f.write(f"{'='*50}\n")

    # Log to DagHub
    if mlflow.active_run():
        mlflow.log_artifact(plot_path)
        mlflow.log_artifact(txt_path)
        mlflow.log_metrics({
            "test_auc": auc_score,
            "test_accuracy": accuracy,
            "test_f1": f1,
            "test_precision": precision,
            "test_recall": recall,
            "test_specificity": specificity
        })

    print(f"✅ {model_name} → AUC: {auc_score:.4f} | "
          f"Acc: {accuracy*100:.1f}% | F1: {f1:.4f}")
    print(f"✅ Saved: {plot_path}")
    print(f"✅ Saved: {txt_path}")
    print(f"✅ Logged to DagHub")

    return {
        "accuracy": accuracy,
        "auc": auc_score,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "confusion_matrix": cm.tolist(),
        "best_epoch": best_epoch,
        "train_time": train_time
    }


# ═══════════════════════════════════════════════════════
# CELL 11: Train ResNet152
# ═══════════════════════════════════════════════════════

# Controlled by CONTROL CELL — os.environ["TRAIN_RESNET"]
if TRAIN_RESNET:
    print("\n" + "=" * 60)
    print("Training ResNet152...")
    print("=" * 60)

    resnet_model = init_model("resnet152")
    resnet_history, resnet_best_epoch, resnet_best_auc, resnet_train_time = verbose_train(
        resnet_model,
        "resnet152",
        train_loader,
        test_loader,
    )
    print("\n✓ ResNet152 training complete!")

    # Generate ResNet152 report immediately after training
    resnet_metrics = save_model_report(
        model_name="resnet152",
        model=resnet_model,
        history=resnet_history,
        best_epoch=resnet_best_epoch,
        train_time=resnet_train_time / 60,  # Convert to minutes
        test_loader=test_loader,
        device=DEVICE
    )
else:
    print("ResNet152 training skipped (set TRAIN_RESNET = True to enable)")

# ═══════════════════════════════════════════════════════
# CELL 12: Train EfficientNetB2
# ═══════════════════════════════════════════════════════

# Controlled by CONTROL CELL — os.environ["TRAIN_EFFICIENTNET"]
if TRAIN_EFFICIENTNET:
    print("\n" + "=" * 60)
    print("Training EfficientNetB2...")
    print("=" * 60)

    efficientnet_model = init_model("efficientnet_b2")
    efficientnet_history, efficientnet_best_epoch, efficientnet_best_auc, efficientnet_train_time = verbose_train(
        efficientnet_model,
        "efficientnet_b2",
        train_loader,
        test_loader,
    )
    print("\n✓ EfficientNetB2 training complete!")

    # Generate EfficientNetB2 report immediately after training
    efficientnet_metrics = save_model_report(
        model_name="efficientnet_b2",
        model=efficientnet_model,
        history=efficientnet_history,
        best_epoch=efficientnet_best_epoch,
        train_time=efficientnet_train_time / 60,  # Convert to minutes
        test_loader=test_loader,
        device=DEVICE
    )
else:
    print("EfficientNetB2 training skipped (set TRAIN_EFFICIENTNET = True to enable)")

# ═══════════════════════════════════════════════════════
# CELL 13: Train Xception
# ═══════════════════════════════════════════════════════

# Controlled by CONTROL CELL — os.environ["TRAIN_XCEPTION"]
if TRAIN_XCEPTION:
    print("\n" + "=" * 60)
    print("Training Xception...")
    print("=" * 60)

    xception_model = init_model("xception")
    xception_history, xception_best_epoch, xception_best_auc, xception_train_time = verbose_train(
        xception_model,
        "xception",
        train_loader,
        test_loader,
    )
    print("\n✓ Xception training complete!")

    # Generate Xception report immediately after training
    xception_metrics = save_model_report(
        model_name="xception",
        model=xception_model,
        history=xception_history,
        best_epoch=xception_best_epoch,
        train_time=xception_train_time / 60,  # Convert to minutes
        test_loader=test_loader,
        device=DEVICE
    )
else:
    print("Xception training skipped (set TRAIN_XCEPTION = True to enable)")

# ═══════════════════════════════════════════════════════
# CELL 14: Evaluate all models
# ═══════════════════════════════════════════════════════

# After all models are trained, set EVALUATE_MODELS = True

if EVALUATE_MODELS:
    print("\n" + "=" * 60)
    print("EVALUATING ALL MODELS")
    print("=" * 60)

    results = {}
    models_to_eval = ["resnet152", "efficientnet_b2", "xception"]

    for model_name in models_to_eval:
        model_path = f"{MODELS_DIR}/{model_name}_best.pth"

        if not Path(model_path).exists():
            print(f"\n⚠ {model_name} checkpoint not found, skipping...")
            continue

        print(f"\nLoading {model_name}...")
        model = init_model(model_name)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.to(DEVICE)

        print(f"Evaluating {model_name}...")
        metrics = evaluate(model, test_loader, DEVICE, model_name)
        results[model_name] = metrics

    # Print comparison table
    if results:
        print("\n" + "=" * 55)
        print("MODEL           ACCURACY    AUC      F1      RECALL")
        print("=" * 55)
        for name, metrics in results.items():
            acc = metrics.get("accuracy", 0.0)
            auc = metrics.get("auc_roc", 0.0)
            f1 = metrics.get("f1_score", 0.0)
            rec = metrics.get("recall", 0.0)
            print(f"{name:<15} {acc:.4f}      {auc:.4f}   {f1:.4f}  {rec:.4f}")
        print("=" * 55)

        # Store for next cell
        resnet_metrics = results.get("resnet152", {})
        efficientnet_metrics = results.get("efficientnet_b2", {})
        xception_metrics = results.get("xception", {})
    else:
        print("No models found for evaluation")
else:
    print("Evaluation skipped (set EVALUATE_MODELS = True after training all models)")

# ═══════════════════════════════════════════════════════
# CELL 15: Generate and save all plots
# ═══════════════════════════════════════════════════════

# After evaluation, set GENERATE_PLOTS = True

if GENERATE_PLOTS:
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    plot_files = []

    # Training history plots
    histories = [
        ("resnet152", resnet_history),
        ("efficientnet_b2", efficientnet_history),
        ("xception", xception_history),
    ]

    for model_name, history in histories:
        if history and any(history.values()):
            print(f"\nGenerating training history plot for {model_name}...")
            plot_training_history(history, model_name)
            plot_files.append(f"{PLOTS_DIR}/{model_name}_history.png")

    # Confusion matrices
    # Need to re-run evaluation to get predictions for confusion matrix
    print("\nGenerating confusion matrices...")
    for model_name in ["resnet152", "efficientnet_b2", "xception"]:
        model_path = f"{MODELS_DIR}/{model_name}_best.pth"
        if Path(model_path).exists():
            model = init_model(model_name)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()

            all_preds = []
            all_labels = []

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(DEVICE)
                    outputs = model(images)
                    probs = torch.sigmoid(outputs).squeeze()
                    preds = (probs >= 0.5).int()
                    all_preds.extend(preds.cpu().numpy().tolist())
                    all_labels.extend(labels.cpu().numpy().astype(int).tolist())

            plot_confusion_matrix(all_labels, all_preds, model_name)
            plot_files.append(f"{PLOTS_DIR}/{model_name}_confusion_matrix.png")

    # Model comparison
    if results:
        print("\nGenerating model comparison plot...")
        model_names = list(results.keys())
        accuracies = [results[m]["accuracy"] for m in model_names]
        aucs = [results[m]["auc_roc"] for m in model_names]
        plot_model_comparison(model_names, accuracies, aucs)
        plot_files.append(f"{PLOTS_DIR}/model_comparison.png")

    # Log all plots as MLflow artifacts
    print("\nLogging plots to MLflow...")
    for plot_file in plot_files:
        if Path(plot_file).exists():
            mlflow.log_artifact(plot_file)
            print(f"  ✓ {plot_file}")

    print("\n✓ All plots generated and logged")
else:
    print("Plot generation skipped (set GENERATE_PLOTS = True after evaluation)")


def save_best_model_visualization(all_metrics, all_histories,
                                   all_best_epochs, all_train_times):
    """Identify best model by AUC-ROC and generate detailed visualization.

    Args:
        all_metrics: Dict of model metrics {"resnet152": {...}, ...}
        all_histories: Dict of training histories
        all_best_epochs: Dict of best epoch numbers
        all_train_times: Dict of training times in minutes

    """
    # Find best model by AUC
    best_model_name = max(all_metrics,
                          key=lambda k: all_metrics[k]["auc"])
    best = all_metrics[best_model_name]
    best_history = all_histories[best_model_name]
    best_epoch = all_best_epochs[best_model_name]
    best_train_time = all_train_times[best_model_name]

    print(f"\n{'='*60}")
    print(f"  🏆 BEST MODEL: {best_model_name.upper()}")
    print(f"  AUC-ROC: {best['auc']:.4f}")
    print(f"{'='*60}")

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Create visualization
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle(
        f"🏆 BEST MODEL: {best_model_name.upper()}\n"
        f"Breast Cancer Detection — BreakHis Dataset | "
        f"AUC: {best['auc']:.4f} | {date_str}",
        fontsize=18, fontweight="bold",
        y=0.98, color="gold"
    )

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.45, wspace=0.35)

    # Plot 1: Training history
    ax1 = fig.add_subplot(gs[0, 0])
    epochs_range = range(1, len(best_history["train_loss"]) + 1)
    ax1.plot(epochs_range, best_history["train_loss"],
             "b-", linewidth=2, label="Train Loss")
    ax1.plot(epochs_range, best_history["val_loss"],
             "r-", linewidth=2, label="Val Loss")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs_range, best_history["val_auc"],
                  "g--", linewidth=2, label="Val AUC")
    ax1_twin.set_ylabel("AUC", color="green")
    ax1.axvline(x=best_epoch, color="gold",
                linestyle="--", linewidth=2,
                label=f"Best epoch {best_epoch}")
    ax1.axvline(x=5, color="white",
                linestyle=":", linewidth=1, alpha=0.5)
    ax1.text(2, max(best_history["train_loss"])*0.95,
             "Phase 1", color="white", fontsize=8)
    ax1.text(6, max(best_history["train_loss"])*0.95,
             "Phase 2", color="white", fontsize=8)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2,
               fontsize=7, loc="upper right")
    ax1.set_title("Training History", color="white")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy curve
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs_range, best_history["train_accuracy"],
             "b-", linewidth=2, label="Train Acc")
    ax2.plot(epochs_range, best_history["val_accuracy"],
             "r-", linewidth=2, label="Val Acc")
    ax2.axvline(x=best_epoch, color="gold",
                linestyle="--", linewidth=2)
    ax2.set_title("Accuracy Curves", color="white")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Confusion matrix
    ax3 = fig.add_subplot(gs[0, 2])
    cm_array = np.array(best["confusion_matrix"])
    sns.heatmap(cm_array, annot=True, fmt="d",
                cmap="Blues", ax=ax3,
                xticklabels=["Benign", "Malignant"],
                yticklabels=["Benign", "Malignant"],
                annot_kws={"size": 14, "weight": "bold"})
    ax3.set_title(f"Confusion Matrix\n"
                  f"AUC: {best['auc']:.4f}",
                  color="white")
    ax3.set_ylabel("True Label")
    ax3.set_xlabel("Predicted Label")

    # Plot 4: ROC Curve
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.text(0.5, 0.5,
             f"AUC = {best['auc']:.4f}",
             ha="center", va="center",
             fontsize=20, color="gold",
             fontweight="bold",
             transform=ax4.transAxes)
    ax4.set_title("AUC-ROC Score", color="white")
    ax4.axis("off")

    # Plot 5: Metrics bar chart
    ax5 = fig.add_subplot(gs[1, 1])
    metric_names = ["Accuracy", "AUC-ROC", "F1",
                     "Precision", "Recall", "Specificity"]
    metric_values = [
        best["accuracy"], best["auc"], best["f1"],
        best["precision"], best["recall"], best["specificity"]
    ]
    colors = ["#3498db","#e74c3c","#2ecc71",
              "#f39c12","#9b59b6","#1abc9c"]
    bars = ax5.bar(metric_names, metric_values,
                   color=colors, alpha=0.85)
    for bar, val in zip(bars, metric_values):
        ax5.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center",
                 va="bottom", fontsize=9,
                 color="white", fontweight="bold")
    ax5.set_ylim(0, 1.15)
    ax5.axhline(y=0.9, color="gold",
                linestyle="--", alpha=0.5)
    ax5.set_title("All Metrics", color="white")
    ax5.set_ylabel("Score")
    ax5.tick_params(axis="x", rotation=30)
    ax5.grid(True, alpha=0.3, axis="y")

    # Plot 6: Model comparison bars
    ax6 = fig.add_subplot(gs[1, 2])
    model_names_list = list(all_metrics.keys())
    auc_values = [all_metrics[m]["auc"]
                  for m in model_names_list]
    bar_colors = ["gold" if m == best_model_name
                  else "#3498db"
                  for m in model_names_list]
    bars6 = ax6.bar(model_names_list, auc_values,
                    color=bar_colors, alpha=0.85)
    for bar, val in zip(bars6, auc_values):
        ax6.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.002,
                 f"{val:.4f}", ha="center",
                 va="bottom", fontsize=9,
                 color="white", fontweight="bold")
    ax6.set_ylim(min(auc_values) - 0.05, 1.05)
    ax6.set_title("Model AUC Comparison\n(Gold = Best)",
                  color="white")
    ax6.set_ylabel("AUC-ROC")
    ax6.tick_params(axis="x", rotation=15)
    ax6.grid(True, alpha=0.3, axis="y")

    # Plot 7: Summary table (full width)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis("off")

    table_data = []
    for mname, mdata in all_metrics.items():
        table_data.append([
            "🏆 " + mname.upper()
            if mname == best_model_name
            else mname.upper(),
            f"{mdata['accuracy']*100:.2f}%",
            f"{mdata['auc']:.4f}",
            f"{mdata['f1']:.4f}",
            f"{mdata['precision']:.4f}",
            f"{mdata['recall']:.4f}",
            f"{mdata['specificity']:.4f}",
            f"{all_best_epochs[mname]}",
            f"{all_train_times[mname]:.1f} min"
        ])

    col_labels = ["Model", "Accuracy", "AUC-ROC",
                  "F1", "Precision", "Recall",
                  "Specificity", "Best Epoch", "Train Time"]

    table = ax7.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0.1, 1, 0.85]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white",
                                fontweight="bold")
        else:
            model_row = table_data[row-1][0]
            if "🏆" in model_row:
                cell.set_facecolor("#7d6608")
                cell.set_text_props(color="gold",
                                    fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#1a1a2e")
                cell.set_text_props(color="white")
            else:
                cell.set_facecolor("#16213e")
                cell.set_text_props(color="white")
        cell.set_edgecolor("#444444")

    ax7.set_title(
        "⚠️  Research prototype only. Not for clinical use.\n"
        "Results on held-out test set — "
        "split by patient ID (no data leakage)",
        color="gray", fontsize=9, pad=10
    )

    # Save to Kaggle outputs
    plot_path = "outputs/plots/best_model_report.png"
    plt.savefig(plot_path, dpi=150,
                bbox_inches="tight",
                facecolor="black")
    plt.show()
    print(f"✅ Best model visualization saved: {plot_path}")

    # Save best model txt summary
    txt_path = "outputs/reports/best_model_summary.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"  🏆 BEST MODEL SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"  Date         : {date_str}\n")
        f.write(f"  Best Model   : {best_model_name.upper()}\n")
        f.write(f"  Dataset      : BreakHis All Magnifications\n")
        f.write(f"  Total images : 7,909\n")
        f.write("=" * 60 + "\n\n")
        f.write("  ALL MODELS COMPARISON:\n")
        f.write(f"  {'─'*56}\n")
        f.write(f"  {'Model':<20} {'AUC':>8} {'Acc':>8} "
                f"{'F1':>8} {'Recall':>8}\n")
        f.write(f"  {'─'*56}\n")
        for mname, mdata in all_metrics.items():
            marker = " ← BEST" if mname == best_model_name else ""
            f.write(f"  {mname.upper():<20} "
                    f"{mdata['auc']:>8.4f} "
                    f"{mdata['accuracy']:>8.4f} "
                    f"{mdata['f1']:>8.4f} "
                    f"{mdata['recall']:>8.4f}"
                    f"{marker}\n")
        f.write(f"  {'─'*56}\n\n")
        f.write("=" * 60 + "\n")
        f.write("  ⚠️  Research prototype. Not for clinical use.\n")
        f.write("=" * 60 + "\n")

    print(f"✅ Best model summary saved: {txt_path}")

    # Log everything to DagHub
    if mlflow.active_run():
        mlflow.log_artifact(plot_path)
        mlflow.log_artifact(txt_path)
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_model_auc", best["auc"])
        print(f"✅ Best model report logged to DagHub")

    print(f"\n{'='*60}")
    print(f"  🏆 BEST MODEL: {best_model_name.upper()}")
    print(f"  AUC-ROC     : {best['auc']:.4f}")
    print(f"  Accuracy    : {best['accuracy']*100:.2f}%")
    print(f"  F1 Score    : {best['f1']:.4f}")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════
# CELL 16: Final summary
# ═══════════════════════════════════════════════════════

# After all steps complete, set SHOW_SUMMARY = True

if SHOW_SUMMARY:
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    # Collect all results
    all_metrics = {}
    all_histories = {}
    all_best_epochs = {}
    all_train_times = {}

    # Add whichever models have been trained
    if "resnet_metrics" in dir():
        all_metrics["resnet152"] = resnet_metrics
        all_histories["resnet152"] = resnet_history
        all_best_epochs["resnet152"] = resnet_best_epoch
        all_train_times["resnet152"] = resnet_train_time

    if "efficientnet_metrics" in dir():
        all_metrics["efficientnet_b2"] = efficientnet_metrics
        all_histories["efficientnet_b2"] = efficientnet_history
        all_best_epochs["efficientnet_b2"] = efficientnet_best_epoch
        all_train_times["efficientnet_b2"] = efficientnet_train_time

    if "xception_metrics" in dir():
        all_metrics["xception"] = xception_metrics
        all_histories["xception"] = xception_history
        all_best_epochs["xception"] = xception_best_epoch
        all_train_times["xception"] = xception_train_time

    if all_metrics:
        save_best_model_visualization(
            all_metrics=all_metrics,
            all_histories=all_histories,
            all_best_epochs=all_best_epochs,
            all_train_times=all_train_times
        )
    else:
        print("No models trained yet.")

    print("\nWorkflow:")
    print("  1. Run Cell 1-10 to setup")
    print("  2. Run Cell 11 (ResNet152)")
    print("  3. Set TRAIN_EFFICIENTNET = True, run Cell 12")
    print("  4. Set TRAIN_XCEPTION = True, run Cell 13")
    print("  5. Set EVALUATE_MODELS = True, run Cell 14")
    print("  6. Set GENERATE_PLOTS = True, run Cell 15")
    print("  7. Set SHOW_SUMMARY = True, run Cell 16")

    print(f"\nDagHub Experiments URL:")
    print(f"  https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}/experiments")

    print("\n" + "=" * 60)
    print("All done!")
    print("=" * 60)
else:
    print("\nFinal summary skipped (set SHOW_SUMMARY = True when complete)")
    print("\nTo complete the workflow:")
    print("  1. Train ResNet152 (Cell 11)")
    print("  2. Train EfficientNetB2 (Cell 12)")
    print("  3. Train Xception (Cell 13)")
    print("  4. Evaluate all models (Cell 14)")
    print("  5. Generate plots (Cell 15)")
    print("  6. Show final summary (Cell 16)")
