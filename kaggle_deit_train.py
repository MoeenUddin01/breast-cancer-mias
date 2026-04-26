"""DeiT-B Distilled Transformer Training for BreakHis on Kaggle."""
from __future__ import annotations

# SECTION 1 - Imports
import atexit
import fcntl
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import dagshub
import mlflow
import cv2
import psutil
from kaggle_secrets import UserSecretsClient
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from torchvision import transforms as T

sys.path.insert(0, "/kaggle/working/breast-cancer-mias")

from src.data.loader import load_data
from src.data.splitter import split_by_patient_id
from src.data.dataset import MIASDataset
from src.transformers.deit_model import get_deit_model
from src.transformers.deit_trainer import train_deit
from src.transformers.deit_config import (
    DEIT_MODEL_NAME,
    DEIT_BATCH_SIZE,
    DEIT_ACCUMULATION_STEPS,
    DEIT_EPOCHS,
    DEIT_IMAGE_SIZE,
    DEIT_PHASE1_EPOCHS,
    DEIT_UNFREEZE_BLOCKS,
    DEIT_DROP_PATH_RATE,
)

# SECTION 2 - Config
DATA_DIR = "/kaggle/input/datasets/ambarish/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/"
OUTPUT_DIR = "/kaggle/working/outputs/"
MODELS_DIR = "/kaggle/working/outputs/models/"
PLOTS_DIR = "/kaggle/working/outputs/plots/"
REPORTS_DIR = "/kaggle/working/outputs/reports/"
SEED = 42
TEST_SIZE = 0.15
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = DEIT_IMAGE_SIZE
QUIET_TQDM = True
ENABLE_OFFLINE_AUG = True
BATCH_SIZE = DEIT_BATCH_SIZE
TRAIN_LOCK_PATH = Path("/tmp/deit_training_train_script.lock")


def _acquire_train_script_lock() -> object | None:
    """Allow only one kaggle_deit_train.py process at a time."""
    lock_file = open(TRAIN_LOCK_PATH, "w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return None

    lock_file.write(str(os.getpid()))
    lock_file.flush()

    def _cleanup_lock() -> None:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
            if TRAIN_LOCK_PATH.exists():
                TRAIN_LOCK_PATH.unlink()
        except OSError:
            pass

    atexit.register(_cleanup_lock)
    return lock_file


_train_lock = _acquire_train_script_lock()
if _train_lock is None:
    print("⚠️ kaggle_deit_train.py is already running. Exiting duplicate run.")
    raise SystemExit(0)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

print("=" * 60)
print("  DeiT-B DISTILLED TRANSFORMER TRAINING")
print("  BreakHis Breast Cancer Detection")
print("=" * 60)
print(f"  Device : {DEVICE}")
print(f"  Dataset: {DATA_DIR}")
print(f"  Batch size     : {DEIT_BATCH_SIZE}")
print(f"  Phase 1 epochs : {DEIT_PHASE1_EPOCHS}")
print(f"  Phase 2 epochs : {DEIT_EPOCHS}")
print(f"  Total max      : {DEIT_PHASE1_EPOCHS + DEIT_EPOCHS}")
print("  Offline aug    : ON")

# SECTION 3 - Seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f"✓ Seed set to {SEED}")

# SECTION 4 - DagHub init
try:
    secrets = UserSecretsClient()
    # Optional Hugging Face auth to avoid unauthenticated Hub warnings/rate limits.
    try:
        hf_token = secrets.get_secret("HF_TOKEN")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
            print("✓ Hugging Face token loaded from Kaggle secrets")
    except Exception:
        # Keep training flow unchanged if HF secret is not configured.
        pass

    os.environ["MLFLOW_TRACKING_USERNAME"] = secrets.get_secret(
        "MLFLOW_TRACKING_USERNAME"
    )
    os.environ["MLFLOW_TRACKING_PASSWORD"] = secrets.get_secret(
        "MLFLOW_TRACKING_PASSWORD"
    )
    dagshub.init(
        repo_owner="MoeenUddin01",
        repo_name="breast-cancer-mias",
        mlflow=True,
    )
    mlflow.set_experiment("BreakHis_Transformer_Comparison")
    print("✓ DagHub initialized")
    print(f"✓ MLflow URI: {mlflow.get_tracking_uri()}")
except Exception as e:
    print(f"⚠️ DagHub failed: {e}")

# SECTION 5 - Load data
print("\n⏳ Loading BreakHis dataset...")
data = load_data(DATA_DIR)
print(f"✓ Total samples: {len(data)}")
ram = psutil.virtual_memory().used / 1e9
print(f"RAM after load_data()    : {ram:.1f} GB")

# SECTION 6 - Split by patient
print("\n⏳ Splitting by patient ID...")
train_data, test_data = split_by_patient_id(data, TEST_SIZE, SEED)
print(f"✓ Train: {len(train_data)} | Test: {len(test_data)}")
ram = psutil.virtual_memory().used / 1e9
print(f"RAM after split()        : {ram:.1f} GB")

# SECTION 7 - Offline augmentation (5× expansion)
aug_dir = "/kaggle/working/aug_cache/"
os.makedirs(aug_dir, exist_ok=True)


def augment_stain(img: np.ndarray) -> np.ndarray:
    """Apply random per-channel stain perturbation."""
    img_float = img.astype(np.float32) / 255.0
    for ch in range(3):
        scale = np.random.uniform(0.85, 1.15)
        img_float[:, :, ch] *= scale
    return np.clip(img_float * 255, 0, 255).astype(np.uint8)


def augment_for_deit(train_data_paths: list[tuple[str, str, int]]) -> list[tuple[str, str, int]]:
    """Apply offline augmentation for DeiT training (5× expansion).

    Args:
        train_data_paths: List of (patient_id, image_path, label) tuples.

    Returns:
        Augmented list with original + 4 transformed copies per sample.

    """
    augmented_paths = list(train_data_paths)
    for idx, (pid, image_path, label) in enumerate(tqdm(
        train_data_paths,
        desc="Augmenting for DeiT",
        disable=QUIET_TQDM,
    ), start=1):
        img = cv2.imread(image_path)
        if img is None:
            print(f"[WARNING] Failed to load image for augmentation: {image_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        augs = {
            "hf": cv2.flip(img, 1),
            "vf": cv2.flip(img, 0),
            "stain1": augment_stain(img),
            "stain2": augment_stain(img),
        }

        for suffix, aug_img in augs.items():
            aug_path = os.path.join(aug_dir, f"{pid}_{idx}_{suffix}.png")
            cv2.imwrite(aug_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
            augmented_paths.append((f"{pid}_{suffix}", aug_path, label))

    print(f"✓ {len(train_data_paths)} → {len(augmented_paths)} samples (5× expansion)")
    return augmented_paths


ram = psutil.virtual_memory().used / 1e9
print(f"RAM before augmentation  : {ram:.1f} GB")
ram_before = ram
train_data = augment_for_deit(train_data)
ram_after = psutil.virtual_memory().used / 1e9
print(f"RAM after aug : {ram_after:.1f} GB")

if ram_after > 12:
    print("⚠️ CRITICAL: RAM too high — stopping")
    raise MemoryError("Insufficient RAM for augmentation")
else:
    print("✅ RAM safe")

# SECTION 8 - Datasets and DataLoaders
# DeiT-specific stronger transforms
deit_train_transforms = T.Compose([
    T.Resize(DEIT_IMAGE_SIZE),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomRotation(degrees=180),
    T.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2,
        hue=0.1,
    ),
    T.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1),
    ),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    T.RandomErasing(
        p=0.25,
        scale=(0.02, 0.15),
    ),
])

deit_test_transforms = T.Compose([
    T.Resize(DEIT_IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

train_dataset = MIASDataset(train_data, deit_train_transforms, IMAGE_SIZE)
test_dataset = MIASDataset(test_data, deit_test_transforms, IMAGE_SIZE)

# WeightedRandomSampler for class imbalance (recalculate after augmentation)
train_labels = [item[2] for item in train_data]
class_counts = torch.bincount(torch.tensor(train_labels), minlength=2)
if (class_counts == 0).any():
    print(
        "⚠️ Train split contains only one class after patient split/augmentation. "
        "Sampling weights fallback applied for the missing class."
    )
class_weights = 1.0 / class_counts.clamp_min(1).float()
sample_weights = [class_weights[l].item() for l in train_labels]
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

deit_train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    drop_last=True,
)
deit_test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

print(f"  Train batches  : {len(deit_train_loader)}")
print(f"✓ Test batches : {len(deit_test_loader)}")

# SECTION 10 - Initialize model
print("\n⏳ Initializing DeiT-B Distilled...")
deit_model = get_deit_model()
deit_model = deit_model.to(DEVICE)

# SECTION 11 - Train with MLflow
print("\n🚀 Starting DeiT-B Distilled training...")

with mlflow.start_run(run_name="deit_b_distilled"):
    mlflow.log_params({
        "model": DEIT_MODEL_NAME,
        "pretrained_on": "ImageNet-1k+distillation",
        "dataset": "BreakHis_AllMagnifications",
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "batch_size": BATCH_SIZE,
        "grad_accumulation": DEIT_ACCUMULATION_STEPS,
        "effective_batch": BATCH_SIZE * DEIT_ACCUMULATION_STEPS,
        "architecture": "DeiT-Distilled-Transformer",
        "image_size": DEIT_IMAGE_SIZE[0],
        "phase1_epochs": DEIT_PHASE1_EPOCHS,
        "unfreeze_blocks": DEIT_UNFREEZE_BLOCKS,
        "scheduler_p1": "LinearWarmup",
        "scheduler_p2": "CosineAnnealingWarmRestarts",
        "label_smoothing": 0.05,
        "drop_path_rate": DEIT_DROP_PATH_RATE,
        "offline_aug": "5x_hflip_vflip_stain1_stain2",
    })

    deit_history, deit_best_epoch, deit_train_time = train_deit(
        model=deit_model,
        train_loader=deit_train_loader,
        val_loader=deit_test_loader,
        model_name="deit",
        device=DEVICE,
    )

# SECTION 11 - Evaluation and report
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

deit_model.eval()
all_probs, all_labels = [], []
with torch.no_grad():
    for images, labels in deit_test_loader:
        images = images.to(DEVICE)
        outputs = deit_model(images)
        probs = torch.sigmoid(outputs).squeeze()
        if probs.dim() == 0:
            probs = probs.unsqueeze(0)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

all_labels = np.array(all_labels)
all_probs = np.array(all_probs)
all_preds = (all_probs >= 0.5).astype(int)

accuracy = accuracy_score(all_labels, all_preds)
try:
    auc_score = roc_auc_score(all_labels, all_probs)
except ValueError:
    print("⚠️ Test AUC undefined for single-class labels; reporting AUC=0.5.")
    auc_score = 0.5
f1 = f1_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

# Save txt report
date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
txt_path = REPORTS_DIR + "deit_report.txt"
with open(txt_path, "w") as f:
    f.write(f"{'=' * 50}\n")
    f.write("  DEIT-B DISTILLED RESULTS\n")
    f.write(f"{'=' * 50}\n")
    f.write(f"  Date        : {date_str}\n")
    f.write(f"  Model       : {DEIT_MODEL_NAME}\n")
    f.write("  Pretrained  : ImageNet-1k + distillation\n")
    f.write("  Dataset     : BreakHis All Magnifications\n")
    f.write(f"{'=' * 50}\n")
    f.write(f"  AUC-ROC     : {auc_score:.4f}\n")
    f.write(f"  Accuracy    : {accuracy * 100:.2f}%\n")
    f.write(f"  F1 Score    : {f1:.4f}\n")
    f.write(f"  Precision   : {precision:.4f}\n")
    f.write(f"  Recall      : {recall:.4f}\n")
    f.write(f"  Specificity : {specificity:.4f}\n")
    f.write(f"  Best Epoch  : {deit_best_epoch}\n")
    f.write(f"  Train Time  : {deit_train_time:.1f} min\n")
    f.write(f"  TN={tn} FP={fp} FN={fn} TP={tp}\n")
    f.write(f"{'=' * 50}\n")
    f.write(
        classification_report(
            all_labels, all_preds, target_names=["Benign", "Malignant"]
        )
    )

# Save visualization
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("dark_background")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    f"DeiT-B Dist  |  AUC: {auc_score:.4f}  |  "
    f"Acc: {accuracy * 100:.1f}%  |  F1: {f1:.4f}  |  "
    f"Best Epoch: {deit_best_epoch}  |  "
    f"Time: {deit_train_time:.1f} min",
    fontsize=12,
    fontweight="bold",
    color="gold",
    y=1.02,
)

# Plot 1: Training curves
ax1 = axes[0]
epochs_range = range(1, len(deit_history["train_loss"]) + 1)
ax1.plot(
    epochs_range,
    deit_history["train_loss"],
    "b-",
    linewidth=2,
    label="Train Loss",
)
ax1.plot(
    epochs_range, deit_history["val_loss"], "r-", linewidth=2, label="Val Loss"
)
ax1_twin = ax1.twinx()
ax1_twin.plot(
    epochs_range, deit_history["val_auc"], "g--", linewidth=2, label="Val AUC"
)
ax1_twin.set_ylabel("AUC", color="green")
ax1.axvline(
    x=deit_best_epoch, color="gold", linestyle="--", linewidth=1.5
)
ax1.axvline(x=8, color="white", linestyle=":", linewidth=1, alpha=0.5)
ax1.text(
    3, max(deit_history["train_loss"]) * 0.95, "P1", color="white", fontsize=8
)
ax1.text(
    9, max(deit_history["train_loss"]) * 0.95, "P2", color="white", fontsize=8
)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper right"
)
ax1.set_title("Training Curves", color="white")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True, alpha=0.3)

# Plot 2: Confusion matrix
ax2 = axes[1]
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax2,
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"],
    annot_kws={"size": 14, "weight": "bold"},
)
ax2.set_title("Confusion Matrix", color="white")
ax2.set_ylabel("True")
ax2.set_xlabel("Predicted")

# Plot 3: Metrics bars
ax3 = axes[2]
metric_names = ["AUC", "Accuracy", "F1", "Precision", "Recall", "Specificity"]
metric_values = [auc_score, accuracy, f1, precision, recall, specificity]
colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
bars = ax3.bar(metric_names, metric_values, color=colors, alpha=0.85)
for bar, val in zip(bars, metric_values):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{val:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="white",
        fontweight="bold",
    )
ax3.set_ylim(0, 1.15)
ax3.axhline(y=0.9, color="gold", linestyle="--", alpha=0.5)
ax3.set_title("Metrics", color="white")
ax3.set_ylabel("Score")
ax3.tick_params(axis="x", rotation=30)
ax3.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plot_path = PLOTS_DIR + "deit_report.png"
plt.savefig(plot_path, dpi=100, bbox_inches="tight", facecolor="black")
plt.show()
plt.close()

# Log to DagHub
with mlflow.start_run(run_name="deit_b_evaluation"):

    def safe_log_artifact(path: str) -> None:
        """Log artifact to MLflow with existence check.

        Args:
            path: Path to the artifact file.

        """
        if os.path.exists(path):
            mlflow.log_artifact(path)
        else:
            print(f"⚠️ Skipping artifact — file not found: {path}")

    mlflow.log_metrics({
        "test_auc": auc_score,
        "test_accuracy": accuracy,
        "test_f1": f1,
        "test_precision": precision,
        "test_recall": recall,
        "test_specificity": specificity,
    })
    safe_log_artifact(txt_path)
    safe_log_artifact(plot_path)
    safe_log_artifact(MODELS_DIR + "deit_best.pth")

# Final print
print(
    f"""
╔══════════════════════════════════════╗
║   DeiT-B DISTILLED COMPLETE          ║
║   AUC-ROC  : {auc_score:.4f}                  ║
║   Accuracy : {accuracy * 100:.2f}%                  ║
║   F1 Score : {f1:.4f}                  ║
║   Time     : {deit_train_time:.1f} minutes            ║
╚══════════════════════════════════════╝
"""
)
