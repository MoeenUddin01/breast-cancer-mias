"""Training logic specifically for DeiT-B Distilled.

Separate from CNN and ViT trainers to avoid interference.
Uses gradient accumulation and label smoothing for better training.
"""

from __future__ import annotations

import time

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LinearLR,
)

from src.transformers.deit_config import (
    DEIT_ACCUMULATION_STEPS,
    DEIT_EPOCHS,
    DEIT_PATIENCE,
    DEIT_PHASE1_EPOCHS,
    DEIT_PHASE1_LR,
    DEIT_PHASE2_LR_BACKBONE,
    DEIT_PHASE2_LR_HEAD,
    DEIT_UNFREEZE_BLOCKS,
    DEIT_WEIGHT_DECAY,
)
from src.transformers.deit_model import unfreeze_deit_blocks


def _compute_pos_weight(train_loader: torch.utils.data.DataLoader, device: torch.device) -> torch.Tensor:
    """Compute pos_weight from train labels for class imbalance.

    Args:
        train_loader: DataLoader containing training data.
        device: Device to move the computed weight tensor to.

    Returns:
        Tensor containing the positive class weight.

    """
    num_positives = 0
    num_negatives = 0
    for _, labels in train_loader:
        labels_int = labels.int()
        num_positives += labels_int.sum().item()
        num_negatives += (1 - labels_int).sum().item()
    if num_positives == 0 or num_negatives == 0:
        print(
            "  [WARN] Single-class train split detected while computing pos_weight; "
            "falling back to pos_weight=1.0"
        )
        pos_weight = torch.tensor([1.0], device=device)
    else:
        pos_weight = torch.tensor([num_negatives / num_positives], device=device)
    print(
        f"  pos_weight: {pos_weight.item():.4f} "
        f"(neg={num_negatives}, pos={num_positives})"
    )
    return pos_weight


def train_deit(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    model_name: str,
    device: torch.device,
) -> tuple[dict, int, float]:
    """Two-phase training for DeiT-B Distilled with gradient accumulation.

    Phase 1: Train head only (8 epochs, lr=5e-4) with linear warmup
    Phase 2: Fine-tune last 8 blocks (remaining, lr=2e-6/2e-5)

    Features:
    - Gradient accumulation (effective batch = 16 * 4 = 64)
    - Label smoothing (smooth=0.05)
    - CosineAnnealingWarmRestarts scheduler in Phase 2

    Args:
        model: The DeiT model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        model_name: Name identifier for saving checkpoints.
        device: Device to run training on.

    Returns:
        Tuple of (history dict, best_epoch int, train_time float).

    """
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_auc": [],
    }

    pos_weight = _compute_pos_weight(train_loader, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_val_auc = 0.0
    best_epoch = 0
    no_improve = 0
    start_time = time.time()
    scaler = GradScaler()

    total_epochs = DEIT_PHASE1_EPOCHS + DEIT_EPOCHS

    print(f"\nTraining progress: 0.0% (0/{total_epochs} epochs)")

    optimizer_p1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=DEIT_PHASE1_LR,
        weight_decay=DEIT_WEIGHT_DECAY,
    )

    warmup_scheduler = LinearLR(
        optimizer_p1,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=DEIT_PHASE1_EPOCHS,
    )

    for epoch in range(DEIT_PHASE1_EPOCHS):
        train_loss, train_acc = _train_epoch_ga(
            model, train_loader, optimizer_p1, criterion, device, epoch, scaler
        )
        val_loss, val_acc, val_auc = _validate(model, val_loader, criterion, device)
        warmup_scheduler.step()

        _log_epoch(
            epoch,
            total_epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_auc,
            best_val_auc,
        )
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_auc"].append(val_auc)
        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_auc": val_auc,
                "learning_rate": optimizer_p1.param_groups[0]["lr"],
                "phase": 1,
            },
            step=epoch,
        )
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            torch.save(
                model.state_dict(), f"outputs/models/{model_name}_best.pth"
            )

    unfreeze_deit_blocks(model, DEIT_UNFREEZE_BLOCKS)

    backbone_params = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and "head" not in name
    ]
    head_params = [p for p in model.head.parameters()]

    optimizer_p2 = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": DEIT_PHASE2_LR_BACKBONE},
            {"params": head_params, "lr": DEIT_PHASE2_LR_HEAD},
        ],
        weight_decay=DEIT_WEIGHT_DECAY,
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer_p2, T_0=15, T_mult=2, eta_min=1e-8
    )

    for epoch in range(DEIT_PHASE1_EPOCHS, total_epochs):
        train_loss, train_acc = _train_epoch_ga(
            model, train_loader, optimizer_p2, criterion, device, epoch, scaler
        )
        val_loss, val_acc, val_auc = _validate(model, val_loader, criterion, device)
        scheduler.step()
        current_lr_backbone = optimizer_p2.param_groups[0]["lr"]
        current_lr_head = optimizer_p2.param_groups[1]["lr"]

        _log_epoch(
            epoch,
            total_epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_auc,
            best_val_auc,
        )
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_auc"].append(val_auc)
        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_auc": val_auc,
                "learning_rate_backbone": current_lr_backbone,
                "learning_rate_head": current_lr_head,
                "phase": 2,
            },
            step=epoch,
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(
                model.state_dict(), f"outputs/models/{model_name}_best.pth"
            )
        else:
            no_improve += 1

        if no_improve >= DEIT_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    train_time = (time.time() - start_time) / 60
    mlflow.log_metrics(
        {
            "best_val_auc": best_val_auc,
            "total_epochs_trained": len(history["train_loss"]),
            "train_time_minutes": train_time,
        }
    )
    mlflow.log_artifact(f"outputs/models/{model_name}_best.pth")

    print(f"\nTraining completed in {train_time:.1f} minutes")
    print(f"Best validation AUC: {best_val_auc:.4f} (epoch {best_epoch})")

    return history, best_epoch, train_time


def _train_epoch_ga(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    epoch: int,
) -> tuple[float, float]:
    """Single training epoch with gradient accumulation, label smoothing, and mixed precision.

    Args:
        model: Model to train.
        loader: DataLoader for training data.
        optimizer: Optimizer instance.
        criterion: Loss criterion.
        device: Device to run on.
        epoch: Current epoch number.
        scaler: GradScaler for mixed precision training.

    Returns:
        Tuple of (average loss, accuracy).

    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(loader)
    smooth = 0.05  # Label smoothing factor

    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        with autocast():
            outputs = model(images)
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)

            # Label smoothing
            labels_sm = labels * (1 - smooth) + 0.5 * smooth

            loss = criterion(outputs, labels_sm)
            loss = loss / DEIT_ACCUMULATION_STEPS  # Scale for gradient accumulation

        scaler.scale(loss).backward()

        # Gradient accumulation step
        if (batch_idx + 1) % DEIT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Accumulate statistics (use original labels for accuracy)
        total_loss += loss.item() * DEIT_ACCUMULATION_STEPS
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # Handle final partial accumulation window.
    if total_batches % DEIT_ACCUMULATION_STEPS != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg_loss = total_loss / total_batches
    accuracy = correct / total
    return avg_loss, accuracy


def _validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Validation pass.

    Args:
        model: Model to validate.
        loader: DataLoader for validation data.
        criterion: Loss criterion.
        device: Device to run on.

    Returns:
        Tuple of (average loss, accuracy, AUC).

    """
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels_dev = labels.to(device).unsqueeze(1)
            outputs = model(images)
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            loss = criterion(outputs, labels_dev)
            total_loss += loss.item()
            probs = torch.sigmoid(outputs).squeeze()
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    avg_loss = total_loss / len(loader)
    accuracy = (all_preds == all_labels).mean()
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # Happens when val set contains only one class.
        print("  [WARN] Validation AUC is undefined for single-class labels; using 0.5.")
        auc = 0.5

    return avg_loss, accuracy, auc


def _log_epoch(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    val_auc: float,
    best_val_auc: float,
) -> None:
    """Print concise single-line epoch progress.

    Args:
        epoch: Current epoch number (0-indexed).
        total_epochs: Total number of epochs planned.
        train_loss: Average training loss.
        train_acc: Training accuracy.
        val_loss: Validation loss.
        val_acc: Validation accuracy.
        val_auc: Validation AUC-ROC.
        best_val_auc: Best validation AUC achieved so far.

    """
    done = epoch + 1
    pct = done / total_epochs * 100.0
    is_best = val_auc > best_val_auc
    marker = " | NEW BEST" if is_best else ""
    print(
        f"\rTraining progress: {pct:5.1f}% ({done}/{total_epochs})"
        f" | val_auc={val_auc:.4f}{marker}",
        end="",
        flush=True,
    )
