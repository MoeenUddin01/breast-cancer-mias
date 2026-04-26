"""Training logic specifically for ViT-B/16.

Separate from CNN trainer to avoid any interference.
Uses same verbose output style as CNN trainer.
"""

from __future__ import annotations

import time

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.transformers.vit_config import (
    VIT_EPOCHS,
    VIT_PATIENCE,
    VIT_PHASE1_EPOCHS,
    VIT_PHASE1_LR,
    VIT_PHASE2_LR_BACKBONE,
    VIT_PHASE2_LR_HEAD,
    VIT_UNFREEZE_BLOCKS,
    VIT_WEIGHT_DECAY,
)
from src.transformers.vit_model import unfreeze_vit_blocks


def _compute_pos_weight(train_loader, device):
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
    pos_weight = torch.tensor([num_negatives / num_positives]).to(device)
    print(
        f"  pos_weight: {pos_weight.item():.4f} "
        f"(neg={num_negatives}, pos={num_positives})"
    )
    return pos_weight


def train_vit(model, train_loader, val_loader, model_name, device):
    """Two-phase training for ViT-B/16.

    Phase 1: Train head only (5 epochs, lr=1e-3)
    Phase 2: Fine-tune last 4 blocks (remaining, lr=1e-5/1e-4)

    Args:
        model: The ViT model to train.
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

    # ── PHASE 1: Head only ───────────────
    print(
        f"\n  Phase 1: Training head only "
        f"for {VIT_PHASE1_EPOCHS} epochs "
        f"(lr={VIT_PHASE1_LR})"
    )

    optimizer_p1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=VIT_PHASE1_LR,
        weight_decay=VIT_WEIGHT_DECAY,
    )

    for epoch in range(VIT_PHASE1_EPOCHS):
        train_loss, train_acc = _train_epoch(
            model, train_loader, optimizer_p1, criterion, device, epoch
        )
        val_loss, val_acc, val_auc = _validate(model, val_loader, criterion, device)
        _log_epoch(
            epoch,
            VIT_PHASE1_EPOCHS + VIT_EPOCHS,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_auc,
            best_val_auc,
            optimizer_p1,
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

    # ── PHASE 2: Fine-tune last 4 blocks ─
    print(
        f"\n  Phase 2: Fine-tuning last "
        f"{VIT_UNFREEZE_BLOCKS} transformer blocks"
    )
    unfreeze_vit_blocks(model, VIT_UNFREEZE_BLOCKS)

    backbone_params = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and "head" not in name
    ]
    head_params = [p for p in model.head.parameters()]

    optimizer_p2 = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": VIT_PHASE2_LR_BACKBONE},
            {"params": head_params, "lr": VIT_PHASE2_LR_HEAD},
        ],
        weight_decay=VIT_WEIGHT_DECAY,
    )

    # CosineAnnealingWarmRestarts — better than ReduceLROnPlateau
    # for transformers
    scheduler = CosineAnnealingWarmRestarts(
        optimizer_p2, T_0=10, T_mult=2, eta_min=1e-7
    )

    for epoch in range(VIT_PHASE1_EPOCHS, VIT_PHASE1_EPOCHS + VIT_EPOCHS):
        train_loss, train_acc = _train_epoch(
            model, train_loader, optimizer_p2, criterion, device, epoch
        )
        val_loss, val_acc, val_auc = _validate(model, val_loader, criterion, device)
        scheduler.step(epoch)
        current_lr = optimizer_p2.param_groups[0]["lr"]

        _log_epoch(
            epoch,
            VIT_PHASE1_EPOCHS + VIT_EPOCHS,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_auc,
            best_val_auc,
            optimizer_p2,
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
                "learning_rate": current_lr,
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
            print(f"  ✓ New best saved: AUC={val_auc:.4f}")
        else:
            no_improve += 1

        if no_improve >= VIT_PATIENCE:
            print(f"  Early stopping at epoch {epoch + 1}")
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

    print(f"\n  Training completed in {train_time:.1f} minutes")
    print(f"  Best validation AUC: {best_val_auc:.4f} " f"(epoch {best_epoch})")

    return history, best_epoch, train_time


def _train_epoch(model, loader, optimizer, criterion, device, epoch):
    """Single training epoch with progress bar.

    Args:
        model: Model to train.
        loader: DataLoader for training data.
        optimizer: Optimizer instance.
        criterion: Loss criterion.
        device: Device to run on.
        epoch: Current epoch number.

    Returns:
        Tuple of (average loss, accuracy).

    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    total_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0 or batch_idx == total_batches - 1:
            avg_loss = total_loss / (batch_idx + 1)
            avg_acc = correct / total * 100
            pct = (batch_idx + 1) / total_batches
            filled = int(20 * pct)
            bar = "█" * filled + "░" * (20 - filled)
            print(
                f"\r  [{bar}] {pct * 100:.1f}%  "
                f"batch {batch_idx + 1}/{total_batches}  "
                f"│  loss={avg_loss:.4f}  "
                f"acc={avg_acc:.2f}%",
                end="",
                flush=True,
            )

    print()
    avg_loss = total_loss / total_batches
    accuracy = correct / total
    return avg_loss, accuracy


def _validate(model, loader, criterion, device):
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
    auc = roc_auc_score(all_labels, all_probs)

    return avg_loss, accuracy, auc


def _log_epoch(
    epoch,
    total_epochs,
    train_loss,
    train_acc,
    val_loss,
    val_acc,
    val_auc,
    best_val_auc,
    optimizer,
):
    """Print epoch results box.

    Args:
        epoch: Current epoch number (0-indexed).
        total_epochs: Total number of epochs planned.
        train_loss: Average training loss.
        train_acc: Training accuracy.
        val_loss: Validation loss.
        val_acc: Validation accuracy.
        val_auc: Validation AUC-ROC.
        best_val_auc: Best validation AUC achieved so far.
        optimizer: Optimizer instance (for learning rate display).

    """
    is_best = val_auc > best_val_auc
    marker = "  ⭐ NEW BEST!" if is_best else ""
    print(
        f"""
╔═══════════════════════════════════════╗
║  EPOCH {epoch + 1}/{total_epochs} RESULTS{marker:<14}║
╠═══════════════════════════════════════╣
║  METRIC        TRAIN        VAL       ║
║  Loss     {train_loss:>10.4f}  {val_loss:>10.4f}  ║
║  Accuracy {train_acc:>10.4f}  {val_acc:>10.4f}  ║
║  AUC              —  {val_auc:>10.4f}  ║
╚═══════════════════════════════════════╝"""
    )
