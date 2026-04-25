"""Verify BreakHis dataset structure and counts."""

from __future__ import annotations

import os


def verify_breakhis_dataset(base_dir: str | None = None) -> dict[str, int]:
    """Verify BreakHis 400X dataset structure and return counts.

    Args:
        base_dir: Path to BreakHis breast/ directory. If None, uses default.

    Returns:
        Dictionary with benign, malignant, and total counts.

    """
    if base_dir is None:
        base_dir = "/kaggle/input/breakhis/BreaKHis_v1/histology_slides/breast/"

    benign_400x = []
    malignant_400x = []

    benign_path = os.path.join(base_dir, "benign/SOB/")
    malignant_path = os.path.join(base_dir, "malignant/SOB/")

    for root, _dirs, files in os.walk(benign_path):
        if "400X" in root:
            benign_400x.extend([f for f in files if f.endswith(".png")])

    for root, _dirs, files in os.walk(malignant_path):
        if "400X" in root:
            malignant_400x.extend([f for f in files if f.endswith(".png")])

    benign_count = len(benign_400x)
    malignant_count = len(malignant_400x)
    total = benign_count + malignant_count

    print("=" * 50)
    print("  BREAKHIS 400X DATASET COUNTS")
    print("=" * 50)
    print(f"  Benign images    : {benign_count}")
    print(f"  Malignant images : {malignant_count}")
    print(f"  Total            : {total}")
    print(f"  Imbalance ratio  : 1:{malignant_count / benign_count:.1f}")
    print()

    # After split (85/15)
    train_est = int(total * 0.85)
    test_est = int(total * 0.15)
    print("  AFTER SPLIT (85/15):")
    print(f"  Train images     : ~{train_est}")
    print(f"  Test images      : ~{test_est}")
    print()

    # After augmentation (5x)
    aug_est = train_est * 5
    print("  AFTER AUGMENTATION (5x):")
    print(f"  Train images     : ~{aug_est}")
    print(f"  Train batches    : ~{aug_est // 32}")
    print(f"  Test batches     : ~{test_est // 32}")
    print("=" * 50)

    return {
        "benign": benign_count,
        "malignant": malignant_count,
        "total": total,
    }


if __name__ == "__main__":
    verify_breakhis_dataset()
