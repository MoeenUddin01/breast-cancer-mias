"""Data loader for BreakHis breast cancer dataset.

Handles loading of RGB PNG histology images from BreakHis dataset structure.
Uses 400X magnification images only.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.utils import config


def _extract_patient_id(filename: str) -> str:
    """Extract patient ID from BreakHis filename.

    Filename format: SOB_B_TA-14-4659-400-001.png
    Patient ID is between 3rd and 5th dash: "14-4659"

    Args:
        filename: The image filename.

    Returns:
        The extracted patient ID.

    """
    parts = filename.replace(".png", "").split("-")
    if len(parts) >= 5:
        return f"{parts[2]}-{parts[3]}"
    return filename


def load_data(
    data_dir: str | None = None,
    magnifications: list[str] | None = None,
) -> list[tuple[str, np.ndarray, int]]:
    """Load BreakHis dataset from directory structure.

    Walks through data_dir/benign/SOB/ and data_dir/malignant/SOB/
    recursively to find all PNG images in folders matching the
    specified magnifications. Patient ID includes magnification suffix
    to differentiate same-patient images at different magnifications.

    Args:
        data_dir: Root directory containing benign/ and malignant/ folders.
            If None, uses config.DATA_DIR.
        magnifications: List of magnification levels to load.
            If None, uses config.MAGNIFICATIONS.

    Returns:
        List of tuples containing:
            - patient_id (str): Extracted patient ID with magnification suffix
                (e.g., "14-4659_400X")
            - image_array (np.ndarray): RGB image loaded with OpenCV
            - label (int): Binary label (0 for benign, 1 for malignant)

    Raises:
        FileNotFoundError: If data_dir does not exist.

    """
    if data_dir is None:
        data_dir = config.DATA_DIR

    if magnifications is None:
        magnifications = config.MAGNIFICATIONS

    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    data: list[tuple[str, np.ndarray, int]] = []
    mag_counts: dict[str, dict[str, int]] = {
        mag: {"benign": 0, "malignant": 0} for mag in magnifications
    }
    base_patient_ids: set[str] = set()

    # Process benign images
    benign_path = data_path / "benign" / "SOB"
    if benign_path.exists():
        for image_file in benign_path.rglob("*.png"):
            for mag in magnifications:
                if mag in str(image_file.parent):
                    try:
                        img = cv2.imread(str(image_file))
                        if img is None:
                            print(f"[WARNING] Failed to load image: {image_file}")
                            continue
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        base_patient_id = _extract_patient_id(image_file.name)
                        patient_id = f"{base_patient_id}_{mag}"
                        data.append((patient_id, img, 0))
                        base_patient_ids.add(base_patient_id)
                        mag_counts[mag]["benign"] += 1
                    except Exception as e:
                        print(f"[ERROR] Failed to process {image_file}: {e}")
                    break  # Only count once per magnification

    # Process malignant images
    malignant_path = data_path / "malignant" / "SOB"
    if malignant_path.exists():
        for image_file in malignant_path.rglob("*.png"):
            for mag in magnifications:
                if mag in str(image_file.parent):
                    try:
                        img = cv2.imread(str(image_file))
                        if img is None:
                            print(f"[WARNING] Failed to load image: {image_file}")
                            continue
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        base_patient_id = _extract_patient_id(image_file.name)
                        patient_id = f"{base_patient_id}_{mag}"
                        data.append((patient_id, img, 1))
                        base_patient_ids.add(base_patient_id)
                        mag_counts[mag]["malignant"] += 1
                    except Exception as e:
                        print(f"[ERROR] Failed to process {image_file}: {e}")
                    break  # Only count once per magnification

    # Print per-magnification counts
    print("\n" + "=" * 50)
    print("  BREAKHIS DATASET COUNTS BY MAGNIFICATION")
    print("=" * 50)
    total_benign = 0
    total_malignant = 0
    for mag in magnifications:
        b_count = mag_counts[mag]["benign"]
        m_count = mag_counts[mag]["malignant"]
        total_benign += b_count
        total_malignant += m_count
        print(f"  {mag:4s} → benign: {b_count:4d}  malignant: {m_count:4d}")
    print("-" * 50)
    print(f"  Total → benign: {total_benign:4d}  malignant: {total_malignant:4d}  patients: {len(base_patient_ids):4d}")
    print("=" * 50)

    return data
