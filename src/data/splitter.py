"""Dataset splitter for BreakHis breast cancer data.

Splits data by unique patient ID BEFORE augmentation to prevent
data leakage between train and test sets. One patient has multiple
images, so splitting by image ID would leak patient data.
"""

from __future__ import annotations

from sklearn.model_selection import train_test_split


def split_by_patient_id(
    data: list[tuple[str, str, int]],
    test_size: float,
    seed: int,
) -> tuple[list[tuple[str, str, int]], list[tuple[str, str, int]]]:
    """Split data by unique patient IDs with stratification.

    Performs a stratified train/test split at the patient ID level to
    ensure no patient_id appears in both train and test sets.
    This prevents data leakage since one patient has multiple images.

    Args:
        data: List of (patient_id, image_path, label) tuples from loader.py.
        test_size: Fraction of patients to use for the test set (0.0 to 1.0).
        seed: Random seed for reproducibility, passed to train_test_split
            as random_state.

    Returns:
        Tuple containing:
            - train_data: List of (patient_id, image_array, label) for training.
            - test_data: List of (patient_id, image_path, label) for testing.

    Raises:
        ValueError: If data is empty or if split results in leakage.

    """
    if not data:
        raise ValueError("Data cannot be empty")

    try:
        # Group images by base patient_id (without magnification suffix)
        # to ensure all magnifications of same patient stay together
        patient_images: dict[str, list[tuple[str, str, int]]] = {}
        patient_labels: dict[str, int] = {}

        for patient_id, image_path, label in data:
            # Extract base patient ID (e.g., "14-4659" from "14-4659_400X")
            base_patient_id = patient_id.split("_")[0]
            if base_patient_id not in patient_images:
                patient_images[base_patient_id] = []
                patient_labels[base_patient_id] = label
            patient_images[base_patient_id].append((patient_id, image_path, label))

        unique_patients = list(patient_images.keys())
        patient_label_list = [patient_labels[pid] for pid in unique_patients]

        if len(unique_patients) == 0:
            raise ValueError("No unique patient IDs found in data")

        if len(set(patient_label_list)) < 2:
            print(f"[WARNING] Only one class found ({set(patient_label_list)}), stratification disabled")
            stratify = None
        else:
            stratify = patient_label_list

        # Perform stratified split on unique patient IDs
        train_patients, test_patients = train_test_split(
            unique_patients,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except Exception as e:
        print(f"[ERROR] Failed to split data: {e}")
        raise

    try:
        # Convert to sets for O(1) lookup
        train_patient_set = set(train_patients)
        test_patient_set = set(test_patients)

        # Verify no overlap
        overlap = train_patient_set & test_patient_set
        if overlap:
            raise ValueError(f"Data leakage detected: {overlap} appear in both sets")

        # Collect all images for train and test patients
        train_data: list[tuple[str, str, int]] = []
        test_data: list[tuple[str, str, int]] = []

        for patient_id in train_patients:
            train_data.extend(patient_images[patient_id])

        for patient_id in test_patients:
            test_data.extend(patient_images[patient_id])

    except Exception as e:
        print(f"[ERROR] Failed to assign data to train/test sets: {e}")
        raise

    print(f"Unique patients in train: {len(train_patients)}")
    print(f"Unique patients in test: {len(test_patients)}")
    print(f"Total images in train: {len(train_data)}")
    print(f"Total images in test: {len(test_data)}")

    return train_data, test_data
