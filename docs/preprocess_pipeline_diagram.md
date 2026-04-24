# Preprocess and Save Pipeline Diagram

## Data Flow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    src/pipelines/preprocess_and_save.py                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT: Raw Data                                                             │
│  dataset/all-mias/                                                           │
│  ├── Info.txt              ← Metadata (image_id, severity, etc.)            │
│  ├── mdb001.pgm            ← Raw grayscale mammogram                        │
│  ├── mdb002.pgm                                                            │
│  ├── ... (119 more PGM files)                                              │
│  └── processed/            ← Will be created/overwritten                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Load Raw Data (loader.py)                                           │
│  ─────────────────────────────────                                           │
│  • Parse Info.txt → extract (image_id, label) pairs                          │
│  • Load PGM files with OpenCV                                               │
│  • Filter: Keep only B (benign=0) and M (malignant=1)                       │
│  • Output: List of tuples (image_id, image_array, label)                     │
│                                                                              │
│  Example:                                                                  │
│  [("mdb001", array(...), 0),  # benign                                      │
│   ("mdb002", array(...), 1),  # malignant                                    │
│   ...]                                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Split by Image ID (splitter.py)                                    │
│  ─────────────────────────────────────                                       │
│  • Extract unique image IDs                                                 │
│  • Stratified split: 80% train, 20% test                                    │
│  • Verify: No image_id appears in both sets                                 │
│  • Output: train_data (80%), test_data (20%)                                │
│                                                                              │
│  ┌─────────────┐    ┌─────────────┐                                          │
│  │  TRAIN 80%  │    │   TEST 20%  │                                          │
│  │  (~96 imgs) │    │  (~24 imgs) │                                          │
│  └─────────────┘    └─────────────┘                                          │
└─────────────────────────────────────────────────────────────────────────────┘
           │                            │
           ▼                            ▼
┌────────────────────────────────┐  ┌────────────────────────────────┐
│  STEP 3A: Process TRAIN Split   │  │  STEP 3B: Process TEST Split  │
│  ─────────────────────────────  │  │  ──────────────────────────── │
│                                 │  │                                │
│  for each image:                │  │  for each image:               │
│    ├─ apply_clahe()            │  │    ├─ apply_clahe()           │
│    │   └─ CLAHE enhancement   │  │    │   └─ CLAHE enhancement    │
│    ├─ preprocess_image()      │  │    ├─ preprocess_image()      │
│    │   ├─ grayscale→3ch       │  │    │   ├─ grayscale→3ch       │
│    │   ├─ resize to 224x224  │  │    │   ├─ resize to 224x224   │
│    │   └─ normalize [0,1]    │  │    │   └─ normalize [0,1]     │
│    ├─ convert to uint8         │  │    ├─ convert to uint8        │
│    └─ save as PNG              │  │    └─ save as PNG             │
│                                 │  │                                │
└────────────────────────────────┘  └────────────────────────────────┘
           │                            │
           ▼                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: Processed Data                                                      │
│  dataset/all-mias/processed/                                                 │
│  ├── train/                      ← 80% of processed images                   │
│  │   ├── mdb001.png              ← CLAHE + resized + 3-channel              │
│  │   ├── mdb003.png                                                          │
│  │   └── ... (~96 PNG files)                                                 │
│  └── test/                       ← 20% of processed images                    │
│      ├── mdb002.png              ← CLAHE + resized + 3-channel              │
│      ├── mdb005.png                                                          │
│      └── ... (~24 PNG files)                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Safety Features

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         ANTI-LEAKAGE SAFEGUARDS                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. SPLIT BEFORE PREPROCESSING                                               │
│     ┌─────────┐    ┌─────────┐    ┌─────────┐                                │
│     │  LOAD   │───▶│  SPLIT  │───▶│ PROCESS │                                │
│     └─────────┘    └─────────┘    └─────────┘                                │
│                                                                             │
│  2. ID-LEVEL SPLIT (not sample-level)                                       │
│     • Each unique image_id goes to ONLY train OR test                       │
│     • No duplicate images across splits                                      │
│                                                                             │
│  3. VERIFICATION STEP                                                        │
│     • Explicit check: train_ids ∩ test_ids = ∅                              │
│     • Raises error if any overlap detected                                   │
│                                                                             │
│  4. PHYSICAL SEPARATION                                                       │
│     • Train and test saved to different directories                         │
│     • Test data never touched during training pipeline                      │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

## Function Call Chain

```
preprocess_and_save()
│
├─▶ load_data()
│   └─▶ open Info.txt
│   └─▶ cv2.imread() for each PGM
│
├─▶ split_by_image_id() ───────────────────────────┐
│   ├─▶ extract unique IDs                         │
│   ├─▶ train_test_split(stratify=labels)          │
│   └─▶ verify no overlap ◄─── DATA LEAKAGE CHECK  │
│                                                  │
├─▶ for train_data: ◄──────────────────────────────┘
│   ├─▶ apply_clahe(image)
│   ├─▶ preprocess_image(enhanced, image_size)
│   └─▶ cv2.imwrite(train_path/image.png)
│
└─▶ for test_data:
    ├─▶ apply_clahe(image)
    ├─▶ preprocess_image(enhanced, image_size)
    └─▶ cv2.imwrite(test_path/image.png)
```

## File Dependencies

```
preprocess_and_save.py
    ├── src/data/loader.py ───────────┐
    │   └── Info.txt + *.pgm          │
    ├── src/data/splitter.py ──────────┤
    │   └── sklearn.train_test_split  │ INPUT DATA
    ├── src/data/preprocessor.py ──────┤    (raw)
    │   ├── apply_clahe()             │
    │   └── preprocess_image()        │
    └── src/utils/config_loader.py ───┘
        └── src/utils/config.yaml
```

## Configuration Used

```yaml
# From config.yaml
data:
  dir: "dataset/all-mias/"              ← INPUT
  processed_dir: "dataset/all-mias/processed/"  ← OUTPUT

image:
  size: [224, 224]                      ← Resize target

training:
  test_size: 0.2                        ← 20% test, 80% train
  seed: 42                              ← Reproducibility
```

## Command to Run

```bash
cd /home/moeenuddin/Desktop/Deep_learning/breast-cancer-mias
source .venv/bin/activate
python -m src.pipelines.preprocess_and_save
```
