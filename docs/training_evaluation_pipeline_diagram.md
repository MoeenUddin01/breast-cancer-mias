# Training and Evaluation Pipeline Diagram

## End-to-End ML Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE ML PIPELINE FLOW                                    │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                       │
    ┌──────────────────────────────────┼──────────────────────────────────┐
    │                                  │                                  │
    ▼                                  ▼                                  ▼
┌─────────────┐                ┌─────────────┐                    ┌─────────────┐
│   INPUT     │                │   TRAINING  │                    │  EVALUATION │
│  (Data)     │───────────────▶│   (Train)   │───────────────────▶│   (Test)    │
└─────────────┘                └─────────────┘                    └─────────────┘
                                       │                                  │
                                       ▼                                  ▼
                              ┌─────────────┐                    ┌─────────────┐
                              │  ARTIFACTS  │                    │   OUTPUTS   │
                              │  (Models)   │                    │  (Reports)  │
                              └─────────────┘                    └─────────────┘
```

## Training Pipeline Detail

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         src/training/trainer.py                                      │
│                           train(model, train_loader, val_loader, ...)             │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Setup  (trainer.py:87-113)                                                  │
│  ────────────                                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  Config (SimpleNamespace)  ← User provides via config object               │    │
│  │  ├── LEARNING_RATE: float    (e.g., 1e-4)                                   │    │
│  │  ├── EPOCHS: int             (e.g., 50)                                    │    │
│  │  ├── PATIENCE: int           (e.g., 10)                                   │    │
│  │  └── DEVICE: torch.device    (cuda/cpu)                                   │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                       │                                              │
│  ┌────────────────────────────────────┴─────────────────────────────────────────┐   │
│  │  Model + Optimizer + Loss  (trainer.py:107-113)                                 │  │
│  │  ├─▶ model.to(DEVICE)                                                         │  │
│  │  ├─▶ optimizer = Adam(lr=LEARNING_RATE)                                      │  │
│  │  └─▶ criterion = BCEWithLogitsLoss(pos_weight=computed_from_data)             │  │
│  └───────────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Training Loop  (trainer.py:128-145)                                          │
│  ───────────────────────────────────────────────────                                  │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  FOR EACH BATCH (images, labels) in train_loader:  (trainer.py:133-143)      │    │
│  │                                                                             │    │
│  │  1. Forward:    outputs = model(images)                                     │    │
│  │  2. Loss:       loss = criterion(outputs, labels)                          │    │
│  │  3. Backward:   loss.backward()                                             │    │
│  │  4. Step:       optimizer.step()                                            │    │
│  │                                                                             │    │
│  │  Accumulate: epoch_loss += loss.item()                                      │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                       │                                              │
│                                       ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  COMPUTE: avg_train_loss = epoch_loss / len(train_loader)  (trainer.py:145) │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Validation  (validator.py:15-91)                                           │
│  ─────────────────────────────────────────────────                                  │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  model.eval() + torch.no_grad()  (validator.py:58)                            │    │
│  │                                                                             │    │
│  │  FOR EACH BATCH (images, labels) in val_loader:  (validator.py:64-75)        │    │
│  │    ├─▶ outputs = model(images)                                              │    │
│  │    ├─▶ loss = criterion(outputs, labels)                    ──┐            │    │
│  │    ├─▶ probs = sigmoid(outputs)                              │            │    │
│  │    ├─▶ preds = probs >= 0.5                                  │ Metrics     │    │
│  │    └─▶ Collect: all_preds, all_labels, all_probs             │ Computation │    │
│  │                                                              ◄───          │    │
│  │  COMPUTE:                                                    │            │    │
│  │    ├─▶ val_loss = avg(losses)                                │            │    │
│  │    ├─▶ val_accuracy = accuracy_score(labels, preds) ◄────────┘            │    │
│  │    └─▶ val_auc = roc_auc_score(labels, probs)                              │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: Logging & Early Stopping  (trainer.py:151-168)                           │
│  ───────────────────────────────────────                                            │
│                                                                                      │
│  Console Output:                                                                     │
│  "Epoch 5/50 - Train Loss: 0.2341, Val Loss: 0.1982, Val Acc: 87.50%, Val AUC: 0.9234"│
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  EarlyStopping (callbacks.py:16-125)                                         │    │
│  │  ├─ mode: "max" (monitoring val_auc)                                          │    │
│  │  ├─ patience: config.PATIENCE                                                │    │
│  │  └─ save_path: artifacts/{model_name}/                                         │    │
│  │                                                                             │    │
│  │  IF val_auc improved:                                                        │    │
│  │    ├─▶ counter = 0                                                          │    │
│  │    ├─▶ best_score = val_auc                                                 │    │
│  │    └─▶ SAVE: torch.save(model.state_dict(), {model_name}_best.pth)         │    │
│  │        "EarlyStopping: Saved new best model to artifacts/..."                  │    │
│  │  ELSE:                                                                       │    │
│  │    └─▶ counter += 1                                                         │    │
│  │                                                                             │    │
│  │  IF counter >= patience:                                                     │    │
│  │    └─▶ "Early stopping triggered at epoch {epoch}"                           │    │
│  │    └─▶ break loop                                                            │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: Save Final Artifacts  (trainer.py:28-84)                                  │
│  ─────────────────────────────                                                      │
│                                                                                      │
│  _save_model_and_results()  ← Called from trainer.py:172                             │
│  │                                                                                   │
│  ├─▶ artifacts/{model_name}/  ← Created by trainer.py:47                              │
│  │   ├─ {model_name}_final.pth     ← Final model weights                           │
│  │   ├─ {model_name}_best.pth      ← Best validation AUC checkpoint                 │
│  │   └─ results.json               ← Training history + metrics                     │
│  │       {                                                                          │
│  │         "model_name": "resnet",                                                  │
│  │         "epochs_trained": 35,                                                    │
│  │         "final_train_loss": 0.1234,                                            │
│  │         "final_val_auc": 0.9234,                                                 │
│  │         "best_val_auc": 0.9312,                                                 │
│  │         "config": { "learning_rate": 0.0001, ... },                              │
│  │         "history": { "train_loss": [...], "val_auc": [...] }                     │
│  │       }                                                                          │
│  │                                                                                  │
│  └─▶ "Model saved to artifacts/resnet/resnet_final.pth"                             │
│  └─▶ "Results saved to artifacts/resnet/results.json"                               │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Evaluation Pipeline Detail

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         evaluator.py  (evaluate:15-91)                               │
│                      evaluate(model, dataloader, device, model_name)                │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Inference  (evaluator.py:67-85)                                             │
│  ────────────────                                                                    │
│  model.eval() + torch.no_grad()                                                       │
│                                                                                      │
│  FOR EACH BATCH:                                                                      │
│    ├─▶ outputs = model(images)                                                      │
│    ├─▶ probs = sigmoid(outputs)          ← Probabilities for AUC                    │
│    ├─▶ preds = (probs >= 0.5).int()      ← Binary predictions for accuracy/F1        │
│    └─▶ Collect: all_predictions, all_probabilities, all_labels                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Metrics Computation  (evaluator.py:87-96)                                   │
│  ─────────────────────────                                                           │
│                                                                                      │
│  ┌─────────────────────────┐  ┌─────────────────────────┐  ┌────────────────────────┐│
│  │     accuracy_score       │  │      roc_auc_score      │  │     f1_score           ││
│  │         ┌───┐            │  │         ┌───┐            │  │         ┌───┐          ││
│  │  preds ─▶│ = │──▶ 0.8750  │  │  probs ─▶│ = │──▶ 0.9234  │  │  preds ─▶│ = │──▶ 0.82 ││
│  │  labels ─▶│   │          │  │  labels ─▶│   │          │  │  labels ─▶│   │        ││
│  │         └───┘            │  │         └───┘            │  │         └───┘          ││
│  └─────────────────────────┘  └─────────────────────────┘  └────────────────────────┘│
│                                                                                      │
│  ┌─────────────────────────┐  ┌─────────────────────────┐                           │
│  │    precision_score      │  │      recall_score       │                           │
│  │         ┌───┐            │  │         ┌───┐            │                           │
│  │  preds ─▶│ = │──▶ 0.85   │  │  preds ─▶│ = │──▶ 0.80   │                           │
│  │  labels ─▶│   │          │  │  labels ─▶│   │          │                           │
│  │         └───┘            │  │         └───┘            │                           │
│  └─────────────────────────┘  └─────────────────────────┘                           │
│                                                                                      │
│  classification_report() with target_names=["No Cancer", "Cancer"]                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Output Generation  (evaluator.py:104-144)                                   │
│  ─────────────────────────                                                           │
│                                                                                      │
│  Console Print:                                                                      │
│  ═══════════════════════════════════════════════════                                  │
│  Evaluation Results: resnet                                                         │
│  ═══════════════════════════════════════════════════                                  │
│  Accuracy:  0.8750                                                                   │
│  AUC-ROC:   0.9234                                                                   │
│  F1-Score:  0.8200                                                                   │
│  Precision: 0.8500                                                                   │
│  Recall:    0.8000                                                                   │
│  ═══════════════════════════════════════════════════                                  │
│                                                                                      │
│  Classification Report:                                                              │
│                precision    recall  f1-score   support                              │
│  No Cancer       0.9000     0.9167    0.9082        12                               │
│  Cancer          0.8333     0.8000    0.8163         5                               │
│                                                                                      │
│  File Write:  (evaluator.py:125-144)                                                 │
│  └─▶ outputs/reports/{model_name}_report.txt                                         │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Visualization Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         visualizer.py                                                │
│  plot_training_history:18-78  plot_confusion_matrix:81-136  plot_model_comparison:139-212 │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
          ▼                            ▼                            ▼
┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
│ plot_training_    │      │ plot_confusion_   │      │ plot_model_       │
│ history()         │      │ matrix()          │      │ comparison()      │
│ (18-78)           │      │ (81-136)          │      │ (139-212)         │
│                   │      │                   │      │                   │
│ Input: history    │      │ Input: y_true,    │      │ Input: model_names│
│ Output:           │      │        y_pred     │      │        accuracies │
│ {model}_history.png│      │ Output:           │      │        aucs       │
│                   │      │ {model}_confusion │      │ Output:           │
│ ┌───────────────┐ │      │ _matrix.png       │      │ model_comparison  │
│ │  Loss Curves  │ │      │                   │      │ .png              │
│ │  ───────────  │ │      │ ┌───────────────┐ │      │                   │
│ │  train │────  │ │      │ │    Pred       │ │      │ ┌───────────────┐ │
│ │  val   │────  │ │      │ │   N    C      │ │      │ │  Accuracy vs  │ │
│ └───────────────┘ │      │ │ N  TN   FP    │ │      │ │  AUC-ROC      │ │
│                   │      │ │ C  FN   TP    │ │      │ │  ───────────  │ │
│ ┌───────────────┐ │      │ └───────────────┘ │      │ │  ███ ████     │ │
│ │  Val Accuracy │ │      │                   │      │ │  ███ ████     │ │
│ │  ───────────  │ │      │ N = No Cancer     │      │ └───────────────┘ │
│ │     │────     │ │      │ C = Cancer        │      │                   │
│ └───────────────┘ │      │ TN/TP = correct   │      │  ResNet EffNet  │
│                   │      │ FP/FN = errors    │      │   Xcept           │
└───────────────────┘      └───────────────────┘      └───────────────────┘
```

## File Dependencies

```
trainer.py
    ├── src/data/dataset.py ────────────────┐
    │   └── MIASDataset + DataLoader        │
    ├── src/models/{resnet,efficientnet,    │
    │   xception}_model.py ────────────┐    │  INPUT
    │   └── create_*_model()            │    │  (Model)
    ├── src/training/validator.py ──────┤    │
    │   └── validate()                   │    │
    ├── src/training/callbacks.py ──────┤    │
    │   └── EarlyStopping                │    │
    └── config (LEARNING_RATE, EPOCHS,   │    │
        PATIENCE, DEVICE) ◄──────────────┘    │
                                              │
evaluator.py                                │
    ├── src/models/{model}.py ────────────────┤
    ├── src/data/dataset.py ─────────────────┤
    └── sklearn.metrics ─────────────────────┘
                                              │
visualizer.py
    ├── matplotlib.pyplot
    ├── seaborn
    └── sklearn.metrics.confusion_matrix
```

## Output Locations Summary

```
artifacts/                          outputs/
│                                   │
├── resnet/                         ├── models/
│   ├── resnet_final.pth            ├── plots/
│   ├── resnet_best.pth             │   ├── resnet_history.png
│   └── results.json                │   ├── resnet_confusion_matrix.png
│                                   │   ├── efficientnet_history.png
├── efficientnet/                   │   ├── xception_history.png
│   ├── efficientnet_final.pth      │   └── model_comparison.png
│   ├── efficientnet_best.pth       └── reports/
│   └── results.json                    ├── resnet_report.txt
│                                       ├── efficientnet_report.txt
└── xception/                           └── xception_report.txt
    ├── xception_final.pth
    ├── xception_best.pth
    └── results.json
```

## Command to Run Training

```bash
cd /home/moeenuddin/Desktop/Deep_learning/breast-cancer-mias
source .venv/bin/activate

# Train a specific model
python -m src.pipelines.resnet_model_training
python -m src.pipelines.efficientnet_model_training
python -m src.pipelines.xception_model_training
```

## Command to Evaluate

```python
import torch
from src.models.resnet_model import create_resnet_model
from src.data.dataset import create_test_dataloader
from src.evaluation.evaluator import evaluate
from src.evaluation.visualizer import (
    plot_training_history,
    plot_confusion_matrix,
)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_resnet_model()
model.load_state_dict(torch.load("artifacts/resnet/resnet_best.pth"))

# Evaluate
results = evaluate(model, test_loader, device, "resnet")

# Visualize
plot_training_history(history, "resnet")
plot_confusion_matrix(y_true, y_pred, "resnet")
```
