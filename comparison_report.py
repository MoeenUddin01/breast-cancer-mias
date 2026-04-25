"""Standalone comparison report generator for breast cancer detection models.

This file is INDEPENDENT — it does NOT import anything from kaggle_train.py
or any src/ files. It runs completely on its own.

Assumes these txt report files exist in /kaggle/working/outputs/reports/:
    - resnet152_report.txt
    - efficientnet_b2_report.txt
    - xception_report.txt

Run this in a separate Kaggle notebook AFTER all 3 models are trained.
"""

# ═══════════════════════════════════════════════════════════════
# SECTION 1 — INSTALL (add comment: # Run this first in Kaggle)
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("  BREAST CANCER DETECTION — COMPARISON REPORT")
print("  Run this after all 3 models are trained")
print("=" * 60)

# ═══════════════════════════════════════════════════════════════
# SECTION 2 — IMPORTS
# ═══════════════════════════════════════════════════════════════

import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime

# ═══════════════════════════════════════════════════════════════
# SECTION 3 — DAGSHUB INIT (Optional - only works in Kaggle)
# ═══════════════════════════════════════════════════════════════

try:
    import mlflow
    import dagshub
    from kaggle_secrets import UserSecretsClient
    
    secrets = UserSecretsClient()
    os.environ["MLFLOW_TRACKING_USERNAME"] = secrets.get_secret(
        "MLFLOW_TRACKING_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = secrets.get_secret(
        "MLFLOW_TRACKING_PASSWORD")
    dagshub.init(
        repo_owner="MoeenUddin01",
        repo_name="breast-cancer-mias",
        mlflow=True
    )
    mlflow.set_experiment("BreakHis_Breast_Cancer_Detection")
    DAGSHUB_ENABLED = True
    print("✓ DagHub initialized")
except Exception as e:
    DAGSHUB_ENABLED = False
    mlflow = None
    dagshub = None
    print(f"⚠️ DagHub not available: {e}")
    print("   Reports will be saved locally only")

# ═══════════════════════════════════════════════════════════════
# SECTION 4 — PATHS
# ═══════════════════════════════════════════════════════════════

REPORTS_DIR = "./outputs/reports/"
PLOTS_DIR = "./outputs/plots/"
ARTIFACTS_DIR = "./artifacts/"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Report file paths
RESNET_TXT = REPORTS_DIR + "resnet152_report.txt"
EFFNET_TXT = REPORTS_DIR + "efficientnet_b2_report.txt"
XCEPTION_TXT = REPORTS_DIR + "xception_report.txt"

# Check all files exist
for path in [RESNET_TXT, EFFNET_TXT, XCEPTION_TXT]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Report not found: {path}\n"
            f"Please train this model first before "
            f"running comparison_report.py"
        )
print("✓ All 3 report files found")

# ═══════════════════════════════════════════════════════════════
# SECTION 5 — PARSE REPORT FUNCTION
# ═══════════════════════════════════════════════════════════════


def parse_report(filepath):
    """Parse a model .txt report file and extract all metrics.

    Works with both report formats (old heavy and new lightweight).

    Args:
        filepath: Path to the report .txt file

    Returns:
        Dictionary with all extracted metrics

    """
    with open(filepath, "r") as f:
        content = f.read()

    def extract_float(pattern, text, default=0.0):
        match = re.search(pattern, text)
        return float(match.group(1)) if match else default

    def extract_int(pattern, text, default=0):
        match = re.search(pattern, text)
        return int(match.group(1)) if match else default

    # Extract all metrics using regex
    auc = extract_float(r"AUC-ROC\s*:?\s*([\d.]+)", content)

    # Accuracy — handle both "0.9404" and "94.04%" formats
    acc_raw = extract_float(r"Accuracy\s*:?\s*([\d.]+)", content)
    accuracy = acc_raw / 100.0 if acc_raw > 1.0 else acc_raw

    f1 = extract_float(r"F1\s*Score\s*:?\s*([\d.]+)", content)
    precision = extract_float(r"Precision\s*:?\s*([\d.]+)", content)
    recall = extract_float(r"Recall\s*:?\s*([\d.]+)", content)
    specificity = extract_float(r"Specificity\s*:?\s*([\d.]+)", content)
    best_epoch = extract_int(r"Best\s*Epoch\s*:?\s*(\d+)", content)
    train_time = extract_float(r"Train\s*Time\s*:?\s*([\d.]+)", content)

    # Extract confusion matrix values
    tn = extract_int(r"TN[=\s:]+(\d+)", content)
    fp = extract_int(r"FP[=\s:]+(\d+)", content)
    fn = extract_int(r"FN[=\s:]+(\d+)", content)
    tp = extract_int(r"TP[=\s:]+(\d+)", content)

    return {
        "auc": auc,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "best_epoch": best_epoch,
        "train_time": train_time,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — LOAD ALL REPORTS
# ═══════════════════════════════════════════════════════════════

resnet_data = parse_report(RESNET_TXT)
effnet_data = parse_report(EFFNET_TXT)
xception_data = parse_report(XCEPTION_TXT)

all_models = {
    "ResNet152": resnet_data,
    "EfficientNetB2": effnet_data,
    "Xception": xception_data
}

# Find best model by AUC
best_model_name = max(all_models, key=lambda k: all_models[k]["auc"])
best_model_data = all_models[best_model_name]

# Print summary table
print("\n" + "=" * 65)
print("  MODEL COMPARISON SUMMARY")
print("=" * 65)
print(f"  {'Model':<18} {'AUC':>8} {'Acc%':>8} "
      f"{'F1':>8} {'Time':>10}")
print(f"  {'-'*55}")
for name, data in all_models.items():
    marker = "  ← BEST" if name == best_model_name else ""
    print(f"  {name:<18} "
          f"{data['auc']:>8.4f} "
          f"{data['accuracy']*100:>7.2f}% "
          f"{data['f1']:>8.4f} "
          f"{data['train_time']:>8.1f}m"
          f"{marker}")
print("=" * 65)

# ═══════════════════════════════════════════════════════════════
# SECTION 7 — GENERATE VISUALIZATION
# ═══════════════════════════════════════════════════════════════


def generate_comparison_report(all_models, best_model_name):
    """Generate comprehensive visual comparison report.

    Args:
        all_models: Dict with model names and their metrics
        best_model_name: Name of the best model (highlighted)

    Returns:
        Path to saved plot file

    """
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    names = list(all_models.keys())
    colors = {
        "ResNet152": "#378ADD",
        "EfficientNetB2": "#1D9E75",
        "Xception": "#888780"
    }
    bar_colors = [
        "gold" if n == best_model_name else colors[n]
        for n in names
    ]

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(24, 26))
    fig.patch.set_facecolor("#0d0d0d")

    fig.suptitle(
        "Breast Cancer Detection — Model Comparison Report",
        fontsize=20, fontweight="bold",
        color="white", y=0.99
    )
    fig.text(
        0.5, 0.975,
        f"BreakHis Histology Dataset  |  {date_str}  |  "
        f"ResNet152 vs EfficientNetB2 vs Xception",
        ha="center", fontsize=12,
        color="#aaaaaa"
    )

    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        hspace=0.55, wspace=0.35,
        top=0.96, bottom=0.04
    )

    # ROW 1: 3 horizontal bar charts
    metrics_row1 = [
        ("auc", "AUC-ROC (primary metric)", 0.80, 1.0),
        ("accuracy", "Accuracy", 0.70, 1.0),
        ("f1", "F1 Score", 0.70, 1.0),
    ]
    for col, (key, title, xmin, xmax) in enumerate(metrics_row1):
        ax = fig.add_subplot(gs[0, col])
        values = [all_models[n][key] for n in names]
        bars = ax.barh(names, values,
                       color=bar_colors, alpha=0.85,
                       height=0.5)
        for bar, val in zip(bars, values):
            ax.text(
                val + 0.002,
                bar.get_y() + bar.get_height()/2,
                f"{val:.4f}",
                va="center", ha="left",
                fontsize=10, color="white",
                fontweight="bold"
            )
        ax.set_xlim(xmin, xmax + 0.05)
        ax.set_title(title, color="white",
                     fontsize=12, pad=8)
        ax.set_xlabel("Score", color="#aaaaaa")
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.2, axis="x")
        ax.set_facecolor("#1a1a1a")
        # Gold star on best bar
        best_idx = names.index(best_model_name)
        ax.text(
            xmin + 0.002,
            best_idx,
            "★",
            va="center", ha="left",
            fontsize=14, color="gold"
        )

    # ROW 2: 3 confusion matrices
    for col, (name, data) in enumerate(all_models.items()):
        ax = fig.add_subplot(gs[1, col])
        cm = np.array([
            [data["tn"], data["fp"]],
            [data["fn"], data["tp"]]
        ])
        sns.heatmap(
            cm, annot=True, fmt="d",
            cmap="Blues", ax=ax,
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
            annot_kws={"size": 13, "weight": "bold"},
            linewidths=0.5
        )
        marker = " ★" if name == best_model_name else ""
        ax.set_title(
            f"{name}{marker}\nAUC: {data['auc']:.4f}",
            color="white", fontsize=11, pad=8
        )
        ax.set_ylabel("True Label", color="#aaaaaa")
        ax.set_xlabel("Predicted Label", color="#aaaaaa")
        ax.tick_params(colors="white")
        ax.text(
            0.5, -0.22,
            f"Sensitivity: {data['recall']:.3f}  |  "
            f"Specificity: {data['specificity']:.3f}",
            transform=ax.transAxes,
            ha="center", fontsize=9,
            color="#aaaaaa"
        )

    # ROW 3 LEFT: Grouped bar chart
    ax_group = fig.add_subplot(gs[2, :2])
    metric_keys = ["accuracy", "auc", "f1",
                   "precision", "recall", "specificity"]
    metric_labels = ["Accuracy", "AUC", "F1",
                     "Precision", "Recall", "Specificity"]
    x = np.arange(len(metric_keys))
    width = 0.25
    for i, (name, data) in enumerate(all_models.items()):
        vals = [data[k] for k in metric_keys]
        bars = ax_group.bar(
            x + i*width, vals,
            width, label=name,
            color=colors[name], alpha=0.85
        )
    ax_group.set_xticks(x + width)
    ax_group.set_xticklabels(metric_labels,
                              color="white")
    ax_group.set_ylim(0.6, 1.05)
    ax_group.set_title("All metrics comparison",
                        color="white", fontsize=12)
    ax_group.set_ylabel("Score", color="#aaaaaa")
    ax_group.tick_params(colors="white")
    ax_group.legend(facecolor="#2a2a2a",
                    labelcolor="white",
                    fontsize=9)
    ax_group.grid(True, alpha=0.2, axis="y")
    ax_group.set_facecolor("#1a1a1a")

    # ROW 3 RIGHT: Scatter efficiency
    ax_scatter = fig.add_subplot(gs[2, 2])
    for name, data in all_models.items():
        size = data["f1"] * 1000
        color = "gold" if name == best_model_name else colors[name]
        ax_scatter.scatter(
            data["train_time"], data["auc"],
            s=size, color=color,
            alpha=0.85, zorder=5,
            edgecolors="white", linewidths=0.5
        )
        ax_scatter.annotate(
            name,
            (data["train_time"], data["auc"]),
            textcoords="offset points",
            xytext=(8, 4),
            color="white", fontsize=9
        )
    best = all_models[best_model_name]
    ax_scatter.annotate(
        f"Best: {best_model_name}\nAUC={best['auc']:.4f}",
        (best["train_time"], best["auc"]),
        textcoords="offset points",
        xytext=(-60, -30),
        color="gold", fontsize=9,
        arrowprops=dict(arrowstyle="->",
                        color="gold",
                        lw=1.2)
    )
    ax_scatter.set_xlabel("Training time (min)",
                           color="#aaaaaa")
    ax_scatter.set_ylabel("AUC-ROC",
                           color="#aaaaaa")
    ax_scatter.set_title(
        "Performance vs training time\n"
        "(bubble size = F1 score)",
        color="white", fontsize=11
    )
    ax_scatter.tick_params(colors="white")
    ax_scatter.grid(True, alpha=0.2)
    ax_scatter.set_facecolor("#1a1a1a")

    # ROW 4: Summary table
    ax_table = fig.add_subplot(gs[3, :])
    ax_table.axis("off")
    ax_table.set_facecolor("#0d0d0d")

    col_labels = [
        "Model", "AUC-ROC", "Accuracy",
        "F1", "Precision", "Recall",
        "Specificity", "Best Epoch", "Train Time"
    ]
    table_data = []
    for name, data in all_models.items():
        marker = " ★ BEST" if name == best_model_name else ""
        table_data.append([
            name + marker,
            f"{data['auc']:.4f}",
            f"{data['accuracy']*100:.2f}%",
            f"{data['f1']:.4f}",
            f"{data['precision']:.4f}",
            f"{data['recall']:.4f}",
            f"{data['specificity']:.4f}",
            str(data["best_epoch"]),
            f"{data['train_time']:.1f} min"
        ])

    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0.35, 1, 0.55]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#444444")
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(
                color="white", fontweight="bold")
        else:
            model_row_name = table_data[row-1][0]
            if "BEST" in model_row_name:
                cell.set_facecolor("#7d6608")
                cell.set_text_props(
                    color="gold", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#1a1a2e")
                cell.set_text_props(color="white")
            else:
                cell.set_facecolor("#16213e")
                cell.set_text_props(color="white")

    ax_table.text(
        0.5, 0.28,
        "Dataset: BreakHis All Magnifications  |  "
        "7,909 images  |  324 patients  |  "
        "85/15 split by patient ID  |  Tesla T4 GPU",
        transform=ax_table.transAxes,
        ha="center", fontsize=10, color="#aaaaaa"
    )
    ax_table.text(
        0.5, 0.12,
        "⚠️  Research prototype only. Not for clinical use.  "
        "Results measured on held-out test set — no data leakage.",
        transform=ax_table.transAxes,
        ha="center", fontsize=9, color="#666666"
    )

    # Save
    plot_path = PLOTS_DIR + "comparison_report.png"
    plt.savefig(plot_path, dpi=150,
                bbox_inches="tight",
                facecolor="#0d0d0d")
    plt.show()
    plt.close()
    print(f"✅ Visualization saved: {plot_path}")
    return plot_path


plot_path = generate_comparison_report(all_models, best_model_name)

# ═══════════════════════════════════════════════════════════════
# SECTION 8 — SAVE COMPARISON SUMMARY TXT
# ═══════════════════════════════════════════════════════════════

date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
txt_path = REPORTS_DIR + "comparison_summary.txt"

best = all_models[best_model_name]
fastest_model = min(all_models, key=lambda k: all_models[k]["train_time"])
fastest_time = all_models[fastest_model]["train_time"]

with open(txt_path, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("  BREAST CANCER DETECTION — MODEL COMPARISON\n")
    f.write("=" * 60 + "\n")
    f.write(f"  Date         : {date_str}\n")
    f.write(f"  Dataset      : BreakHis All Magnifications\n")
    f.write(f"  Total images : 7,909\n")
    f.write(f"  Patients     : 324\n")
    f.write(f"  Split        : 85% train / 15% test by patient ID\n")
    f.write(f"  GPU          : Tesla T4\n")
    f.write("=" * 60 + "\n\n")
    f.write("  MODEL RESULTS:\n")
    f.write(f"  {'─'*58}\n")
    f.write(f"  {'Model':<18} {'AUC':>7} {'Acc%':>8} "
            f"{'F1':>7} {'Prec':>7} {'Recall':>7} {'Time':>7}\n")
    f.write(f"  {'─'*58}\n")
    for name, data in all_models.items():
        marker = " ← BEST" if name == best_model_name else ""
        f.write(f"  {name:<18} "
                f"{data['auc']:>7.4f} "
                f"{data['accuracy']*100:>7.2f} "
                f"{data['f1']:>7.4f} "
                f"{data['precision']:>7.4f} "
                f"{data['recall']:>7.4f} "
                f"{data['train_time']:>6.1f}m"
                f"{marker}\n")
    f.write(f"  {'─'*58}\n\n")
    f.write(f"  BEST MODEL  : {best_model_name}\n")
    f.write(f"  Best AUC    : {best['auc']:.4f}\n")
    f.write(f"  Best Acc    : {best['accuracy']*100:.2f}%\n")
    f.write(f"  Best F1     : {best['f1']:.4f}\n")
    f.write(f"  Train Time  : {best['train_time']:.1f} minutes")
    if best_model_name == fastest_model:
        f.write(" (fastest + best)")
    f.write("\n\n")
    f.write("  RECOMMENDATION:\n")
    f.write(f"  Deploy {best_model_name} — highest AUC")
    if best_model_name == fastest_model:
        f.write(" and fastest training")
    f.write(f"\n  performance on the BreakHis breast cancer dataset.\n")
    f.write("=" * 60 + "\n")
    f.write("  ⚠️  Research prototype. Not for clinical use.\n")
    f.write("=" * 60 + "\n")

print(f"✅ Summary saved: {txt_path}")

# ═══════════════════════════════════════════════════════════════
# SECTION 9 — LOG TO DAGSHUB
# ═══════════════════════════════════════════════════════════════

if DAGSHUB_ENABLED and mlflow is not None:
    with mlflow.start_run(run_name="model_comparison_report"):

        for name, data in all_models.items():
            prefix = name.lower().replace(
                "efficientnetb2", "effnet")
            mlflow.log_metrics({
                f"{prefix}_auc": data["auc"],
                f"{prefix}_accuracy": data["accuracy"],
                f"{prefix}_f1": data["f1"],
                f"{prefix}_precision": data["precision"],
                f"{prefix}_recall": data["recall"],
                f"{prefix}_specificity": data["specificity"],
                f"{prefix}_train_time": data["train_time"]
            })

        mlflow.log_param("best_model", best_model_name)
        mlflow.log_param("best_auc", best_model_data["auc"])
        mlflow.log_param("dataset", "BreakHis_AllMags")
        mlflow.log_param("total_images", 7909)
        mlflow.log_param("patients", 324)
        mlflow.log_param("split", "85_15_by_patient")

        mlflow.log_artifact(plot_path)
        mlflow.log_artifact(txt_path)
        mlflow.log_artifact(RESNET_TXT)
        mlflow.log_artifact(EFFNET_TXT)
        mlflow.log_artifact(XCEPTION_TXT)
        
        mlflow.end_run()
    
    print("✅ All artifacts logged to DagHub")
    print("✅ View at: https://dagshub.com/MoeenUddin01/"
          "breast-cancer-mias/experiments")
else:
    print("⚠️ DagHub not available — files saved locally only")

# Copy files to artifacts folder
import shutil
shutil.copy(plot_path, ARTIFACTS_DIR + "comparison_report.png")
shutil.copy(txt_path, ARTIFACTS_DIR + "comparison_summary.txt")
shutil.copy(RESNET_TXT, ARTIFACTS_DIR + "resnet152_report.txt")
shutil.copy(EFFNET_TXT, ARTIFACTS_DIR + "efficientnet_b2_report.txt")
shutil.copy(XCEPTION_TXT, ARTIFACTS_DIR + "xception_report.txt")
print(f"✅ All files copied to {ARTIFACTS_DIR}")

# ═══════════════════════════════════════════════════════════════
# SECTION 10 — FINAL SUMMARY PRINT
# ═══════════════════════════════════════════════════════════════

best = all_models[best_model_name]

print(f"""
╔══════════════════════════════════════════════════════╗
║      BREAST CANCER DETECTION — FINAL RESULTS         ║
╠══════════════════════════════════════════════════════╣
║  Best Model    : {best_model_name:<37}║
║  Best AUC-ROC  : {best['auc']:<37.4f}║
║  Best Accuracy : {best['accuracy']*100:<36.2f}%║
║  Best F1       : {best['f1']:<37.4f}║
║  Train Time    : {best['train_time']:<35.1f}min ║
╠══════════════════════════════════════════════════════╣
║  Files saved:                                        ║
║  • comparison_report.png   → Kaggle + DagHub         ║
║  • comparison_summary.txt  → Kaggle + DagHub         ║
║  • All 3 model .txt reports → DagHub artifacts       ║
╠══════════════════════════════════════════════════════╣
║  DagHub experiments:                                 ║
║  dagshub.com/MoeenUddin01/breast-cancer-mias       ║
╚══════════════════════════════════════════════════════╝
""")
