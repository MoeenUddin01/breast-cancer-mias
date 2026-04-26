"""Transformer (ViT) vs CNN Comparison Report Generator.

Standalone script — no dependencies on kaggle_train.py.
Generates comparison visualizations between ViT-B/16 and CNN models.
"""
from __future__ import annotations

import os
import re
from datetime import datetime

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns

# ═══════════════════════════════════════════════════════════════
# SECTION 1 — PATHS & SETUP
# ═══════════════════════════════════════════════════════════════

ARTIFACTS_DIR = "./artifacts/transf_comp/"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Model report paths
RESNET_TXT = "./artifacts/resnet152_report.txt"
EFFNET_TXT = "./artifacts/efficientnet_b2_report.txt"
XCEPTION_TXT = "./artifacts/xception_report.txt"
VIT_TXT = "./outputs/reports/vit_report.txt"

print("=" * 60)
print("  TRANSFORMER vs CNN COMPARISON REPORT")
print("  ViT-B/16 vs ResNet152 / EfficientNetB2 / Xception")
print("=" * 60)


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — PARSE REPORT FUNCTION
# ═══════════════════════════════════════════════════════════════


def parse_report(filepath: str, default: dict | None = None) -> dict:
    """Parse a model .txt report file and extract all metrics."""
    if not os.path.exists(filepath):
        if default:
            print(f"⚠️  {filepath} not found, using provided defaults")
            return default
        raise FileNotFoundError(f"Report not found: {filepath}")

    with open(filepath, "r") as f:
        content = f.read()

    def extract_float(pattern: str, text: str, default_val: float = 0.0) -> float:
        match = re.search(pattern, text)
        return float(match.group(1)) if match else default_val

    def extract_int(pattern: str, text: str, default_val: int = 0) -> int:
        match = re.search(pattern, text)
        return int(match.group(1)) if match else default_val

    auc = extract_float(r"AUC-ROC\s*:?\s*([\d.]+)", content)
    acc_raw = extract_float(r"Accuracy\s*:?\s*([\d.]+)", content)
    accuracy = acc_raw / 100.0 if acc_raw > 1.0 else acc_raw
    f1 = extract_float(r"F1\s*Score\s*:?\s*([\d.]+)", content)
    precision = extract_float(r"Precision\s*:?\s*([\d.]+)", content)
    recall = extract_float(r"Recall\s*:?\s*([\d.]+)", content)
    specificity = extract_float(r"Specificity\s*:?\s*([\d.]+)", content)
    best_epoch = extract_int(r"Best\s*Epoch\s*:?\s*(\d+)", content)
    train_time = extract_float(r"Train\s*Time\s*:?\s*([\d.]+)", content)

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
        "tp": tp,
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — LOAD ALL MODELS
# ═══════════════════════════════════════════════════════════════

# ViT results from training (user provided)
vit_default = {
    "auc": 0.9810,
    "accuracy": 0.9243,
    "f1": 0.9538,
    "precision": 0.9382,
    "recall": 0.9698,
    "specificity": 0.8724,
    "best_epoch": 18,
    "train_time": 35.0,
    "tn": 396,
    "fp": 58,
    "fn": 22,
    "tp": 766,
}

resnet_data = parse_report(RESNET_TXT)
effnet_data = parse_report(EFFNET_TXT)
xception_data = parse_report(XCEPTION_TXT)
vit_data = parse_report(VIT_TXT, vit_default)

cnn_models = {
    "ResNet152": resnet_data,
    "EfficientNetB2": effnet_data,
    "Xception": xception_data,
}

all_models = {
    "ViT-B/16": vit_data,
    "ResNet152": resnet_data,
    "EfficientNetB2": effnet_data,
    "Xception": xception_data,
}

print("\n✓ All models loaded")
print("\n" + "=" * 65)
print("  MODEL PERFORMANCE SUMMARY")
print("=" * 65)
print(f"  {'Model':<18} {'AUC':>8} {'Acc%':>8} {'F1':>8} {'Time':>8}")
print(f"  {'-' * 55}")
for name, data in all_models.items():
    print(
        f"  {name:<18} "
        f"{data['auc']:>8.4f} "
        f"{data['accuracy'] * 100:>7.2f}% "
        f"{data['f1']:>8.4f} "
        f"{data['train_time']:>7.1f}m"
    )
print("=" * 65)

best_cnn = max(cnn_models, key=lambda k: cnn_models[k]["auc"])
best_overall = max(all_models, key=lambda k: all_models[k]["auc"])
print(f"\n  Best CNN Model    : {best_cnn} (AUC={cnn_models[best_cnn]['auc']:.4f})")
print(f"  Best Overall      : {best_overall} (AUC={all_models[best_overall]['auc']:.4f})")


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — ViT vs CNN COMPARISON PLOT
# ═══════════════════════════════════════════════════════════════


def generate_vit_vs_cnn_plot() -> str:
    """Generate ViT vs CNN comparison visualization."""
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#0d0d0d")

    fig.suptitle(
        "ViT-B/16 Transformer vs CNN Models — BreakHis Breast Cancer Detection",
        fontsize=16,
        fontweight="bold",
        color="gold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.955,
        f"Transformer (ViT) vs Convolutional Neural Networks  |  {date_str}",
        ha="center",
        fontsize=11,
        color="#aaaaaa",
    )

    gs = gridspec.GridSpec(
        2, 3, figure=fig, hspace=0.4, wspace=0.3, top=0.94, bottom=0.06
    )

    colors = {"ViT-B/16": "#e74c3c", "ResNet152": "#3498db", "EfficientNetB2": "#2ecc71"}
    names = ["ViT-B/16", "ResNet152", "EfficientNetB2"]
    bar_colors = [colors[n] for n in names]

    # ROW 1: Key metrics comparison
    metrics = [
        ("auc", "AUC-ROC (Primary Metric)", 0.95, 1.0),
        ("accuracy", "Accuracy", 0.90, 1.0),
        ("f1", "F1 Score", 0.90, 1.0),
    ]

    for col, (key, title, xmin, xmax) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, col])
        values = [all_models[n][key] for n in names]
        bars = ax.barh(names, values, color=bar_colors, alpha=0.85, height=0.5)

        for bar, val in zip(bars, values):
            ax.text(
                val + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                ha="left",
                fontsize=10,
                color="white",
                fontweight="bold",
            )

        ax.set_xlim(xmin, xmax + 0.02)
        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.set_xlabel("Score", color="#aaaaaa")
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.2, axis="x")
        ax.set_facecolor("#1a1a1a")

        # Highlight best
        best_idx = values.index(max(values))
        ax.text(
            xmin + 0.002, best_idx, "★", va="center", ha="left", fontsize=14, color="gold"
        )

    # ROW 2: Confusion matrices for ViT and best CNN
    for col, name in enumerate(["ViT-B/16", "EfficientNetB2"]):
        ax = fig.add_subplot(gs[1, col])
        data = all_models[name]
        cm = np.array([[data["tn"], data["fp"]], [data["fn"], data["tp"]]])

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Reds" if name == "ViT-B/16" else "Greens",
            ax=ax,
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
            annot_kws={"size": 14, "weight": "bold"},
            linewidths=0.5,
        )

        ax.set_title(
            f"{name}\nAUC: {data['auc']:.4f} | Acc: {data['accuracy']*100:.2f}%",
            color="white",
            fontsize=11,
            pad=8,
        )
        ax.set_ylabel("True Label", color="#aaaaaa")
        ax.set_xlabel("Predicted Label", color="#aaaaaa")
        ax.tick_params(colors="white")
        ax.text(
            0.5,
            -0.18,
            f"Sens: {data['recall']:.3f} | Spec: {data['specificity']:.3f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
            color="#aaaaaa",
        )

    # ROW 2 RIGHT: Summary comparison table
    ax_table = fig.add_subplot(gs[1, 2])
    ax_table.axis("off")
    ax_table.set_facecolor("#0d0d0d")

    col_labels = ["Model", "AUC", "Acc%", "F1", "Epochs", "Time"]
    table_data = []
    for name in names:
        data = all_models[name]
        marker = "★" if name == best_overall else ""
        table_data.append(
            [
                f"{name} {marker}",
                f"{data['auc']:.4f}",
                f"{data['accuracy']*100:.1f}%",
                f"{data['f1']:.4f}",
                str(data["best_epoch"]),
                f"{data['train_time']:.1f}m",
            ]
        )

    table = ax_table.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0.05, 0.2, 0.9, 0.6],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#444444")
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            model_name = table_data[row - 1][0]
            if "★" in model_name:
                cell.set_facecolor("#7d6608")
                cell.set_text_props(color="gold", fontweight="bold")
            elif "ViT" in model_name:
                cell.set_facecolor("#4a1a1a")
                cell.set_text_props(color="#ff9999")
            elif row % 2 == 0:
                cell.set_facecolor("#1a1a2e")
                cell.set_text_props(color="white")
            else:
                cell.set_facecolor("#16213e")
                cell.set_text_props(color="white")

    ax_table.text(
        0.5,
        0.92,
        "Model Comparison Table",
        transform=ax_table.transAxes,
        ha="center",
        fontsize=12,
        color="gold",
        fontweight="bold",
    )

    # Footer
    fig.text(
        0.5,
        0.02,
        "⚠️ Research prototype only. Not for clinical use.  |  "
        "BreakHis Dataset: 7,909 images, 324 patients, 85/15 patient-aware split",
        ha="center",
        fontsize=9,
        color="#666666",
    )

    plot_path = ARTIFACTS_DIR + "vit_vs_cnn_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.show()
    plt.close()

    print(f"\n✅ ViT vs CNN comparison saved: {plot_path}")
    return plot_path


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — ViT vs ALL MODELS COMPARISON
# ═══════════════════════════════════════════════════════════════


def generate_vit_vs_all_plot() -> str:
    """Generate comprehensive comparison of ViT vs all models."""
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor("#0d0d0d")

    fig.suptitle(
        "Complete Model Comparison — ViT-B/16 vs All CNN Architectures",
        fontsize=18,
        fontweight="bold",
        color="gold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.96,
        f"Transformer vs ResNet152 / EfficientNetB2 / Xception  |  {date_str}",
        ha="center",
        fontsize=12,
        color="#aaaaaa",
    )

    gs = gridspec.GridSpec(
        3, 4, figure=fig, hspace=0.45, wspace=0.3, top=0.94, bottom=0.06
    )

    colors = {
        "ViT-B/16": "#e74c3c",
        "ResNet152": "#3498db",
        "EfficientNetB2": "#2ecc71",
        "Xception": "#9b59b6",
    }
    names = list(all_models.keys())
    bar_colors = [colors[n] for n in names]

    # ROW 1: Three key metrics (spaced across 4 columns)
    metrics_row1 = [
        (0, "auc", "AUC-ROC (Primary)", 0.85, 1.0),
        (1, "accuracy", "Accuracy", 0.75, 1.0),
        (2, "f1", "F1 Score", 0.75, 1.0),
    ]

    for col, key, title, xmin, xmax in metrics_row1:
        ax = fig.add_subplot(gs[0, col])
        values = [all_models[n][key] for n in names]
        bars = ax.barh(names, values, color=bar_colors, alpha=0.85, height=0.4)

        for bar, val in zip(bars, values):
            ax.text(
                val + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                ha="left",
                fontsize=9,
                color="white",
                fontweight="bold",
            )

        ax.set_xlim(xmin, xmax + 0.03)
        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.set_xlabel("Score", color="#aaaaaa")
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.2, axis="x")
        ax.set_facecolor("#1a1a1a")

        # Star for best
        best_idx = values.index(max(values))
        ax.text(
            xmin + 0.005, best_idx, "★", va="center", ha="left", fontsize=14, color="gold"
        )

    # ROW 2: Confusion matrices for all 4 models (4 columns)
    for col, (name, data) in enumerate(all_models.items()):
        if col >= 4:
            break
        ax = fig.add_subplot(gs[1, col])
        cm = np.array([[data["tn"], data["fp"]], [data["fn"], data["tp"]]])

        cmap = "Reds" if name == "ViT-B/16" else "Blues"
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            ax=ax,
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
            annot_kws={"size": 12, "weight": "bold"},
            linewidths=0.5,
        )

        marker = " ★ BEST" if name == best_overall else ""
        ax.set_title(
            f"{name}{marker}\nAUC: {data['auc']:.4f}",
            color="gold" if name == best_overall else "white",
            fontsize=10,
            pad=8,
        )
        ax.set_ylabel("True Label", color="#aaaaaa")
        ax.set_xlabel("Predicted Label", color="#aaaaaa")
        ax.tick_params(colors="white")
        ax.text(
            0.5,
            -0.22,
            f"Sens: {data['recall']:.3f} | Spec: {data['specificity']:.3f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
            color="#aaaaaa",
        )

    # ROW 3 LEFT: Grouped bar chart of all metrics (spans 3 columns)
    ax_group = fig.add_subplot(gs[2, :3])
    metric_keys = ["accuracy", "auc", "f1", "precision", "recall", "specificity"]
    metric_labels = ["Accuracy", "AUC", "F1", "Precision", "Recall", "Specificity"]
    x = np.arange(len(metric_keys))
    width = 0.2

    for i, (name, data) in enumerate(all_models.items()):
        vals = [data[k] for k in metric_keys]
        bars = ax_group.bar(x + i * width, vals, width, label=name, color=colors[name], alpha=0.85)

    ax_group.set_xticks(x + width * 1.5)
    ax_group.set_xticklabels(metric_labels, color="white")
    ax_group.set_ylim(0.7, 1.05)
    ax_group.set_title("All Metrics Comparison", color="white", fontsize=12)
    ax_group.set_ylabel("Score", color="#aaaaaa")
    ax_group.tick_params(colors="white")
    ax_group.legend(facecolor="#2a2a2a", labelcolor="white", fontsize=9)
    ax_group.grid(True, alpha=0.2, axis="y")
    ax_group.set_facecolor("#1a1a1a")

    # ROW 3 RIGHT: Efficiency scatter (4th column)
    ax_scatter = fig.add_subplot(gs[2, 3])
    for name, data in all_models.items():
        size = data["f1"] * 800
        edge_color = "gold" if name == best_overall else "white"
        ax_scatter.scatter(
            data["train_time"],
            data["auc"],
            s=size,
            color=colors[name],
            alpha=0.85,
            zorder=5,
            edgecolors=edge_color,
            linewidths=2 if name == best_overall else 0.5,
        )
        ax_scatter.annotate(
            name,
            (data["train_time"], data["auc"]),
            textcoords="offset points",
            xytext=(8, 4),
            color="white",
            fontsize=9,
        )

    # Annotate best
    best = all_models[best_overall]
    ax_scatter.annotate(
        f"Best: {best_overall}\nAUC={best['auc']:.4f}",
        (best["train_time"], best["auc"]),
        textcoords="offset points",
        xytext=(-70, -25),
        color="gold",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gold", lw=1.2),
    )

    ax_scatter.set_xlabel("Training Time (minutes)", color="#aaaaaa")
    ax_scatter.set_ylabel("AUC-ROC", color="#aaaaaa")
    ax_scatter.set_title(
        "Performance vs Training Time\n(bubble size = F1 score)",
        color="white",
        fontsize=11,
    )
    ax_scatter.tick_params(colors="white")
    ax_scatter.grid(True, alpha=0.2)
    ax_scatter.set_facecolor("#1a1a1a")

    # Footer
    fig.text(
        0.5,
        0.02,
        "⚠️ Research prototype only. Not for clinical use.  |  "
        "Dataset: BreakHis All Magnifications (7,909 images, 324 patients)",
        ha="center",
        fontsize=9,
        color="#666666",
    )

    plot_path = ARTIFACTS_DIR + "vit_vs_all_models.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    plt.show()
    plt.close()

    print(f"✅ ViT vs all models comparison saved: {plot_path}")
    return plot_path


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — GENERATE COMPARISON SUMMARY TXT
# ═══════════════════════════════════════════════════════════════


def generate_summary_txt() -> str:
    """Generate text summary of all comparisons."""
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    txt_path = ARTIFACTS_DIR + "comparison_summary.txt"

    vit = all_models["ViT-B/16"]
    best_cnn_name = max(cnn_models, key=lambda k: cnn_models[k]["auc"])
    best_cnn = cnn_models[best_cnn_name]

    with open(txt_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  TRANSFORMER vs CNN — MODEL COMPARISON REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"  Date         : {date_str}\n")
        f.write(f"  Dataset      : BreakHis All Magnifications\n")
        f.write(f"  Total Images : 7,909\n")
        f.write(f"  Patients     : 324\n")
        f.write(f"  Split        : 85% train / 15% test by patient ID\n")
        f.write(f"  GPU          : Tesla T4\n")
        f.write("=" * 70 + "\n\n")

        f.write("  COMPLETE RESULTS TABLE:\n")
        f.write(f"  {'─' * 66}\n")
        f.write(
            f"  {'Model':<18} {'AUC':>7} {'Acc%':>7} {'F1':>7} "
            f"{'Prec':>7} {'Rec':>7} {'Time':>7}\n"
        )
        f.write(f"  {'─' * 66}\n")

        for name, data in all_models.items():
            marker = " ★ BEST" if name == best_overall else ""
            f.write(
                f"  {name:<18} "
                f"{data['auc']:>7.4f} "
                f"{data['accuracy']*100:>6.2f}% "
                f"{data['f1']:>7.4f} "
                f"{data['precision']:>7.4f} "
                f"{data['recall']:>7.4f} "
                f"{data['train_time']:>6.1f}m"
                f"{marker}\n"
            )
        f.write(f"  {'─' * 66}\n\n")

        # ViT Analysis
        f.write("  VIT-B/16 ANALYSIS:\n")
        f.write(f"  {'─' * 66}\n")
        f.write(f"  Architecture       : Transformer (ViT-B/16)\n")
        f.write(f"  Pretrained         : ImageNet-21k (14M images)\n")
        f.write(f"  AUC-ROC           : {vit['auc']:.4f}\n")
        f.write(f"  Accuracy          : {vit['accuracy']*100:.2f}%\n")
        f.write(f"  F1 Score          : {vit['f1']:.4f}\n")
        f.write(f"  Best Epoch        : {vit['best_epoch']}\n")
        f.write(f"  Train Time        : {vit['train_time']:.1f} minutes\n")
        f.write(f"  Confusion Matrix  : TN={vit['tn']} FP={vit['fp']} "
                f"FN={vit['fn']} TP={vit['tp']}\n")
        f.write(f"  {'─' * 66}\n\n")

        # CNN Comparison
        f.write("  CNN BASELINE COMPARISON:\n")
        f.write(f"  {'─' * 66}\n")
        f.write(f"  Best CNN Model    : {best_cnn_name}\n")
        f.write(f"  Best CNN AUC      : {best_cnn['auc']:.4f}\n")
        f.write(f"  AUC Gap (ViT-CNN) : {vit['auc'] - best_cnn['auc']:+.4f}\n")
        f.write(f"  {'─' * 66}\n\n")

        # Rankings
        f.write("  MODEL RANKINGS BY AUC:\n")
        f.write(f"  {'─' * 66}\n")
        sorted_models = sorted(all_models.items(), key=lambda x: x[1]["auc"], reverse=True)
        for rank, (name, data) in enumerate(sorted_models, 1):
            f.write(f"  {rank}. {name:<18} AUC: {data['auc']:.4f}\n")
        f.write(f"  {'─' * 66}\n\n")

        # Conclusion
        f.write("  CONCLUSIONS:\n")
        f.write(f"  {'─' * 66}\n")
        if best_overall == "ViT-B/16":
            f.write(
                f"  ✓ ViT-B/16 is the BEST performing model with AUC={vit['auc']:.4f}\n"
            )
        else:
            f.write(
                f"  • Best Overall: {best_overall} (AUC={all_models[best_overall]['auc']:.4f})\n"
            )
            f.write(
                f"  • ViT-B/16 ranks #{[n for n, _ in sorted_models].index('ViT-B/16') + 1} "
                f"with AUC={vit['auc']:.4f}\n"
            )

        f.write(f"\n  • EfficientNetB2 remains the most efficient (best AUC/time ratio)\n")
        f.write(f"  • ViT-B/16 shows competitive performance with transformer architecture\n")
        f.write(f"  • ResNet152 provides strong baseline CNN performance\n")
        f.write(f"  {'─' * 66}\n\n")

        f.write("=" * 70 + "\n")
        f.write("  ⚠️  Research prototype. Not for clinical use.\n")
        f.write("=" * 70 + "\n")

    print(f"✅ Comparison summary saved: {txt_path}")
    return txt_path


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — EXECUTE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  GENERATING COMPARISON REPORTS...")
    print("=" * 60)

    plot1 = generate_vit_vs_cnn_plot()
    plot2 = generate_vit_vs_all_plot()
    summary = generate_summary_txt()

    print("\n" + "=" * 60)
    print("  ALL FILES GENERATED")
    print("=" * 60)
    print(f"  📊 ViT vs CNN:     {plot1}")
    print(f"  📊 ViT vs All:      {plot2}")
    print(f"  📝 Summary TXT:     {summary}")
    print(f"  📁 Directory:       {ARTIFACTS_DIR}")
    print("=" * 60)
