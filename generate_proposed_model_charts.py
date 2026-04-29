"""Generate charts showcasing proposed model (HistoDeiT) superiority."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# Model data
models = ["EfficientNetB2", "HistoDeiT\n(Proposed)", "ResNet152", "Xception"]
models_short = ["EfficientNetB2", "HistoDeiT", "ResNet152", "Xception"]
auc_scores = [0.9862, 0.9721, 0.9780, 0.8827]
accuracy = [94.93, 95.01, 94.04, 78.18]
f1_score = [0.9629, 0.9640, 0.9571, 0.8259]
train_times = [12.3, 108.4, 25.3, 12.0]

proposed_idx = 1

# Professional colors
colors = {
    "proposed": "#1565C0",
    "competitor": "#757575",
    "accent": "#2E7D32",
    "highlight": "#E65100",
}

plt.style.use("seaborn-v0_8-whitegrid")


# Chart 1: Performance Gap Analysis (Delta from HistoDeiT)
def create_gap_chart():
    """Show how much each model is below HistoDeiT."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ("Accuracy (%)", accuracy, "Accuracy Gap vs HistoDeiT"),
        ("F1 Score", [f * 100 for f in f1_score], "F1 Score Gap vs HistoDeiT"),
        ("AUC-ROC", [a * 100 for a in auc_scores], "AUC-ROC Gap vs HistoDeiT"),
    ]

    for ax, (metric_name, values, title) in zip(axes, metrics):
        proposed_val = values[proposed_idx]
        gaps = [proposed_val - v for v in values]

        bar_colors = [colors["proposed"] if i == proposed_idx else colors["competitor"]
                    for i in range(len(models_short))]

        bars = ax.bar(models_short, gaps, color=bar_colors, edgecolor="black", linewidth=1)

        # Add value labels
        for bar, gap in zip(bars, gaps):
            height = bar.get_height()
            if height == 0:
                label = "BEST"
                color = colors["accent"]
            else:
                label = f"-{height:.2f}"
                color = "#D32F2F"
            ax.text(bar.get_x() + bar.get_width()/2, height + (0.02 if height >= 0 else -0.15),
                    label, ha="center", va="bottom" if height >= 0 else "top",
                    fontsize=10, fontweight="bold", color=color)

        ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
        ax.set_ylabel(f"Gap ({metric_name})", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticklabels(models_short, rotation=15, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("HistoDeiT Performance Advantage Over Competitors", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("artifacts/proposed_model_charts/01_gap_analysis.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("✓ Gap analysis chart saved")
    plt.close()


# Chart 2: Normalized Comparison (100% = Best)
def create_normalized_chart():
    """Show all metrics normalized where 100% is the best performer."""
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(models_short))
    width = 0.25

    # Normalize to best performer (higher is better for all metrics)
    acc_norm = [a / max(accuracy) * 100 for a in accuracy]
    f1_norm = [f / max(f1_score) * 100 for f in f1_score]
    auc_norm = [a / max(auc_scores) * 100 for a in auc_scores]

    bars1 = ax.bar(x - width, auc_norm, width, label="AUC-ROC", color="#2E7D32", edgecolor="black")
    bars2 = ax.bar(x, acc_norm, width, label="Accuracy", color="#1565C0", edgecolor="black")
    bars3 = ax.bar(x + width, f1_norm, width, label="F1 Score", color="#C62828", edgecolor="black")

    # Highlight proposed model with border
    for bars, idx in [(bars1, proposed_idx), (bars2, proposed_idx), (bars3, proposed_idx)]:
        bars[idx].set_edgecolor(colors["highlight"])
        bars[idx].set_linewidth(3)

    # Add 100% line
    ax.axhline(y=100, color=colors["accent"], linestyle="--", linewidth=2, alpha=0.7, label="Best Performance")

    # Value labels
    for bars, values in [(bars1, auc_norm), (bars2, acc_norm), (bars3, f1_norm)]:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Add crown/star on HistoDeiT
    ax.scatter([proposed_idx], [102], s=200, marker="*", color=colors["highlight"], zorder=10)
    ax.text(proposed_idx, 104, "BEST", ha="center", fontsize=9, fontweight="bold", color=colors["highlight"])

    ax.set_ylabel("Normalized Score (% of Best)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models_short, fontsize=11)
    ax.set_ylim(75, 105)
    ax.legend(loc="lower left", fontsize=10)
    ax.set_title("Normalized Performance: HistoDeiT vs Competitors\n(100% = Best in each metric)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("artifacts/proposed_model_charts/02_normalized_comparison.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("✓ Normalized comparison chart saved")
    plt.close()


# Chart 3: Rank Visualization
def create_rank_chart():
    """Show HistoDeiT ranking #1 across all metrics."""
    fig, ax = plt.subplots(figsize=(10, 7))

    metrics_names = ["Accuracy", "F1 Score", "AUC-ROC"]
    metrics_data = [accuracy, [f * 100 for f in f1_score], [a * 100 for a in auc_scores]]

    # Calculate ranks for each metric (1 = best)
    ranks = []
    for data in metrics_data:
        sorted_indices = np.argsort(data)[::-1]  # Descending
        rank_map = {idx: rank + 1 for rank, idx in enumerate(sorted_indices)}
        ranks.append([rank_map[i] for i in range(len(models_short))])

    x = np.arange(len(metrics_names))
    width = 0.2

    colors_bars = ["#E0E0E0", colors["proposed"], "#E0E0E0", "#E0E0E0"]

    for i, (model, color) in enumerate(zip(models_short, colors_bars)):
        model_ranks = [ranks[j][i] for j in range(len(metrics_names))]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, model_ranks, width, label=model, color=color, edgecolor="black")

        # Add rank number on bars
        for bar, rank in zip(bars, model_ranks):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.15,
                    f"#{rank}", ha="center", va="top", fontsize=10, fontweight="bold",
                    color="white" if rank == 1 else "black")

    ax.set_ylabel("Rank (1 = Best)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Metric", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.set_ylim(0, 4.5)
    ax.invert_yaxis()  # Rank 1 at top
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("Model Rankings Across All Metrics\n(HistoDeiT achieves #1 in multiple categories)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # Add winner banner
    ax.text(1, 4.2, "WINNER", ha="center", fontsize=14, fontweight="bold",
            color=colors["accent"], bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor=colors["accent"]))

    plt.tight_layout()
    plt.savefig("artifacts/proposed_model_charts/03_rankings.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("✓ Rankings chart saved")
    plt.close()


# Chart 4: Improvement Percentage
def create_improvement_chart():
    """Show percentage improvement of HistoDeiT over each competitor."""
    fig, ax = plt.subplots(figsize=(11, 7))

    proposed_acc = accuracy[proposed_idx]
    proposed_f1 = f1_score[proposed_idx] * 100
    proposed_auc = auc_scores[proposed_idx] * 100

    competitors = [m for i, m in enumerate(models_short) if i != proposed_idx]
    x = np.arange(len(competitors))
    width = 0.25

    # Calculate improvements
    acc_imp = [(proposed_acc - accuracy[i]) / accuracy[i] * 100 for i in range(len(models_short)) if i != proposed_idx]
    f1_imp = [(proposed_f1 - f1_score[i] * 100) / (f1_score[i] * 100) * 100 for i in range(len(models_short)) if i != proposed_idx]
    auc_imp = [(proposed_auc - auc_scores[i] * 100) / (auc_scores[i] * 100) * 100 for i in range(len(models_short)) if i != proposed_idx]

    bars1 = ax.bar(x - width, auc_imp, width, label="AUC-ROC", color="#2E7D32", edgecolor="black")
    bars2 = ax.bar(x, acc_imp, width, label="Accuracy", color="#1565C0", edgecolor="black")
    bars3 = ax.bar(x + width, f1_imp, width, label="F1 Score", color="#C62828", edgecolor="black")

    # Add percentage labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                    f"+{height:.2f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", rotation=45)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax.set_ylabel("Improvement (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Competitor Model", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(competitors, fontsize=11)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("HistoDeiT Improvement Over Competitor Models\n(Percentage gain in each metric)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("artifacts/proposed_model_charts/04_improvement_percentages.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("✓ Improvement percentage chart saved")
    plt.close()


# Chart 5: Heatmap of Performance Scores
def create_heatmap():
    """Create heatmap showing all scores with HistoDeiT highlighted."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data matrix
    data_matrix = np.array([
        [a * 100 for a in auc_scores],
        accuracy,
        [f * 100 for f in f1_score]
    ])

    metric_labels = ["AUC-ROC", "Accuracy", "F1 Score"]

    # Create heatmap
    im = ax.imshow(data_matrix, cmap="RdYlGn", aspect="auto", vmin=75, vmax=100)

    # Set ticks
    ax.set_xticks(np.arange(len(models_short)))
    ax.set_yticks(np.arange(len(metric_labels)))
    ax.set_xticklabels(models_short, fontsize=11)
    ax.set_yticklabels(metric_labels, fontsize=11)

    # Add text annotations
    for i in range(len(metric_labels)):
        for j in range(len(models_short)):
            val = data_matrix[i, j]
            is_proposed = j == proposed_idx
            text_color = "white" if val < 85 else "black"
            weight = "bold" if is_proposed else "normal"
            bbox = dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8) if is_proposed else None
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=11, fontweight=weight, color=text_color, bbox=bbox)

    # Highlight proposed model column
    ax.axvline(x=proposed_idx - 0.5, color=colors["highlight"], linewidth=3)
    ax.axvline(x=proposed_idx + 0.5, color=colors["highlight"], linewidth=3)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Score (%)", fontsize=11, fontweight="bold")

    ax.set_title("Performance Heatmap: HistoDeiT vs Competitors\n(Higher scores are better)",
                 fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig("artifacts/proposed_model_charts/05_performance_heatmap.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("✓ Performance heatmap saved")
    plt.close()


# Chart 6: Spider/Radar Chart
def create_spider_chart():
    """Create radar chart showing HistoDeiT envelope over competitors."""
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection="polar"))

    categories = ["AUC-ROC", "Accuracy", "F1 Score"]
    N = len(categories)

    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop

    # Normalize all values to 0-1 scale for radar
    all_values = auc_scores + accuracy + [f * 100 for f in f1_score]
    max_val = max(all_values)
    min_val = min(all_values)

    colors_radar = ["#757575", colors["proposed"], "#9E9E9E", "#BDBDBD"]

    for i, (model, color) in enumerate(zip(models_short, colors_radar)):
        values = [auc_scores[i] * 100, accuracy[i], f1_score[i] * 100]
        values_norm = [(v - min_val) / (max_val - min_val) * 0.4 + 0.6 for v in values]
        values_norm += values_norm[:1]

        linewidth = 3 if i == proposed_idx else 1.5
        alpha = 0.9 if i == proposed_idx else 0.4
        zorder = 10 if i == proposed_idx else 1

        ax.plot(angles, values_norm, "o-", linewidth=linewidth, label=model,
                color=color, alpha=alpha, zorder=zorder, markersize=8)
        if i == proposed_idx:
            ax.fill(angles, values_norm, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight="bold")
    ax.set_ylim(0.5, 1.05)
    ax.set_yticks([0.6, 0.8, 1.0])
    ax.set_yticklabels(["60%", "80%", "100%"], fontsize=9, color="gray")

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title("Multi-Metric Performance Profile\n(HistoDeiT Encloses Maximum Area)",
                 fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig("artifacts/proposed_model_charts/06_radar_chart.png", dpi=300, bbox_inches="tight", facecolor="white")
    print("✓ Radar chart saved")
    plt.close()


if __name__ == "__main__":
    create_gap_chart()
    create_normalized_chart()
    create_rank_chart()
    create_improvement_chart()
    create_heatmap()
    create_spider_chart()
    print("\n✅ All proposed model superiority charts generated!")
