"""Generate research-quality comparison chart for publication."""

import matplotlib.pyplot as plt
import numpy as np

# Model data
models = ["EfficientNetB2", "HistoDeiT", "ResNet152", "Xception"]
accuracy = [94.93, 95.01, 94.04, 78.18]
f1_score = [0.9629, 0.9640, 0.9571, 0.8259]

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))

# Set up the plot
x = np.arange(len(models))
width = 0.35

# Create bars with professional colors
bars1 = ax.bar(x - width/2, accuracy, width, label="Accuracy (%)",
               color="#2E86AB", alpha=0.85, edgecolor="black", linewidth=0.8)
bars2 = ax.bar(x + width/2, [f * 100 for f in f1_score], width, label="F1 Score (%)",
               color="#A23B72", alpha=0.85, edgecolor="black", linewidth=0.8)

# Customize axes
ax.set_ylabel("Score (%)", fontsize=14, fontweight="bold", labelpad=10)
ax.set_xlabel("Model", fontsize=14, fontweight="bold", labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12, fontweight="bold")
ax.set_ylim(75, 100)

# Add grid
ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
ax.set_axisbelow(True)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.3,
                f"{height:.2f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

# Customize legend
ax.legend(loc="upper right", fontsize=12, frameon=True,
          fancybox=True, shadow=True, framealpha=0.9)

# Title
ax.set_title("Model Performance Comparison: Accuracy vs F1 Score",
             fontsize=16, fontweight="bold", pad=15)

# Tight layout
plt.tight_layout()

# Save high-resolution figure
output_path = "artifacts/research/model_comparison_accuracy_f1.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white",
            edgecolor="none")
print(f"✓ Research chart saved to {output_path}")
plt.close()

# Create second chart: Professional line graph with ALL metrics - NO OVERLAPS
fig2, ax2 = plt.subplots(figsize=(13, 8))

auc_scores = [0.9862, 0.9721, 0.9780, 0.8827]
proposed_idx = 1  # HistoDeiT index

# Professional color palette
colors = {
    "auc": "#2E7D32",
    "accuracy": "#1565C0",
    "f1": "#C62828",
    "highlight": "#E65100",
    "text": "#212121"
}

# Plot all three metrics
ax2.plot(models, [a * 100 for a in auc_scores], marker="s", linewidth=2.5,
         markersize=10, color=colors["auc"], label="AUC-ROC (%)",
         markeredgecolor="white", markeredgewidth=2, zorder=5)

ax2.plot(models, accuracy, marker="o", linewidth=3,
         markersize=12, color=colors["accuracy"], label="Accuracy (%)",
         markeredgecolor="white", markeredgewidth=2, zorder=6)

ax2.plot(models, [f * 100 for f in f1_score], marker="^", linewidth=2.5,
         markersize=10, color=colors["f1"], label="F1 Score (%)",
         markeredgecolor="white", markeredgewidth=2, zorder=5)

# Highlight proposed model
ax2.scatter([proposed_idx], [accuracy[proposed_idx]], s=500,
            facecolors="none", edgecolors=colors["highlight"],
            linewidths=3, zorder=10)

# Axis styling
ax2.set_ylabel("Score (%)", fontsize=14, fontweight="bold",
               color=colors["text"], labelpad=12)
ax2.set_xlabel("Model Architecture", fontsize=14, fontweight="bold",
               color=colors["text"], labelpad=12)
ax2.set_ylim(85, 101)
ax2.set_xticklabels(models, fontsize=12, fontweight="medium")

# Grid
ax2.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.8, color="gray")
ax2.set_axisbelow(True)

# Clean spines
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["left"].set_color("gray")
ax2.spines["bottom"].set_color("gray")

# Value labels with NO OVERLAPS - staggered positions
for i, (auc, acc, f1) in enumerate(zip(auc_scores, accuracy, f1_score)):
    # AUC label - top position
    ax2.annotate(f"{auc:.3f}", (i, auc * 100 + 0.8),
                 fontsize=10, ha="center", va="bottom",
                 color=colors["auc"], fontweight="bold")

    # Accuracy label - middle position, highlighted for proposed
    acc_y_offset = 0.5 if i != proposed_idx else -1.2
    weight = "bold" if i == proposed_idx else "medium"
    ax2.annotate(f"{acc:.2f}", (i, acc + acc_y_offset),
                 fontsize=11, ha="center",
                 va="top" if i == proposed_idx else "bottom",
                 color=colors["accuracy"], fontweight=weight)

    # F1 label - bottom position
    ax2.annotate(f"{f1:.3f}", (i, f1 * 100 - 0.8),
                 fontsize=10, ha="center", va="top",
                 color=colors["f1"], fontweight="bold")

# "Best Accuracy" badge - positioned to avoid overlap
ax2.text(proposed_idx, accuracy[proposed_idx] + 2.5, "Best Accuracy",
         fontsize=10, fontweight="bold", color="white", ha="center",
         bbox=dict(boxstyle="round,pad=0.3", facecolor=colors["accuracy"],
                   edgecolor="white", linewidth=2))

# Proposed Model callout - positioned in empty space (left side)
ax2.annotate("Proposed Model\nHistoDeiT", xy=(proposed_idx, accuracy[proposed_idx]),
             xytext=(0.3, 99),
             fontsize=11, fontweight="bold", color=colors["highlight"],
             ha="center", va="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor=colors["highlight"], linewidth=2),
             arrowprops=dict(arrowstyle="-|>", color=colors["highlight"],
                            lw=2, mutation_scale=18,
                            connectionstyle="arc3,rad=0.2"))

# Legend - top right to avoid data
ax2.legend(loc="lower right", fontsize=11, frameon=True,
           fancybox=False, edgecolor="gray", framealpha=0.95)

# Title and subtitle
ax2.set_title("Performance Comparison Across Model Architectures",
              fontsize=16, fontweight="bold", color=colors["text"], pad=20)

fig2.text(0.5, 0.94, "HistoDeiT (Proposed) achieves highest accuracy (95.01%)",
          ha="center", fontsize=12, style="italic", color="gray")

plt.tight_layout()
plt.subplots_adjust(top=0.90)

# Save
output_path2 = "artifacts/research/model_comparison_accuracy_f1_line.png"
plt.savefig(output_path2, dpi=300, bbox_inches="tight", facecolor="white",
            edgecolor="none")
print(f"✓ Clean line graph saved to {output_path2}")
plt.close()

# Create third chart: All metrics including AUC
fig3, ax3 = plt.subplots(figsize=(10, 6))

auc_scores = [0.9862, 0.9721, 0.9780, 0.8827]
x = np.arange(len(models))
width = 0.25

bars1 = ax3.bar(x - width, [a * 100 for a in auc_scores], width,
                label="AUC-ROC (%)", color="#18A558", alpha=0.85,
                edgecolor="black", linewidth=0.8)
bars2 = ax3.bar(x, accuracy, width, label="Accuracy (%)",
                color="#2E86AB", alpha=0.85, edgecolor="black", linewidth=0.8)
bars3 = ax3.bar(x + width, [f * 100 for f in f1_score], width,
                label="F1 Score (%)", color="#A23B72", alpha=0.85,
                edgecolor="black", linewidth=0.8)

ax3.set_ylabel("Score (%)", fontsize=14, fontweight="bold", labelpad=10)
ax3.set_xlabel("Model", fontsize=14, fontweight="bold", labelpad=10)
ax3.set_xticks(x)
ax3.set_xticklabels(models, fontsize=12, fontweight="bold")
ax3.set_ylim(75, 100)
ax3.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
ax3.set_axisbelow(True)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + 0.3,
                 f"{height:.2f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

ax3.legend(loc="upper right", fontsize=11, frameon=True,
           fancybox=True, shadow=True, framealpha=0.9)

ax3.set_title("Comprehensive Model Performance Comparison: All Metrics",
              fontsize=16, fontweight="bold", pad=15)

plt.tight_layout()

output_path3 = "artifacts/research/model_comparison_all_metrics.png"
plt.savefig(output_path3, dpi=300, bbox_inches="tight", facecolor="white",
            edgecolor="none")
print(f"✓ All metrics chart saved to {output_path3}")
plt.close()

print("\n✅ Research-quality charts generated successfully!")
