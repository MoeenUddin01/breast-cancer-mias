"""Generate research-quality comparison chart for publication."""

import matplotlib.pyplot as plt
import numpy as np

# Model data
models = ["EfficientNetB2", "DeiT-Base", "ResNet152", "Xception"]
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

# Create second chart: Line graph style
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Plot lines
ax2.plot(models, accuracy, marker="o", linewidth=2.5, markersize=10,
         color="#2E86AB", label="Accuracy (%)", markeredgecolor="black",
         markeredgewidth=1.5)
ax2.plot(models, [f * 100 for f in f1_score], marker="s", linewidth=2.5,
         markersize=10, color="#A23B72", label="F1 Score (%)",
         markeredgecolor="black", markeredgewidth=1.5)

# Customize axes
ax2.set_ylabel("Score (%)", fontsize=14, fontweight="bold", labelpad=10)
ax2.set_xlabel("Model", fontsize=14, fontweight="bold", labelpad=10)
ax2.set_ylim(75, 100)
ax2.grid(alpha=0.3, linestyle="--", linewidth=0.5)

# Add value labels
for i, (acc, f1) in enumerate(zip(accuracy, [f * 100 for f in f1_score])):
    ax2.text(i, acc + 0.5, f"{acc:.2f}%", ha="center", va="bottom",
             fontsize=11, fontweight="bold", color="#2E86AB")
    ax2.text(i, f1 - 1.5, f"{f1:.2f}%", ha="center", va="top",
             fontsize=11, fontweight="bold", color="#A23B72")

# Customize legend
ax2.legend(loc="upper right", fontsize=12, frameon=True,
           fancybox=True, shadow=True, framealpha=0.9)

# Title
ax2.set_title("Model Performance Comparison: Accuracy vs F1 Score (Line Plot)",
              fontsize=16, fontweight="bold", pad=15)

plt.tight_layout()

# Save line graph
output_path2 = "artifacts/research/model_comparison_accuracy_f1_line.png"
plt.savefig(output_path2, dpi=300, bbox_inches="tight", facecolor="white",
            edgecolor="none")
print(f"✓ Line graph saved to {output_path2}")
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
