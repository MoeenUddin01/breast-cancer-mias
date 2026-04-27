"""Generate comparison charts and summary for DeiT-Base vs CNN models."""

import matplotlib.pyplot as plt
import numpy as np

# Model data
models = ["EfficientNetB2", "DeiT-Base", "ResNet152", "Xception"]
auc_scores = [0.9862, 0.9721, 0.9780, 0.8827]
accuracy_scores = [94.93, 95.01, 94.04, 78.18]
f1_scores = [0.9629, 0.9640, 0.9571, 0.8259]
train_times = [12.3, 108.4, 25.3, 12.0]

# Colors
colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("DeiT-Base vs CNN Models: Performance Comparison", fontsize=16, fontweight="bold")

# Plot 1: AUC-ROC
bars1 = axes[0, 0].bar(models, auc_scores, color=colors, alpha=0.8)
axes[0, 0].set_ylabel("AUC-ROC", fontsize=12, fontweight="bold")
axes[0, 0].set_title("AUC-ROC Score Comparison", fontsize=13)
axes[0, 0].set_ylim(0.85, 1.0)
axes[0, 0].grid(axis="y", alpha=0.3)
for bar, score in zip(bars1, auc_scores):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{score:.4f}", ha="center", fontweight="bold")
# Highlight DeiT
bars1[1].set_edgecolor("black")
bars1[1].set_linewidth(2)

# Plot 2: Accuracy
bars2 = axes[0, 1].bar(models, accuracy_scores, color=colors, alpha=0.8)
axes[0, 1].set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
axes[0, 1].set_title("Accuracy Comparison", fontsize=13)
axes[0, 1].set_ylim(75, 100)
axes[0, 1].grid(axis="y", alpha=0.3)
for bar, score in zip(bars2, accuracy_scores):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{score:.2f}%", ha="center", fontweight="bold")
# Highlight DeiT
bars2[1].set_edgecolor("black")
bars2[1].set_linewidth(2)

# Plot 3: F1 Score
bars3 = axes[1, 0].bar(models, f1_scores, color=colors, alpha=0.8)
axes[1, 0].set_ylabel("F1 Score", fontsize=12, fontweight="bold")
axes[1, 0].set_title("F1 Score Comparison", fontsize=13)
axes[1, 0].set_ylim(0.8, 1.0)
axes[1, 0].grid(axis="y", alpha=0.3)
for bar, score in zip(bars3, f1_scores):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{score:.4f}", ha="center", fontweight="bold")
# Highlight DeiT
bars3[1].set_edgecolor("black")
bars3[1].set_linewidth(2)

# Plot 4: Training Time
bars4 = axes[1, 1].bar(models, train_times, color=colors, alpha=0.8)
axes[1, 1].set_ylabel("Training Time (minutes)", fontsize=12, fontweight="bold")
axes[1, 1].set_title("Training Time Comparison", fontsize=13)
axes[1, 1].set_ylim(0, 120)
axes[1, 1].grid(axis="y", alpha=0.3)
for bar, time in zip(bars4, train_times):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    f"{time:.1f}m", ha="center", fontweight="bold")
# Highlight DeiT
bars4[1].set_edgecolor("black")
bars4[1].set_linewidth(2)

plt.tight_layout()

# Save the comparison chart
output_path = "artifacts/deit_vs_cnn_comparison/deit_vs_cnn_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"✓ Comparison chart saved to {output_path}")
plt.close()

# Create individual metric comparison charts
fig2, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(models))
width = 0.25

rects1 = ax.bar(x - width, auc_scores, width, label="AUC-ROC", color="#2ecc71", alpha=0.8)
rects2 = ax.bar(x, [a/100 for a in accuracy_scores], width, label="Accuracy", color="#3498db", alpha=0.8)
rects3 = ax.bar(x + width, f1_scores, width, label="F1 Score", color="#9b59b6", alpha=0.8)

ax.set_ylabel("Score", fontsize=12, fontweight="bold")
ax.set_title("All Metrics Comparison: DeiT-Base vs CNNs", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(loc="lower right")
ax.set_ylim(0.8, 1.0)
ax.grid(axis="y", alpha=0.3)

# Add value labels on bars
for rects in [rects1, rects2, rects3]:
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 0.005,
                f"{height:.3f}", ha="center", fontsize=9)

plt.tight_layout()
output_path2 = "artifacts/deit_vs_cnn_comparison/all_metrics_comparison.png"
plt.savefig(output_path2, dpi=300, bbox_inches="tight")
print(f"✓ All metrics chart saved to {output_path2}")
plt.close()

print("\n✅ Comparison charts generated successfully!")
