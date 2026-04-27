"""Generate additional research-quality comparison charts."""

import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

# Model data
models = ["EfficientNetB2", "DeiT-Base", "ResNet152", "Xception"]
auc_scores = [0.9862, 0.9721, 0.9780, 0.8827]
accuracy = [94.93, 95.01, 94.04, 78.18]
f1_score = [0.9629, 0.9640, 0.9571, 0.8259]
train_times = [12.3, 108.4, 25.3, 12.0]

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid")

# 1. Radar/Spider Chart for multi-metric comparison
fig1, ax1 = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

# Normalize metrics to 0-1 scale for radar chart
categories = ["AUC-ROC", "Accuracy", "F1 Score"]
N = len(categories)

# Normalize values
auc_norm = [a for a in auc_scores]
acc_norm = [a / 100 for a in accuracy]
f1_norm = f1_score

colors = ["#2E86AB", "#A23B72", "#18A558", "#F18F01"]
markers = ["o", "s", "^", "D"]

for i, model in enumerate(models):
    values = [auc_norm[i], acc_norm[i], f1_norm[i]]
    values += values[:1]  # Close the radar
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    ax1.plot(angles, values, marker=markers[i], linewidth=2.5,
             color=colors[i], label=model, markersize=8, markeredgecolor="black",
             markeredgewidth=1.5)
    ax1.fill(angles, values, alpha=0.15, color=colors[i])

ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(categories, fontsize=12, fontweight="bold")
ax1.set_ylim(0.75, 1.0)
ax1.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])
ax1.set_yticklabels(["0.80", "0.85", "0.90", "0.95", "1.00"], fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax1.set_title("Multi-Metric Performance Comparison (Radar Chart)",
              fontsize=14, fontweight="bold", pad=20)

plt.tight_layout()
plt.savefig("artifacts/research/radar_chart_comparison.png", dpi=300,
            bbox_inches="tight", facecolor="white", edgecolor="none")
print("✓ Radar chart saved")
plt.close()

# 2. Scatter Plot: Accuracy vs F1 Score with AUC as point size
fig2, ax2 = plt.subplots(figsize=(10, 6))

scatter = ax2.scatter(accuracy, [f * 100 for f in f1_score],
                      s=[(a - 0.8) * 500 for a in auc_scores],
                      c=auc_scores, cmap="RdYlGn", alpha=0.7,
                      edgecolors="black", linewidths=1.5)

for i, model in enumerate(models):
    ax2.annotate(model, (accuracy[i], f1_score[i] * 100),
                xytext=(5, 5), textcoords="offset points",
                fontsize=11, fontweight="bold", ha="left")

ax2.set_xlabel("Accuracy (%)", fontsize=14, fontweight="bold", labelpad=10)
ax2.set_ylabel("F1 Score (%)", fontsize=14, fontweight="bold", labelpad=10)
ax2.set_xlim(75, 96)
ax2.set_ylim(80, 97)
ax2.grid(alpha=0.3, linestyle="--", linewidth=0.5)

cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label("AUC-ROC", fontsize=12, fontweight="bold")
cbar.set_ticks([0.88, 0.92, 0.96, 1.00])

ax2.set_title("Accuracy vs F1 Score (Point Size = AUC-ROC)",
              fontsize=16, fontweight="bold", pad=15)

plt.tight_layout()
plt.savefig("artifacts/research/scatter_accuracy_f1_auc.png", dpi=300,
            bbox_inches="tight", facecolor="white", edgecolor="none")
print("✓ Scatter plot saved")
plt.close()

# 3. Training Time vs Performance (Efficiency Analysis)
fig3, ax3 = plt.subplots(figsize=(10, 6))

# Plot scatter with performance as color
scatter2 = ax3.scatter(train_times, auc_scores,
                       s=[a * 200 for a in accuracy],
                       c=accuracy, cmap="viridis", alpha=0.7,
                       edgecolors="black", linewidths=1.5)

for i, model in enumerate(models):
    ax3.annotate(model, (train_times[i], auc_scores[i]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=11, fontweight="bold", ha="left")

ax3.set_xlabel("Training Time (minutes)", fontsize=14, fontweight="bold", labelpad=10)
ax3.set_ylabel("AUC-ROC", fontsize=14, fontweight="bold", labelpad=10)
ax3.set_xlim(0, 115)
ax3.set_ylim(0.87, 0.99)
ax3.grid(alpha=0.3, linestyle="--", linewidth=0.5)

cbar2 = plt.colorbar(scatter2, ax=ax3)
cbar2.set_label("Accuracy (%)", fontsize=12, fontweight="bold")

ax3.set_title("Efficiency Analysis: Training Time vs AUC-ROC (Point Size = Accuracy)",
              fontsize=16, fontweight="bold", pad=15)

# Add efficiency zone annotation
ax3.axvline(x=30, color="red", linestyle="--", linewidth=2, alpha=0.5)
ax3.text(35, 0.89, "Fast Training Zone\n(< 30 min)", fontsize=10,
         style="italic", bbox=dict(facecolor="yellow", alpha=0.3))

plt.tight_layout()
plt.savefig("artifacts/research/efficiency_analysis.png", dpi=300,
            bbox_inches="tight", facecolor="white", edgecolor="none")
print("✓ Efficiency analysis saved")
plt.close()

# 4. Grouped Bar Chart with Performance Categories
fig4, ax4 = plt.subplots(figsize=(10, 6))

x = np.arange(len(models))
width = 0.25

bars1 = ax4.bar(x - width, [a * 100 for a in auc_scores], width,
                label="AUC-ROC (%)", color="#2E86AB", alpha=0.85,
                edgecolor="black", linewidth=1)
bars2 = ax4.bar(x, accuracy, width, label="Accuracy (%)",
                color="#18A558", alpha=0.85, edgecolor="black", linewidth=1)
bars3 = ax4.bar(x + width, [f * 100 for f in f1_score], width,
                label="F1 Score (%)", color="#A23B72", alpha=0.85,
                edgecolor="black", linewidth=1)

ax4.set_ylabel("Score (%)", fontsize=14, fontweight="bold", labelpad=10)
ax4.set_xlabel("Model Architecture", fontsize=14, fontweight="bold", labelpad=10)
ax4.set_xticks(x)
ax4.set_xticklabels(models, fontsize=12, fontweight="bold")
ax4.set_ylim(75, 100)
ax4.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
ax4.set_axisbelow(True)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height + 0.3,
                 f"{height:.2f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")

ax4.legend(loc="upper right", fontsize=11, frameon=True,
           fancybox=True, shadow=True, framealpha=0.9)

# Add performance tier annotations
ax4.axhline(y=95, color="gold", linestyle="--", linewidth=2, alpha=0.6)
ax4.text(3.5, 95.5, "Excellent (>95%)", fontsize=10, style="italic",
         color="darkgoldenrod", fontweight="bold")
ax4.axhline(y=90, color="orange", linestyle="--", linewidth=2, alpha=0.6)
ax4.text(3.5, 90.5, "Good (90-95%)", fontsize=10, style="italic",
         color="darkorange", fontweight="bold")

ax4.set_title("Performance Tier Analysis: All Models Across Metrics",
              fontsize=16, fontweight="bold", pad=15)

plt.tight_layout()
plt.savefig("artifacts/research/performance_tier_analysis.png", dpi=300,
            bbox_inches="tight", facecolor="white", edgecolor="none")
print("✓ Performance tier analysis saved")
plt.close()

# 5. Horizontal Bar Chart for Training Time Comparison
fig5, ax5 = plt.subplots(figsize=(10, 6))

y_pos = np.arange(len(models))
bars = ax5.barh(y_pos, train_times, color="#F18F01", alpha=0.85,
                edgecolor="black", linewidth=1)

ax5.set_yticks(y_pos)
ax5.set_yticklabels(models, fontsize=12, fontweight="bold")
ax5.invert_yaxis()
ax5.set_xlabel("Training Time (minutes)", fontsize=14, fontweight="bold", labelpad=10)
ax5.set_xlim(0, 115)
ax5.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.5)
ax5.set_axisbelow(True)

for i, (bar, time) in enumerate(zip(bars, train_times)):
    ax5.text(time + 2, bar.get_y() + bar.get_height()/2, f"{time:.1f} min",
             va="center", fontsize=11, fontweight="bold")

# Highlight fastest models
ax5.axvline(x=20, color="green", linestyle="--", linewidth=2, alpha=0.5)
ax5.text(22, 3.5, "Fast (<20 min)", fontsize=10, style="italic",
         color="darkgreen", fontweight="bold")

ax5.set_title("Training Time Efficiency Comparison",
              fontsize=16, fontweight="bold", pad=15)

plt.tight_layout()
plt.savefig("artifacts/research/training_time_comparison.png", dpi=300,
            bbox_inches="tight", facecolor="white", edgecolor="none")
print("✓ Training time comparison saved")
plt.close()

print("\n✅ All additional research charts generated successfully!")
print("\nGenerated charts:")
print("  1. Radar chart - Multi-metric comparison")
print("  2. Scatter plot - Accuracy vs F1 with AUC size")
print("  3. Efficiency analysis - Time vs AUC")
print("  4. Performance tier analysis - Grouped bars with tiers")
print("  5. Training time comparison - Horizontal bars")
