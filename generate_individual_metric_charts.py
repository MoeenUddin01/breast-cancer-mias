"""Generate individual metric comparison charts (one by one)."""

import matplotlib.pyplot as plt
import numpy as np

# Model data
models = ["EfficientNetB2", "DeiT-Base", "ResNet152", "Xception"]
auc_scores = [0.9862, 0.9721, 0.9780, 0.8827]
accuracy = [94.93, 95.01, 94.04, 78.18]
f1_score = [0.9629, 0.9640, 0.9571, 0.8259]
train_times = [12.3, 108.4, 25.3, 12.0]

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid")

# 1. AUC-ROC Comparison
fig1, ax1 = plt.subplots(figsize=(10, 6))
colors = ["#2E86AB" if m == "EfficientNetB2" else "#A23B72" if m == "DeiT-Base"
          else "#18A558" if m == "ResNet152" else "#F18F01" for m in models]
bars1 = ax1.bar(models, auc_scores, color=colors, alpha=0.85,
                edgecolor="black", linewidth=1.5)

ax1.set_ylabel("AUC-ROC Score", fontsize=16, fontweight="bold", labelpad=10)
ax1.set_xlabel("Model", fontsize=16, fontweight="bold", labelpad=10)
ax1.set_ylim(0.85, 1.0)
ax1.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
ax1.set_axisbelow(True)

for bar, score in zip(bars1, auc_scores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f"{score:.4f}", ha="center", va="bottom", fontsize=13, fontweight="bold")

# Highlight best
best_idx = auc_scores.index(max(auc_scores))
bars1[best_idx].set_edgecolor("gold")
bars1[best_idx].set_linewidth(3)
ax1.text(best_idx, max(auc_scores) + 0.008, "← BEST",
         ha="center", fontsize=11, fontweight="bold", color="darkgoldenrod")

ax1.set_title("AUC-ROC Score Comparison", fontsize=18, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("artifacts/research/onebyone/01_auc_roc_comparison.png", dpi=300,
            bbox_inches="tight", facecolor="white", edgecolor="none")
print("✓ AUC-ROC chart saved")
plt.close()

# 2. Accuracy Comparison
fig2, ax2 = plt.subplots(figsize=(10, 6))
bars2 = ax2.bar(models, accuracy, color=colors, alpha=0.85,
                edgecolor="black", linewidth=1.5)

ax2.set_ylabel("Accuracy (%)", fontsize=16, fontweight="bold", labelpad=10)
ax2.set_xlabel("Model", fontsize=16, fontweight="bold", labelpad=10)
ax2.set_ylim(75, 100)
ax2.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
ax2.set_axisbelow(True)

for bar, score in zip(bars2, accuracy):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{score:.2f}%", ha="center", va="bottom", fontsize=13, fontweight="bold")

# Highlight best
best_idx = accuracy.index(max(accuracy))
bars2[best_idx].set_edgecolor("gold")
bars2[best_idx].set_linewidth(3)
ax2.text(best_idx, max(accuracy) + 1.5, "← BEST",
         ha="center", fontsize=11, fontweight="bold", color="darkgoldenrod")

ax2.set_title("Accuracy Comparison", fontsize=18, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("artifacts/research/onebyone/02_accuracy_comparison.png", dpi=300,
            bbox_inches="tight", facecolor="white", edgecolor="none")
print("✓ Accuracy chart saved")
plt.close()

# 3. F1 Score Comparison
fig3, ax3 = plt.subplots(figsize=(10, 6))
bars3 = ax3.bar(models, f1_score, color=colors, alpha=0.85,
                edgecolor="black", linewidth=1.5)

ax3.set_ylabel("F1 Score", fontsize=16, fontweight="bold", labelpad=10)
ax3.set_xlabel("Model", fontsize=16, fontweight="bold", labelpad=10)
ax3.set_ylim(0.8, 1.0)
ax3.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
ax3.set_axisbelow(True)

for bar, score in zip(bars3, f1_score):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{score:.4f}", ha="center", va="bottom", fontsize=13, fontweight="bold")

# Highlight best
best_idx = f1_score.index(max(f1_score))
bars3[best_idx].set_edgecolor("gold")
bars3[best_idx].set_linewidth(3)
ax3.text(best_idx, max(f1_score) + 0.008, "← BEST",
         ha="center", fontsize=11, fontweight="bold", color="darkgoldenrod")

ax3.set_title("F1 Score Comparison", fontsize=18, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("artifacts/research/onebyone/03_f1_score_comparison.png", dpi=300,
            bbox_inches="tight", facecolor="white", edgecolor="none")
print("✓ F1 Score chart saved")
plt.close()

# 4. Training Time Comparison
fig4, ax4 = plt.subplots(figsize=(10, 6))
bars4 = ax4.bar(models, train_times, color=colors, alpha=0.85,
                edgecolor="black", linewidth=1.5)

ax4.set_ylabel("Training Time (minutes)", fontsize=16, fontweight="bold", labelpad=10)
ax4.set_xlabel("Model", fontsize=16, fontweight="bold", labelpad=10)
ax4.set_ylim(0, 115)
ax4.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)
ax4.set_axisbelow(True)

for bar, time in zip(bars4, train_times):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
             f"{time:.1f} min", ha="center", va="bottom", fontsize=13, fontweight="bold")

# Highlight best (fastest)
best_idx = train_times.index(min(train_times))
bars4[best_idx].set_edgecolor("gold")
bars4[best_idx].set_linewidth(3)
ax4.text(best_idx, min(train_times) + 8, "← FASTEST",
         ha="center", fontsize=11, fontweight="bold", color="darkgoldenrod")

ax4.set_title("Training Time Comparison", fontsize=18, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("artifacts/research/onebyone/04_training_time_comparison.png", dpi=300,
            bbox_inches="tight", facecolor="white", edgecolor="none")
print("✓ Training Time chart saved")
plt.close()

# 5. AUC-ROC Line Chart
fig5, ax5 = plt.subplots(figsize=(10, 6))
ax5.plot(models, auc_scores, marker="o", linewidth=3, markersize=12,
         color="#2E86AB", markeredgecolor="black", markeredgewidth=2)
ax5.set_ylabel("AUC-ROC Score", fontsize=16, fontweight="bold", labelpad=10)
ax5.set_xlabel("Model", fontsize=16, fontweight="bold", labelpad=10)
ax5.set_ylim(0.85, 1.0)
ax5.grid(alpha=0.3, linestyle="--", linewidth=0.5)

for i, score in enumerate(auc_scores):
    ax5.text(i, score + 0.005, f"{score:.4f}", ha="center",
             fontsize=13, fontweight="bold", color="#2E86AB")

ax5.set_title("AUC-ROC Score Trend", fontsize=18, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("artifacts/research/onebyone/05_auc_roc_trend.png", dpi=300,
            bbox_inches="tight", facecolor="white", edgecolor="none")
print("✓ AUC-ROC trend chart saved")
plt.close()

# 6. Accuracy Line Chart
fig6, ax6 = plt.subplots(figsize=(10, 6))
ax6.plot(models, accuracy, marker="s", linewidth=3, markersize=12,
         color="#18A558", markeredgecolor="black", markeredgewidth=2)
ax6.set_ylabel("Accuracy (%)", fontsize=16, fontweight="bold", labelpad=10)
ax6.set_xlabel("Model", fontsize=16, fontweight="bold", labelpad=10)
ax6.set_ylim(75, 100)
ax6.grid(alpha=0.3, linestyle="--", linewidth=0.5)

for i, score in enumerate(accuracy):
    ax6.text(i, score + 1, f"{score:.2f}%", ha="center",
             fontsize=13, fontweight="bold", color="#18A558")

ax6.set_title("Accuracy Trend", fontsize=18, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("artifacts/research/onebyone/06_accuracy_trend.png", dpi=300,
            bbox_inches="tight", facecolor="white", edgecolor="none")
print("✓ Accuracy trend chart saved")
plt.close()

# 7. F1 Score Line Chart
fig7, ax7 = plt.subplots(figsize=(10, 6))
ax7.plot(models, f1_score, marker="^", linewidth=3, markersize=12,
         color="#A23B72", markeredgecolor="black", markeredgewidth=2)
ax7.set_ylabel("F1 Score", fontsize=16, fontweight="bold", labelpad=10)
ax7.set_xlabel("Model", fontsize=16, fontweight="bold", labelpad=10)
ax7.set_ylim(0.8, 1.0)
ax7.grid(alpha=0.3, linestyle="--", linewidth=0.5)

for i, score in enumerate(f1_score):
    ax7.text(i, score + 0.005, f"{score:.4f}", ha="center",
             fontsize=13, fontweight="bold", color="#A23B72")

ax7.set_title("F1 Score Trend", fontsize=18, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("artifacts/research/onebyone/07_f1_score_trend.png", dpi=300,
            bbox_inches="tight", facecolor="white", edgecolor="none")
print("✓ F1 Score trend chart saved")
plt.close()

# 8. Training Time Line Chart
fig8, ax8 = plt.subplots(figsize=(10, 6))
ax8.plot(models, train_times, marker="D", linewidth=3, markersize=12,
         color="#F18F01", markeredgecolor="black", markeredgewidth=2)
ax8.set_ylabel("Training Time (minutes)", fontsize=16, fontweight="bold", labelpad=10)
ax8.set_xlabel("Model", fontsize=16, fontweight="bold", labelpad=10)
ax8.set_ylim(0, 115)
ax8.grid(alpha=0.3, linestyle="--", linewidth=0.5)

for i, time in enumerate(train_times):
    ax8.text(i, time + 5, f"{time:.1f} min", ha="center",
             fontsize=13, fontweight="bold", color="#F18F01")

ax8.set_title("Training Time Trend", fontsize=18, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("artifacts/research/onebyone/08_training_time_trend.png", dpi=300,
            bbox_inches="tight", facecolor="white", edgecolor="none")
print("✓ Training Time trend chart saved")
plt.close()

print("\n✅ All individual metric charts generated successfully!")
print("\nGenerated charts in artifacts/research/onebyone/:")
print("  1. 01_auc_roc_comparison.png - Bar chart")
print("  2. 02_accuracy_comparison.png - Bar chart")
print("  3. 03_f1_score_comparison.png - Bar chart")
print("  4. 04_training_time_comparison.png - Bar chart")
print("  5. 05_auc_roc_trend.png - Line chart")
print("  6. 06_accuracy_trend.png - Line chart")
print("  7. 07_f1_score_trend.png - Line chart")
print("  8. 08_training_time_trend.png - Line chart")
