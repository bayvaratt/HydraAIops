import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

models = ["LogReg", "LightGBM", "CNN-LSTM", "XGBoost", "Random\nForest"]
source_roc = [0.817, 0.999, 0.985, 0.999, 0.997]
target_roc = [0.744, 0.441, 0.285, 0.136, 0.129]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, source_roc, width, label="Source ROC-AUC", color="#4472C4")
bars2 = ax.bar(x + width/2, target_roc, width, label="Target ROC-AUC", color="#C0504D")

ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, label="Random baseline (0.5)")

ax.set_ylabel("ROC-AUC", fontsize=12)
ax.set_title("Cross-Dataset Generalisation: TON_IoT to CIC-IoT-2023", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0, 1.15)
ax.legend(loc="upper left", fontsize=10)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("figures/cross_dataset_roc.pdf", format="pdf", dpi=300)
print("Saved figures/cross_dataset_roc.pdf")
