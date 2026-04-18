"""Generate a clean grouped bar chart replacement for the OXS radar figure.

The radar chart had heavy line overlap; this version uses bars for each
criterion grouped by model, making it much easier to read off values.
"""
import matplotlib.pyplot as plt
import numpy as np

# Values from Table tab:xai-behav (behaviour-only regime)
# Source: dissertation Table 4.5
models = ["XGBoost", "LightGBM", "LogReg", "Random Forest"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# Per-criterion normalised scores (higher = better, all in [0, 1])
# Source: dissertation Table 4.5 (behaviour-only, deduplicated, 3-seed mean)
# Faithfulness  = mean of clipped comprehensiveness and sufficiency
# Stability     = S2 noise-perturbation Spearman (LogReg = 0 since deterministic)
# Simplicity    = Gini coefficient
# Plausibility  = per-attack mean RMA@5 (LogReg uses binary aggregate)
# Timeliness    = 1 / (1 + ms_per_sample / 1000)
data = {
    "Faithfulness": [0.500, 0.572, 0.551, 0.533],   # (max(0, comp) + suff) / 2
    "Stability":    [0.997, 0.960, 0.997, 0.000],
    "Simplicity":   [0.885, 0.909, 0.879, 0.884],
    "Plausibility": [0.711, 0.519, 0.496, 0.633],
    "Timeliness":   [0.999, 0.991, 0.990, 1.000],
}
oxs_scores = [0.718, 0.705, 0.693, 0.516]

criteria = list(data.keys())
n_criteria = len(criteria)
n_models = len(models)

# Build the bar positions
x = np.arange(n_criteria)
width = 0.18
offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

fig, ax = plt.subplots(figsize=(9.5, 5.2))

for i, model in enumerate(models):
    vals = [data[c][i] for c in criteria]
    label = f"{model} (OXS = {oxs_scores[i]:.3f})"
    bars = ax.bar(x + offsets[i], vals, width,
                  label=label, color=colors[i],
                  edgecolor="white", linewidth=0.6)

ax.set_xticks(x)
ax.set_xticklabels(criteria, fontsize=11)
ax.set_ylabel("Normalised score (0--1, higher is better)", fontsize=11)
ax.set_ylim(0, 1.10)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_title("OXS five-criterion comparison (behaviour-only regime)",
             fontsize=12, pad=12)
ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend at the bottom, single row
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
          ncol=4, frameon=False, fontsize=9.5)

plt.tight_layout()
plt.savefig("figures/oxs_radar.pdf", bbox_inches="tight")
print("Saved figures/oxs_radar.pdf")
