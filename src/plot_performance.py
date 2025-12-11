import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ================================
# ===== CONFIG ===================
# ================================

RESULTS_PATH = "../reports/experiment_results.csv"
OUTPUT_DIR = "poster_figures_block_pca_expanding"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================
# ===== LOAD & FILTER DATA =======
# ================================

df = pd.read_csv(RESULTS_PATH)

# Keep ONLY:
#  - Block PCA
#  - Expanding CV
#  - Full models (not metadata)
df_block = df[
    (df["label"] == "FULL | PCA_MODE=block") &
    (df["cv_mode"] == "expanding")
].copy()

# Normalize model names
df_block["model"] = df_block["model"].str.capitalize()

print("\nFiltered Data Used for Plots:")
print(df_block[["model", "mean_r2", "mean_rmse", "mean_mae", "mean_smape", "mean_spearman", "mean_topk"]])

models = df_block["model"].values

# ================================
# ===== FIGURE 1: R² + RMSE ======
# ================================

plt.figure()

x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, df_block["mean_r2"], width, label="R²")
plt.bar(x + width/2, df_block["mean_rmse"], width, label="RMSE")

plt.axhline(0, linestyle="--", linewidth=1)

plt.xticks(x, models)
plt.ylabel("Score")
plt.title("Model Accuracy (Block PCA + Expanding CV)")
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figure_1_accuracy.png", dpi=300)
plt.close()

# ================================
# ===== FIGURE 2: MAE + sMAPE ====
# ================================

plt.figure()

plt.bar(x - width/2, df_block["mean_mae"], width, label="MAE")
plt.bar(x + width/2, df_block["mean_smape"], width, label="sMAPE")

plt.xticks(x, models)
plt.ylabel("Error")
plt.title("Numerical Prediction Error (Block PCA + Expanding CV)")
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figure_2_error.png", dpi=300)
plt.close()

# ================================
# ===== FIGURE 3: DISCOVERY ======
# ================================

plt.figure()

plt.bar(x - width/2, df_block["mean_spearman"], width, label="Spearman Rank")
plt.bar(x + width/2, df_block["mean_topk"], width, label="Top-10% Hit Rate")

plt.xticks(x, models)
plt.ylabel("Score (0–1)")
plt.title("Ranking & Discovery Power (Block PCA + Expanding CV)")
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/figure_3_discovery.png", dpi=300)
plt.close()

print("\n🎉 FINAL POSTER FIGURES GENERATED 🎉")
print(f"Saved to: {OUTPUT_DIR}/")
