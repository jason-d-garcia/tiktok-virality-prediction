import os
import json
import numpy as np
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

# -----------------------
# CONFIG
# -----------------------

MODEL_DIR = "../models/"
FEATURE_DIR = "saved_features"
OUTPUT_DIR = "shap_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Choose which model to analyze
MODEL_FILE = "xgboost_full_pca_mode=block_standard.json"
FEATURE_FILE = "X_pca_block.npy"   # must match model PCA mode

# -----------------------
# LOAD MODEL
# -----------------------

model = xgb.XGBRegressor()
model.load_model(os.path.join(MODEL_DIR, MODEL_FILE))
print(f"✅ Loaded model: {MODEL_FILE}")

# -----------------------
# LOAD FEATURES
# -----------------------

X = np.load(os.path.join(FEATURE_DIR, FEATURE_FILE))
print(f"✅ Loaded features: {X.shape}")

# -----------------------
# SHAP EXPLAINER
# -----------------------

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print("✅ SHAP values computed")

# -----------------------
# GLOBAL FEATURE IMPORTANCE
# -----------------------

plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_global.png"), dpi=300)
plt.close()

print("✅ Saved global SHAP summary")

# -----------------------
# MEAN ABS SHAP PER FEATURE
# -----------------------

mean_abs_shap = np.abs(shap_values).mean(axis=0)

np.save(
    os.path.join(OUTPUT_DIR, "mean_abs_shap.npy"),
    mean_abs_shap
)

print("✅ Saved mean absolute SHAP values")

# -----------------------
# BLOCK-LEVEL IMPORTANCE (OPTIONAL)
# -----------------------

# Only use this if PCA_MODE == "block"
BLOCK_SLICES = {
    "numeric": slice(0, 32),
    "text_desc": slice(32, 64),
    "text_tags": slice(64, 96),
    "clip": slice(96, 128),
    "whisper": slice(128, 160)
}

block_importance = {}

for block, sl in BLOCK_SLICES.items():
    block_importance[block] = float(np.abs(shap_values[:, sl]).mean())

with open(os.path.join(OUTPUT_DIR, "block_importance.json"), "w") as f:
    json.dump(block_importance, f, indent=2)

print("✅ Saved block-level SHAP importance")
print(block_importance)