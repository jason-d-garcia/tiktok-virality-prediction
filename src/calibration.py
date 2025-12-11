import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from scipy.special import expit

# Try XGBoost import
try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False


# ===============================
# ===== CONFIG ==================
# ===============================

MODEL_DIR = "../models"
FEATURE_DIR = "../data/saved_features"
OUTPUT_DIR = "../reports/calibration_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

N_BINS = 10
N_SPLITS = 3


# ===============================
# ===== LOAD BLOCK PCA DATA =====
# ===============================

X_BLOCK = np.load(f"{FEATURE_DIR}/X_pca_block.npy")
y       = np.load(f"{FEATURE_DIR}/y.npy")

print("Loaded BLOCK PCA feature matrix + y")


# ===============================
# ===== EXPANDING CV SPLIT ======
# ===============================

def expanding_splits(n_samples, n_splits):
    fold_size = n_samples // (n_splits + 1)
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_end  = fold_size * (i + 2)
        yield np.arange(0, train_end), np.arange(train_end, min(test_end, n_samples))


# ===============================
# ===== MODEL LOADING ===========
# ===============================

def load_model(path):
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)

    elif path.endswith(".json"):
        if not HAS_XGB:
            raise RuntimeError("XGBoost not installed.")
        model = xgb.XGBRegressor()
        model.load_model(path)
        return model

    else:
        raise ValueError(f"Unknown model format: {path}")


# ===== Load ONLY BLOCK models =====

MODELS = {}

for fname in os.listdir(MODEL_DIR):
    name = fname.lower()

    if ("block" in name) and ("expanding" in name) and (fname.endswith(".pkl") or fname.endswith(".json")):
        key = fname.replace(".pkl", "").replace(".json", "")
        MODELS[key] = load_model(os.path.join(MODEL_DIR, fname))

print(f"Loaded {len(MODELS)} BLOCK PCA models for calibration")


# ===============================
# ===== CALIBRATION UTILS =======
# ===============================

def regression_calibration_curve(y_true, y_pred, n_bins=10):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["bin"] = pd.qcut(df["y_pred"], q=n_bins, duplicates="drop")

    grouped = df.groupby("bin").agg(
        mean_pred=("y_pred", "mean"),
        mean_true=("y_true", "mean")
    ).reset_index()

    return grouped


def plot_regression_calibration(y_true, y_pred, title, save_path):
    calib_df = regression_calibration_curve(y_true, y_pred, N_BINS)

    plt.figure()
    plt.plot(calib_df["mean_pred"], calib_df["mean_true"], marker="o")
    plt.plot(
        [calib_df["mean_pred"].min(), calib_df["mean_pred"].max()],
        [calib_df["mean_pred"].min(), calib_df["mean_pred"].max()],
        linestyle="--", label="Perfect"
    )
    plt.xlabel("Predicted log(Views)")
    plt.ylabel("Actual log(Views)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def regression_to_probability(y_pred):
    return expit(y_pred)


def plot_top10_calibration(y_true, y_pred, title, save_path):
    y_true_binary = (y_true >= np.percentile(y_true, 90)).astype(int)
    y_prob = regression_to_probability(y_pred)

    prob_true, prob_pred = calibration_curve(
        y_true_binary, y_prob,
        n_bins=N_BINS,
        strategy="quantile"
    )

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ===============================
# ===== RUN EXPANDING CALIB =====
# ===============================

n_samples = len(X_BLOCK)

for model_name, model in MODELS.items():

    print(f"\n===== CALIBRATING MODEL (BLOCK PCA): {model_name} =====")

    all_y_true = []
    all_y_pred = []

    for train_idx, test_idx in expanding_splits(n_samples, N_SPLITS):
        X_train, X_test = X_BLOCK[train_idx], X_BLOCK[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        all_y_true.append(y_test)
        all_y_pred.append(preds)

    # Combine all folds
    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)

    # ---- Regression Calibration ----
    reg_path = f"{OUTPUT_DIR}/{model_name}_regression.png"
    plot_regression_calibration(
        y_true_all,
        y_pred_all,
        title="XGBoost | Regression Calibration (Expanding CV)",
        #title=f"{model_name} | Regression Calibration (Expanding CV)",
        save_path=reg_path
    )

    # ---- Top-10% Calibration ----
    top10_path = f"{OUTPUT_DIR}/{model_name}_top10.png"
    plot_top10_calibration(
        y_true_all,
        y_pred_all,
        title=f"{model_name} | Top-10% Calibration (Expanding CV)",
        save_path=top10_path
    )

    print(f"Saved regression + top10 plots for {model_name}")

print("\nDONE — 3 MODELS CALIBRATED USING EXPANDING CV + BLOCK PCA")
