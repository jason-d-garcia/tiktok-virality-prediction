import numpy as np
import pandas as pd
import pickle
import os

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


from scipy.stats import spearmanr

try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False

# =========================
# ===== CONFIG ===========
# =========================

MODEL_DIR = "../models"
DATA_PATH = "../data/embeddings/tiktok_full_features.pkl"
X_PATH = "../data/embeddings/X_features.npy"

TARGET_NAME = "log_views"

MODEL_TYPES = ["ridge", "elastic", "xgboost"]

PCA_MODES = ["none", "global", "block"]
GLOBAL_PCA_DIM = 64
BLOCK_PCA_DIMS = {
    "text": 32,
    "clip": 32,
    "whisper": 32
}

CV_MODES = ["standard", "expanding"]

N_SPLITS = 3
RANDOM_STATE = 42

COMPARE_METADATA_ONLY = True

# =========================
# ===== MODEL SAVE ========
# ========================

def save_model(model, model_name):
    path_base = os.path.join(MODEL_DIR, model_name)

    # XGBoost → JSON
    if HAS_XGB and isinstance(model, xgb.XGBRegressor):
        model.save_model(path_base + ".json")
        print(f"✅ Saved XGBoost model to {path_base}.json")

    # Sklearn models → Pickle
    else:
        with open(path_base + ".pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"✅ Saved sklearn model to {path_base}.pkl")

def train_and_save_final_model(X, y, model_type, label):
    model = build_model(model_type)
    model.fit(X, y)

    safe_label = label.lower().replace(" ", "_").replace("|", "").replace("=", "")
    name = f"{model_type}_{safe_label}_final"

    save_model(model, name)


# =========================
# ===== LOAD DATA =========
# =========================

df = pd.read_pickle(DATA_PATH)
X_full = np.load(X_PATH)
y = df[TARGET_NAME].values

print(f"\nLoaded X shape: {X_full.shape}")
print(f"Loaded target: {TARGET_NAME}")

# =========================
# ===== FEATURE BLOCKS ====
# =========================

IDX_NUMERIC_END = 21
IDX_DESC_END    = 21 + 512
IDX_TAGS_END    = 21 + 512 + 512
IDX_CLIP_END    = 21 + 512 + 512 + 512
IDX_WHISPER_END = 21 + 512 + 512 + 512 + 512  # = 2069

X_numeric    = X_full[:, :IDX_NUMERIC_END]
X_text_desc = X_full[:, IDX_NUMERIC_END:IDX_DESC_END]
X_text_tags = X_full[:, IDX_DESC_END:IDX_TAGS_END]
X_clip      = X_full[:, IDX_TAGS_END:IDX_CLIP_END]
X_whisper   = X_full[:, IDX_CLIP_END:IDX_WHISPER_END]

# -----------------------------
# OUTLIER / BREAKOUT DETECTION
# -----------------------------

BREAKOUT_THRESHOLD = np.percentile(df["views"], 90)  # top 3%
df["is_breakout"] = df["views"] >= BREAKOUT_THRESHOLD

print(f"Breakout threshold: {int(BREAKOUT_THRESHOLD):,} views")
print(f"Breakout count: {df['is_breakout'].sum()}")

# -----------------------------
# METRICS
# -----------------------------

def clipped_r2(y_true, y_pred, clip_pct=95):
    cap = np.percentile(y_true, clip_pct)
    y_true_clip = np.clip(y_true, None, cap)
    y_pred_clip = np.clip(y_pred, None, cap)
    return r2_score(y_true_clip, y_pred_clip)

def smape(y_true, y_pred, eps=1e-8):
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + eps
    return np.mean(numerator / denominator)

def top_k_hit_rate(y_true, y_pred, k=0.10):
    n = len(y_true)
    k_n = int(n * k)

    true_top_idx = np.argsort(y_true)[-k_n:]
    pred_top_idx = np.argsort(y_pred)[-k_n:]

    hits = len(set(true_top_idx).intersection(set(pred_top_idx)))
    return hits / k_n

# =========================
# ===== FEATURE MASKS ====
# =========================

# numeric metadata features only (no embeddings)
numeric_features = [
    "video length",
    "log_video_length",
    "video_length_sq",

    "hashtag_count",

    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "doy_sin", "doy_cos",

    "videos_last_7d", "videos_last_14d", "videos_last_28d",
    "views_last_7d", "views_last_14d", "views_last_28d",
    "views_per_video_last_7d", "views_per_video_last_14d", "views_per_video_last_28d",
]

X_metadata = df[numeric_features].values

# =========================
# ===== PCA FOR FULL =====
# =========================

def apply_global_pca(X, dim):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=dim, random_state=RANDOM_STATE)
    return pca.fit_transform(X_scaled)

def apply_block_pca():
    def pca_block(X, dim):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=dim, random_state=RANDOM_STATE)
        return pca.fit_transform(X_scaled)

    X_desc_pca = pca_block(X_text_desc, BLOCK_PCA_DIMS["text"])
    X_tags_pca = pca_block(X_text_tags, BLOCK_PCA_DIMS["text"])
    X_clip_pca = pca_block(X_clip, BLOCK_PCA_DIMS["clip"])
    X_whisp_pca= pca_block(X_whisper, BLOCK_PCA_DIMS["whisper"])

    return np.concatenate([
        X_numeric,
        X_desc_pca,
        X_tags_pca,
        X_clip_pca,
        X_whisp_pca
    ], axis=1)


def build_feature_set(pca_mode):
    if pca_mode == "none":
        return X_full

    elif pca_mode == "global":
        print(f"Applying GLOBAL PCA → {GLOBAL_PCA_DIM}")
        return apply_global_pca(X_full, GLOBAL_PCA_DIM)

    elif pca_mode == "block":
        print("Applying BLOCK PCA")
        return apply_block_pca()

    else:
        raise ValueError("Invalid PCA mode")

X_FEATURE_SETS = {}

# --- NO PCA ---
X_FEATURE_SETS["none"] = X_full

# --- GLOBAL PCA ---
X_FEATURE_SETS["global"] = apply_global_pca(X_full, GLOBAL_PCA_DIM)

# --- BLOCK PCA ---
X_FEATURE_SETS["block"] = apply_block_pca()

# Save feature sets for reproducibility
os.makedirs("../data/saved_features", exist_ok=True)

np.save("../data/saved_features/X_metadata.npy", X_metadata)
np.save("../data/saved_features/X_full.npy", X_FEATURE_SETS["none"])
np.save("../data/saved_features/X_pca_global.npy", X_FEATURE_SETS["global"])
np.save("../data/saved_features/X_pca_block.npy", X_FEATURE_SETS["block"])
np.save("../data/saved_features/y.npy", y)

# =========================
# ====== BUILD MODEL ======
# =========================

def build_model(model_type):
    if model_type == "ridge":
        return Ridge(alpha=10.0)

    elif model_type == "elastic":
        return ElasticNet(alpha=0.01, l1_ratio=0.2)

    elif model_type == "xgboost":
        if not HAS_XGB:
            raise RuntimeError("XGBoost not installed")

        return xgb.XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
        )

    else:
        raise ValueError("Invalid MODEL_TYPE")


# =========================
# ===== TIME SPLIT =======
# =========================

def get_cv_splitter(mode, n_splits):
    if mode == "standard":
        return TimeSeriesSplit(n_splits=n_splits)

    elif mode == "expanding":
        class ExpandingSplit:
            def split(self, X):
                n = len(X)
                fold_size = n // (n_splits + 1)

                for i in range(n_splits):
                    train_end = fold_size * (i + 1)
                    test_end = fold_size * (i + 2)

                    train_idx = np.arange(0, train_end)
                    test_idx = np.arange(train_end, min(test_end, n))

                    yield train_idx, test_idx

        return ExpandingSplit()

    else:
        raise ValueError("Invalid CV mode")


# =========================
# ===== TRAIN LOOP =======
# =========================

def run_cv_experiment(X, y, label, model_type, cv_mode):
    print(f"\n----- {label} | MODEL={model_type} | CV={cv_mode} -----")

    cv = get_cv_splitter(cv_mode, N_SPLITS)

    r2s, rmses, maes, smapes = [], [], [], []
    clipped_r2s, spearmans, topk_hits = [], [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = build_model(model_type)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        smape_val = smape(y_test, preds)
        clipped = clipped_r2(y_test, preds)
        spear = spearmanr(y_test, preds).correlation
        topk = top_k_hit_rate(y_test, preds)

        r2s.append(r2)
        rmses.append(rmse)
        maes.append(mae)
        smapes.append(smape_val)
        clipped_r2s.append(clipped)
        spearmans.append(spear)
        topk_hits.append(topk)

        print(
            f"Fold {fold+1} | "
            f"R2={r2:.3f} | RMSE={rmse:.3f} | "
            f"MAE={mae:.3f} | sMAPE={smape_val:.3f} | "
            f"Spearman={spear:.3f} | "
            f"Top-10% Hit={topk:.2f}"
        )

    # Train final model on full data
    final_model = build_model(model_type)
    final_model.fit(X, y)

    return {
        "label": label,
        "model": model_type,
        "cv_mode": cv_mode,
        "mean_r2": np.mean(r2s),
        "mean_rmse": np.mean(rmses),
        "mean_mae": np.mean(maes),
        "mean_smape": np.mean(smapes),
        "mean_spearman": np.mean(spearmans),
        "mean_topk": np.mean(topk_hits),
        "final_model": final_model,
    }

# =========================
# ===== RUN BOTH =========
# =========================

results = []

for model_type in MODEL_TYPES:
    for cv_mode in CV_MODES:

        # ---- Metadata baseline ----
        if COMPARE_METADATA_ONLY:
            res = run_cv_experiment(
                X_metadata, y,
                "METADATA ONLY",
                model_type,
                cv_mode
            )

            model_name = f"{model_type}_metadata_{cv_mode}"
            save_model(res["final_model"], model_name)

            res.pop("final_model")  # remove before CSV logging
            results.append(res)

        # ---- PCA modes ----
        for pca_mode in PCA_MODES:
            X_use = X_FEATURE_SETS[pca_mode]
            label = f"FULL | PCA_MODE={pca_mode}"

            res = run_cv_experiment(
                X_use, y,
                label,
                model_type,
                cv_mode
            )

            model_name = f"{model_type}_{label.replace(' ', '_').lower()}_{cv_mode}"
            save_model(res["final_model"], model_name)

            res.pop("final_model")  # remove before CSV logging
            results.append(res)

# Convert to table for poster / README
results_df = pd.DataFrame(results)
results_df.to_csv("experiment_results.csv", index=False)
print("\n\nFINAL RESULTS TABLE:")
print(results_df)