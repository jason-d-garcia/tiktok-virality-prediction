"""
Leakage-safe evaluation harness.

Differences from train.py:
  - Scaling and PCA are fit inside each training fold (train.py fits them on
    all 140 rows before splitting, so test videos shape the projection).
  - Every model is compared against a creator baseline: the median log views
    of the creator's previous K posts. A model that can't beat this is not
    learning anything a creator doesn't already know.
  - One chronological splitter. train.py's "standard" and "expanding" modes
    produce identical folds for n=140, n_splits=3.
  - Optional "no trailing" feature set. views_last_*d sums the *final* view
    counts of earlier posts, which were not known when the target video was
    published.

Runs from the project root:
    python -m src.evaluate                 # metadata features only (raw CSV)
    python -m src.evaluate --embeddings    # + CLIP/Whisper blocks, if built
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from preprocessing.preprocessing import (
    parse_metric,
    add_datetime_features,
    add_hashtag_features,
    add_trailing_metrics,
    add_video_length_features,
    add_target_columns,
)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# =========================
# ===== CONFIG ===========
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "tiktok_stats.csv"
EMBED_DIR = PROJECT_ROOT / "data" / "embeddings"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "evaluation_results.csv"

N_SPLITS = 3
BASELINE_K = 10
BLOCK_PCA_DIM = 32
RANDOM_STATE = 42

META_FEATURES = [
    "video length", "log_video_length", "video_length_sq",
    "hashtag_count",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
]
TRAILING_FEATURES = [
    f"{stat}_last_{w}d"
    for w in (7, 14, 28)
    for stat in ("videos", "views", "views_per_video")
]

# =========================
# ===== DATA =============
# =========================

def load_frame():
    df = pd.read_csv(RAW_CSV_PATH)
    for col in ["views", "likes", "comments", "shares", "saves"]:
        df[col] = df[col].apply(parse_metric)
    df = add_target_columns(df)
    df = add_datetime_features(df, "published date", "published time")
    df = add_video_length_features(df, "video length")
    df = add_hashtag_features(df, "description")
    df = add_trailing_metrics(df, "published_datetime", "views", (7, 14, 28))
    return df


def load_embedding_blocks(n_rows):
    """CLIP video + Whisper blocks, in the same chronological row order as the CSV."""
    blocks = {}
    for name in ("clip", "whisper"):
        path = EMBED_DIR / f"X_{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run the preprocessing extractors first")
        X = np.load(path)
        assert X.shape[0] == n_rows, f"Row mismatch: {name} has {X.shape[0]} rows, CSV has {n_rows}"
        blocks[name] = X
    return blocks

# =========================
# ===== MODELS ===========
# =========================

class FoldSafeFeatures:
    """Scales tabular columns and PCA-reduces each embedding block, fit on train rows only."""

    def __init__(self, X_tab, blocks, pca_dim):
        self.X_tab = X_tab
        self.blocks = blocks
        self.pca_dim = pca_dim

    def transform(self, train_idx, test_idx):
        tab_scaler = StandardScaler().fit(self.X_tab[train_idx])
        train_parts = [tab_scaler.transform(self.X_tab[train_idx])]
        test_parts = [tab_scaler.transform(self.X_tab[test_idx])]

        for X in self.blocks.values():
            scaler = StandardScaler().fit(X[train_idx])
            dim = min(self.pca_dim, len(train_idx) - 1)
            pca = PCA(n_components=dim, random_state=RANDOM_STATE)
            pca.fit(scaler.transform(X[train_idx]))
            train_parts.append(pca.transform(scaler.transform(X[train_idx])))
            test_parts.append(pca.transform(scaler.transform(X[test_idx])))

        return np.hstack(train_parts), np.hstack(test_parts)


def build_model(model_type):
    if model_type == "ridge":
        return Ridge(alpha=10.0)
    if model_type == "xgboost":
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
    raise ValueError(f"Invalid model type: {model_type}")


def creator_baseline(y, test_idx, k=BASELINE_K):
    """Median log views of the previous k posts (only posts before each test video)."""
    return np.array([np.median(y[max(0, i - k):i]) for i in test_idx])

# =========================
# ===== METRICS ==========
# =========================

def score(y_true, y_pred):
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "spearman": spearmanr(y_true, y_pred).correlation,
    }


def summarize(label, fold_scores):
    out = {"label": label}
    for key in fold_scores[0]:
        vals = np.array([s[key] for s in fold_scores], dtype=float)
        out[f"mean_{key}"] = np.nanmean(vals)
        out[f"fold_{key}"] = " ".join(f"{v:.2f}" for v in vals)
    return out

# =========================
# ===== RUN ==============
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", action="store_true", help="add CLIP/Whisper blocks")
    args = parser.parse_args()

    df = load_frame()
    y = df["log_views"].values
    blocks = load_embedding_blocks(len(df)) if args.embeddings else {}

    feature_sets = {
        "metadata": META_FEATURES,
        "metadata+trailing": META_FEATURES + TRAILING_FEATURES,
    }

    splits = list(TimeSeriesSplit(n_splits=N_SPLITS).split(y))
    results = []

    baseline_scores = [score(y[te], creator_baseline(y, te)) for _, te in splits]
    results.append(summarize(f"creator baseline (median of last {BASELINE_K})", baseline_scores))

    for set_name, cols in feature_sets.items():
        features = FoldSafeFeatures(df[cols].fillna(0).values, blocks, BLOCK_PCA_DIM)
        for model_type in ("ridge", "xgboost"):
            fold_scores = []
            for train_idx, test_idx in splits:
                X_train, X_test = features.transform(train_idx, test_idx)
                model = build_model(model_type)
                model.fit(X_train, y[train_idx])
                fold_scores.append(score(y[test_idx], model.predict(X_test)))
            suffix = " + clip/whisper" if blocks else ""
            results.append(summarize(f"{model_type} | {set_name}{suffix}", fold_scores))

    results_df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(results_df[["label", "mean_r2", "mean_mae", "mean_spearman", "fold_mae"]].round(3))
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
