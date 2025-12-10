"""
shap_cluster_analysis.py

Analysis for:
- XGBoost full model with block PCA ("block_standard")
- Global SHAP explanations
- Block-level SHAP importance (numeric vs text_desc vs text_tags vs clip vs whisper)
- UMAP + HDBSCAN clustering of videos in representation space
- Cluster-wise SHAP fingerprints and summary stats
- NEW: Within-cluster top vs bottom performance virality rules
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shap
import umap
import hdbscan
import xgboost as xgb

from collections import defaultdict

# ====================================================
# ================ CONFIG ============================
# ====================================================

# Paths – adjust if needed
DATA_PATH = "../data/embeddings/tiktok_full_features.pkl"
BLOCK_FEATURES_PATH = "../src/saved_features/X_pca_block.npy"  # X_FEATURE_SETS["block"]
MODEL_DIR = "../models"
MODEL_NAME = "xgboost_full_pca_mode=block_standard"         # no extension

RANDOM_STATE = 42

# Block layout for block PCA features
NUMERIC_DIM = 21  # X_numeric columns
BLOCK_PCA_DIMS = {
    "text": 32,     # desc PCA dim
    "clip": 32,
    "whisper": 32,
}
# We have TWO text blocks (desc + tags), each 32
TEXT_DESC_DIM = BLOCK_PCA_DIMS["text"]
TEXT_TAGS_DIM = BLOCK_PCA_DIMS["text"]
CLIP_DIM = BLOCK_PCA_DIMS["clip"]
WHISPER_DIM = BLOCK_PCA_DIMS["whisper"]

TOP_FEATURES_FOR_SUMMARY = 20
TOP_PERCENT = 20   # top 20% performers within cluster
BOTTOM_PERCENT = 20  # bottom 20% within cluster


# ====================================================
# ================ HELPERS ===========================
# ====================================================

def load_df_and_features():
    """Load dataframe and block-PCA feature matrix."""
    df = pd.read_pickle(DATA_PATH)
    X_block = np.load(BLOCK_FEATURES_PATH)

    assert len(df) == X_block.shape[0], "df and X_block size mismatch"
    return df.reset_index(drop=True), X_block


def load_xgb_model():
    """Load XGBRegressor from JSON saved in training script."""
    model_path = os.path.join(MODEL_DIR, MODEL_NAME + ".json")
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    print(f"✅ Loaded XGBoost model from {model_path}")
    return model


def get_block_indices():
    """Return index ranges for each feature block in the concatenated X."""
    start = 0
    idx_numeric = np.arange(start, start + NUMERIC_DIM)
    start += NUMERIC_DIM

    idx_text_desc = np.arange(start, start + TEXT_DESC_DIM)
    start += TEXT_DESC_DIM

    idx_text_tags = np.arange(start, start + TEXT_TAGS_DIM)
    start += TEXT_TAGS_DIM

    idx_clip = np.arange(start, start + CLIP_DIM)
    start += CLIP_DIM

    idx_whisper = np.arange(start, start + WHISPER_DIM)
    start += WHISPER_DIM

    return {
        "numeric": idx_numeric,
        "text_desc": idx_text_desc,
        "text_tags": idx_text_tags,
        "clip": idx_clip,
        "whisper": idx_whisper,
    }


def block_importance(shap_values, block_indices):
    """Compute mean |SHAP| per block."""
    block_scores = {}
    for block_name, idx in block_indices.items():
        block_scores[block_name] = np.mean(np.abs(shap_values[:, idx]))
    return block_scores


def plot_block_importance(block_scores, out_path=None):
    names = list(block_scores.keys())
    vals = [block_scores[k] for k in names]

    plt.figure(figsize=(6, 4))
    order = np.argsort(vals)[::-1]
    names_sorted = [names[i] for i in order]
    vals_sorted = [vals[i] for i in order]

    plt.bar(names_sorted, vals_sorted)
    plt.ylabel("Mean |SHAP value|")
    plt.title("Block-level importance (XGBoost, block PCA)")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200)
        print(f"📊 Saved block importance plot to {out_path}")
    else:
        plt.show()
    plt.close()


def plot_umap(X_emb, values, title, cmap="viridis", out_path=None):
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(X_emb[:, 0], X_emb[:, 1], c=values, s=35, alpha=0.9, cmap=cmap)
    plt.colorbar(sc)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200)
        print(f"🗺️ Saved UMAP plot to {out_path}")
    else:
        plt.show()
    plt.close()


def plot_umap_clusters(X_emb, cluster_labels, out_path=None):
    plt.figure(figsize=(7, 6))
    unique = np.unique(cluster_labels)

    for label in unique:
        mask = cluster_labels == label
        if label == -1:
            plt.scatter(
                X_emb[mask, 0], X_emb[mask, 1],
                c="lightgray", s=20, alpha=0.5, label="Noise (-1)"
            )
        else:
            plt.scatter(
                X_emb[mask, 0], X_emb[mask, 1],
                s=35, alpha=0.9, label=f"Cluster {label}"
            )

    plt.legend(loc="best", fontsize=8)
    plt.title("UMAP + HDBSCAN clusters")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200)
        print(f"🌐 Saved cluster UMAP plot to {out_path}")
    else:
        plt.show()
    plt.close()


# ====================================================
# ========== NEW: WITHIN-CLUSTER VIRALITY =============
# ====================================================

def analyze_within_cluster_virality(
    df,
    X,
    shap_values,
    y_true,
    cluster_labels,
    block_idx,
    out_dir,
    top_percent=TOP_PERCENT,
    bottom_percent=BOTTOM_PERCENT,
    min_cluster_size=15,
):
    """
    For each non-noise cluster:
      - find top and bottom percentiles by y_true (log_views)
      - compute block-level mean |SHAP| for top vs bottom
      - save CSV + bar plots
    """
    os.makedirs(out_dir, exist_ok=True)

    unique_clusters = np.unique(cluster_labels)
    rows = []

    for c in unique_clusters:
        if c == -1:
            continue  # skip noise

        cluster_mask = cluster_labels == c
        idx_cluster = np.where(cluster_mask)[0]
        n_c = len(idx_cluster)

        if n_c < min_cluster_size:
            print(f"Skipping cluster {c} (size {n_c} < {min_cluster_size})")
            continue

        y_c = y_true[idx_cluster]
        shap_c = shap_values[idx_cluster]

        # thresholds within this cluster
        low_thr = np.percentile(y_c, bottom_percent)
        high_thr = np.percentile(y_c, 100 - top_percent)

        bottom_local = np.where(y_c <= low_thr)[0]
        top_local = np.where(y_c >= high_thr)[0]

        if len(top_local) == 0 or len(bottom_local) == 0:
            print(f"Cluster {c}: not enough samples for top/bottom split, skipping.")
            continue

        shap_top = shap_c[top_local]
        shap_bottom = shap_c[bottom_local]

        # block scores
        block_top = block_importance(shap_top, block_idx)
        block_bottom = block_importance(shap_bottom, block_idx)

        # store rows for CSV
        for tier_name, block_scores in [("top", block_top), ("bottom", block_bottom)]:
            row = {
                "cluster": c,
                "cluster_size": n_c,
                "tier": tier_name,
                "tier_count": len(top_local) if tier_name == "top" else len(bottom_local),
                "mean_true_log_views_tier": float(y_c[top_local].mean())
                if tier_name == "top"
                else float(y_c[bottom_local].mean()),
            }
            for bname, score in block_scores.items():
                row[f"block_{bname}_mean_abs_shap"] = score
            rows.append(row)

        # --- Plot: block SHAP comparison for this cluster ---
        blocks = list(block_top.keys())
        top_vals = [block_top[b] for b in blocks]
        bottom_vals = [block_bottom[b] for b in blocks]

        x = np.arange(len(blocks))
        width = 0.35

        plt.figure(figsize=(5, 3.5))
        plt.bar(x - width / 2, bottom_vals, width, label=f"Bottom {bottom_percent}%")
        plt.bar(x + width / 2, top_vals, width, label=f"Top {top_percent}%")
        plt.xticks(x, blocks, rotation=25, ha="right")
        plt.ylabel("Mean |SHAP|")
        plt.title(f"Cluster {c}: block importance (top vs bottom)")
        plt.legend()
        plt.tight_layout()

        path = os.path.join(out_dir, f"cluster_{c}_top_vs_bottom_block_shap.png")
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"🧩 Saved top vs bottom block SHAP for cluster {c} → {path}")

    if rows:
        rules_df = pd.DataFrame(rows)
        csv_path = os.path.join(out_dir, "cluster_virality_rules.csv")
        rules_df.to_csv(csv_path, index=False)
        print(f"📄 Saved cluster top/bottom virality rules to {csv_path}")
    else:
        print("No clusters had enough data for top/bottom analysis.")


# ====================================================
# ================ MAIN ANALYSIS =====================
# ====================================================

def main():
    os.makedirs("../reports", exist_ok=True)
    out_dir = "../reports"

    # ----- Load data & model -----
    df, X = load_df_and_features()
    y_true = df["log_views"].values
    model = load_xgb_model()

    # ----- Predictions -----
    y_pred = model.predict(X)

    print(f"\nPrediction stats (log_views):")
    print(f"  mean true: {y_true.mean():.3f}, std: {y_true.std():.3f}")
    print(f"  mean pred: {y_pred.mean():.3f}, std: {y_pred.std():.3f}")

    # ====================================================
    # ============ SHAP GLOBAL ===========================
    # ====================================================

    print("\nFitting SHAP TreeExplainer…")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Global summary plot (top N)
    shap.summary_plot(
        shap_values,
        X,
        plot_type="dot",
        max_display=TOP_FEATURES_FOR_SUMMARY,
        show=False,
    )
    plt.tight_layout()
    summary_path = os.path.join(out_dir, "shap_summary_block_xgb.png")
    plt.savefig(summary_path, dpi=250)
    plt.close()
    print(f"📈 Saved SHAP summary plot to {summary_path}")

    # Block-level importance
    block_idx = get_block_indices()
    block_scores = block_importance(shap_values, block_idx)
    print("\nBlock-level mean |SHAP| (global):")
    for k, v in block_scores.items():
        print(f"  {k:10s}: {v:.4f}")

    plot_block_importance(
        block_scores,
        out_path=os.path.join(out_dir, "shap_block_importance.png"),
    )

    # ====================================================
    # ============ UMAP + HDBSCAN ========================
    # ====================================================

    print("\nRunning UMAP on block PCA features…")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=RANDOM_STATE,
    )
    X_umap = reducer.fit_transform(X)
    print("UMAP embedding shape:", X_umap.shape)

    print("Running HDBSCAN clustering…")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=6,
        min_samples=4,
        cluster_selection_epsilon=0.0,
        metric="euclidean",
    )
    cluster_labels = clusterer.fit_predict(X_umap)
    df["cluster"] = cluster_labels

    # ================================
    # ===== SAVE CLUSTER LABELS ======
    # ================================

    CLUSTER_SAVE_DIR = "../data/clusters"
    os.makedirs(CLUSTER_SAVE_DIR, exist_ok=True)

    # 1. Save per-video cluster labels with metadata
    cluster_csv_path = os.path.join(CLUSTER_SAVE_DIR, "video_cluster_labels.csv")
    df_out = df.copy()
    df_out["cluster"] = cluster_labels
    df_out.to_csv(cluster_csv_path, index=False)
    print(f"✅ Saved video cluster labels to {cluster_csv_path}")

    # 2. Save raw labels for fast reload
    cluster_npy_path = os.path.join(CLUSTER_SAVE_DIR, "cluster_labels.npy")
    np.save(cluster_npy_path, cluster_labels)
    print(f"✅ Saved raw cluster labels to {cluster_npy_path}")

    # 3. Save the UMAP embedding too (IMPORTANT for plots later)
    umap_npy_path = os.path.join(CLUSTER_SAVE_DIR, "X_umap.npy")
    np.save(umap_npy_path, X_umap)
    print(f"✅ Saved UMAP embedding to {umap_npy_path}")

    print("\nCluster sizes (including noise=-1):")
    print(df["cluster"].value_counts().sort_index())

    # UMAP plots
    plot_umap(
        X_umap,
        y_pred,
        title="UMAP embedding colored by predicted log_views",
        out_path=os.path.join(out_dir, "umap_pred_log_views.png"),
    )
    plot_umap_clusters(
        X_umap,
        cluster_labels,
        out_path=os.path.join(out_dir, "umap_clusters.png"),
    )

    # ====================================================
    # ============ CLUSTER-WISE SHAP PROFILES ============
    # ====================================================

    print("\nComputing cluster-wise SHAP fingerprints…")
    cluster_shap = defaultdict(dict)
    unique_clusters = np.unique(cluster_labels)

    for c in unique_clusters:
        mask = cluster_labels == c
        n_c = mask.sum()
        if n_c == 0:
            continue

        shap_c = shap_values[mask]
        y_true_c = y_true[mask]
        y_pred_c = y_pred[mask]

        mean_abs_shap = np.mean(np.abs(shap_c), axis=0)
        block_scores_c = block_importance(shap_c, block_idx)

        cluster_shap[c]["size"] = int(n_c)
        cluster_shap[c]["mean_true_log_views"] = float(y_true_c.mean())
        cluster_shap[c]["mean_pred_log_views"] = float(y_pred_c.mean())
        cluster_shap[c]["block_scores"] = block_scores_c
        cluster_shap[c]["mean_abs_shap"] = mean_abs_shap

    rows = []
    for c, info in cluster_shap.items():
        row = {
            "cluster": c,
            "size": info["size"],
            "mean_true_log_views": info["mean_true_log_views"],
            "mean_pred_log_views": info["mean_pred_log_views"],
        }
        for bname, score in info["block_scores"].items():
            row[f"block_{bname}_mean_abs_shap"] = score
        rows.append(row)

    cluster_summary_df = pd.DataFrame(rows).sort_values("cluster")
    csv_path = os.path.join(out_dir, "cluster_shap_summary.csv")
    cluster_summary_df.to_csv(csv_path, index=False)
    print(f"📄 Saved cluster SHAP summary to {csv_path}")

    for c, info in cluster_shap.items():
        if c == -1:
            continue
        scores = info["block_scores"]
        names = list(scores.keys())
        vals = [scores[k] for k in names]
        order = np.argsort(vals)[::-1]

        plt.figure(figsize=(4, 3))
        plt.bar([names[i] for i in order], [vals[i] for i in order])
        plt.title(f"Cluster {c} block SHAP\n(n={info['size']})")
        plt.ylabel("Mean |SHAP|")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        path = os.path.join(out_dir, f"cluster_{c}_block_shap.png")
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"🧩 Saved block SHAP plot for cluster {c} to {path}")

    # ====================================================
    # ============ NEW: WITHIN-CLUSTER RULES ============
    # ====================================================

    print("\nAnalyzing within-cluster top vs bottom performance…")
    analyze_within_cluster_virality(
        df=df,
        X=X,
        shap_values=shap_values,
        y_true=y_true,
        cluster_labels=cluster_labels,
        block_idx=block_idx,
        out_dir=os.path.join(out_dir, "within_cluster"),
        top_percent=TOP_PERCENT,
        bottom_percent=BOTTOM_PERCENT,
        min_cluster_size=15,
    )

    print("\n✅ All analysis done.")


if __name__ == "__main__":
    main()
