"""
cluster_posting_policy_sliding.py

Computes statistically stable posting-time policies using:

✅ 4-hour sliding windows
✅ Per-cluster optimization
✅ Empirical Bayes shrinkage
✅ Minimum sample enforcement

Outputs:
  ../reports/cluster_posting_policies_smoothed.csv
"""

import numpy as np
import pandas as pd
import os

# =========================
# CONFIG
# =========================

DATA_PATH = "../data/embeddings/tiktok_full_features.pkl"
OUT_PATH = "../reports/cluster_posting_policies_smoothed.csv"
CLUSTER_CSV_PATH = "../data/clusters/video_cluster_labels.csv"

WINDOW_SIZE = 4      # hours
MIN_SAMPLES = 8      # minimum per window to be trusted
SHRINKAGE_K = 10     # higher = more conservative smoothing

# ===============================
# Load main dataframe
# ===============================

df = pd.read_pickle(DATA_PATH)

# ===============================
# Attach cluster labels
# ===============================

clusters_df = pd.read_csv(CLUSTER_CSV_PATH)

# Expected column name check (adjust if needed)
assert "cluster" in clusters_df.columns, "CSV must contain a 'cluster' column"

# Safety check: row alignment
assert len(df) == len(clusters_df), "Cluster CSV and df row counts do not match"

df["cluster"] = clusters_df["cluster"].values

print("✅ Cluster labels successfully attached")
print(df["cluster"].value_counts().sort_index())

# =========================
# GLOBAL BASELINE
# =========================

global_mean = df["log_views"].mean()

# =========================
# SLIDING WINDOWS
# =========================

def get_sliding_windows():
    windows = []
    for start in range(24):
        window = [(start + i) % 24 for i in range(WINDOW_SIZE)]
        windows.append(window)
    return windows

windows = get_sliding_windows()

# =========================
# CLUSTER LOOP
# =========================

rows = []

for cluster_id in sorted(df["cluster"].unique()):
    df_c = df[df["cluster"] == cluster_id]

    if len(df_c) < 15:
        print(f"Skipping cluster {cluster_id} (too small)")
        continue

    baseline = df_c["log_views"].mean()

    best_score = -np.inf
    best_window = None
    best_n = None

    for w in windows:
        df_w = df_c[df_c["hour"].isin(w)]

        n = len(df_w)
        if n < MIN_SAMPLES:
            continue

        mean_w = df_w["log_views"].mean()

        # ✅ Empirical Bayes shrinkage
        smoothed = (n * mean_w + SHRINKAGE_K * baseline) / (n + SHRINKAGE_K)

        if smoothed > best_score:
            best_score = smoothed
            best_window = w
            best_n = n

    if best_window is None:
        print(f"⚠️ No valid window found for cluster {cluster_id}")
        continue

    rows.append({
        "cluster": cluster_id,
        "n_videos_cluster": len(df_c),
        "best_window_hours": best_window,
        "center_hour": int(np.round(np.mean(best_window)) % 24),
        "window_size": WINDOW_SIZE,
        "window_sample_count": best_n,
        "cluster_baseline_log_views": baseline,
        "expected_log_views_at_best_time": best_score,
        "expected_lift_log": best_score - baseline,
        "expected_lift_multiplier": float(np.exp(best_score - baseline))
    })

# =========================
# SAVE RESULTS
# =========================

policy_df = pd.DataFrame(rows).sort_values("cluster")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
policy_df.to_csv(OUT_PATH, index=False)

print("\n✅ Posting policy saved to:")
print(OUT_PATH)
print(policy_df)
