import pandas as pd
import os

# =========================
# CONFIG
# =========================

DATA_PATH = "../data/embeddings/tiktok_full_features.pkl"
CLUSTER_PATH = "../data/clusters/video_cluster_labels.csv"

TARGET_COL = "log_views"
DATETIME_COL = "published_datetime"

OUT_DIR = "../reports/posting_policy"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================

df = pd.read_pickle(DATA_PATH)
clusters_df = pd.read_csv(CLUSTER_PATH)

df["cluster"] = clusters_df["cluster"].values
df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])

df["hour"] = df[DATETIME_COL].dt.hour
df["dow"] = df[DATETIME_COL].dt.dayofweek  # 0=Mon

#print(df["hour"].value_counts().sort_index())

print("✅ Loaded data with cluster labels")

# =========================
# PER-CLUSTER POSTING POLICY
# =========================

policies = []

for c in sorted(df["cluster"].unique()):
    if c == -1:
        continue  # skip noise

    sub = df[df["cluster"] == c]

    if len(sub) < 15:
        continue

    # Mean performance by (hour, day)
    grid = (
        sub
        .groupby(["dow", "hour"])[TARGET_COL]
        .mean()
        .reset_index()
    )

    best = grid.loc[grid[TARGET_COL].idxmax()]

    policy = {
        "cluster": int(c),
        "n_videos": len(sub),
        "best_day_of_week": int(best["dow"]),
        "best_hour": int(best["hour"]),
        "expected_log_views": float(best[TARGET_COL]),
        "baseline_log_views": float(sub[TARGET_COL].mean()),
        "expected_uplift": float(best[TARGET_COL] - sub[TARGET_COL].mean()),
    }

    policies.append(policy)

policy_df = pd.DataFrame(policies)
policy_path = os.path.join(OUT_DIR, "cluster_posting_policies.csv")
policy_df.to_csv(policy_path, index=False)

print("\n✅ SAVED CLUSTER POSTING POLICIES:")
print(policy_df)
print(f"\n📄 Saved to: {policy_path}")
