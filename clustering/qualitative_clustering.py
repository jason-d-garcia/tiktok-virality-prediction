import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import umap
import hdbscan

# ================================
# 1. LOAD QUALITATIVE FEATURES
# ================================

# Replace with your actual file name
dfq = pd.read_csv("../data/video_qualitative_features_3s.csv")

# sort by video_id for consistency
dfq = dfq.sort_values("video_id").reset_index(drop=True)

print("Loaded qualitative features:", dfq.shape)
print(dfq.head())

# ================================
# 2. SELECT FEATURE COLUMNS
# ================================

feat_cols = [
    "motion_mean_3s",
    "motion_std_3s",
    "motion_entropy_3s",
    "frames_used_3s",
    "audio_energy_std_3s",
    "audio_energy_mean_3s"
]

Xq = dfq[feat_cols].values

# ================================
# 3. SCALE FEATURES
# ================================

Xq = StandardScaler().fit_transform(Xq)

# ================================
# 4. UMAP DIMENSIONALITY REDUCTION
# ================================

umap_q = umap.UMAP(
    n_neighbors=20,
    min_dist=0.1,
    n_components=2,
    random_state=42
).fit_transform(Xq)

# ================================
# 5. HDBSCAN CLUSTERING
# ================================

cluster_q = hdbscan.HDBSCAN(
    min_cluster_size=5,
    min_samples=2,
    metric="euclidean"
).fit_predict(umap_q)

dfq["qual_cluster"] = cluster_q

print("\nQualitative cluster counts:")
print(dfq["qual_cluster"].value_counts())

# ================================
# 6. SAVE QUALITATIVE CLUSTERS
# ================================

dfq.to_csv("qual_clusters.csv", index=False)
print("\nSaved qual_clusters.csv")

# ================================
# 7. PLOT QUALITATIVE CLUSTERS (POSTER READY)
# ================================

plt.figure(figsize=(8, 6))

for c in sorted(dfq["qual_cluster"].unique()):
    mask = dfq["qual_cluster"] == c
    plt.scatter(
        umap_q[mask, 0],
        umap_q[mask, 1],
        label=f"Cluster {c}",
        alpha=0.8
    )

plt.legend()
plt.title("Qualitative Motion + Audio Clusters (UMAP)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.savefig("qualitative_umap_clusters.png", dpi=200)
plt.show()

print("Saved qualitative_umap_clusters.png")

# ================================
# 8. LOAD EXISTING SEMANTIC CLUSTERS
# ================================

# This should contain: video_id, cluster, log_views
df_semantic = pd.read_csv("../data/clusters/video_cluster_labels.csv")

# Select index, cluster, and log_views only
df_semantic = df_semantic[["Index", "cluster", "log_views"]]

print("\nLoaded semantic clusters:", df_semantic.shape)
print(df_semantic.head())

# ================================
# 9. MERGE QUALITATIVE + SEMANTIC
# ================================
# ---- Normalize both to the same integer key ----

# strip .mp4 and leading zeros from video_id in dfq

df_semantic["merge_id"] = df_semantic["Index"].astype(int)
dfq["merge_id"] = (
    dfq["video_id"]
    .str.replace(".mp4", "", regex=False)   # remove suffix
    .str.lstrip("0")                        # remove left zero padding
    .astype(int)
)

# ---- Sanity checks ----
print("Main IDs:", df_semantic["merge_id"].head().tolist())
print("Qual IDs:", dfq["merge_id"].head().tolist())

assert df_semantic["merge_id"].nunique() == len(df_semantic), "Duplicate IDs in main"
assert dfq["merge_id"].nunique() == len(dfq), "Duplicate IDs in qual"

# ---- Merge safely ----
df_merged = df_semantic.merge(
    dfq.drop(columns=["video_id"]),
    on="merge_id",
    how="inner"
)

# ---- Final verification ----
assert len(df_merged) == len(df_semantic), "Row loss after merge!"
assert df_merged.isna().sum().sum() == 0, "NaNs introduced during merge!"

# ---- Cleanup ----
df_merged = df_merged.drop(columns=["merge_id"])

# ---- Save ----
df_merged.to_csv("full_with_qual_features.csv", index=False)

print("✅ Qualitative features successfully merged and saved.")
# ================================
# 10. PERFORMANCE BY QUALITATIVE STYLE
# ================================

perf_by_qual = df_merged.groupby("qual_cluster")["log_views"].agg([
    "count", "mean", "median", "std"
])

print("\nPerformance by Qualitative Cluster:")
print(perf_by_qual)

perf_by_qual.to_csv("qual_cluster_performance.csv")
print("\nSaved qual_cluster_performance.csv")

# ================================
# 11. QUALITATIVE STYLE DISTRIBUTION WITHIN EACH SEMANTIC CLUSTER
# ================================

style_mix = (
    df_merged
    .groupby("cluster")["qual_cluster"]
    .value_counts(normalize=True)
    .rename("pct")
    .reset_index()
)

print("\nQualitative Style Mix by Semantic Cluster:")
print(style_mix)

style_mix.to_csv("semantic_vs_qualitative_mix.csv", index=False)
print("\nSaved semantic_vs_qualitative_mix.csv")

# ================================
# 12. OPTIONAL: BAR PLOT OF PERFORMANCE BY QUALITATIVE CLUSTER
# ================================

plt.figure(figsize=(7,5))
plt.bar(
    perf_by_qual.index.astype(str),
    perf_by_qual["mean"]
)
plt.title("Mean Log Views by Qualitative Cluster")
plt.xlabel("Qualitative Cluster")
plt.ylabel("Mean Log Views")
plt.tight_layout()
plt.savefig("qual_cluster_performance_bar.png", dpi=200)
plt.show()

print("Saved qual_cluster_performance_bar.png")

# ================================
# DONE
# ================================

print("\n✅ Qualitative clustering pipeline complete.")

import matplotlib.pyplot as plt
import numpy as np



# X_qual_umap: (N,2) UMAP from motion+audio features
# virality_cluster_labels: (N,) from your original HDBSCAN on model embeddings
virality_cluster_labels = np.load("../data/clusters/cluster_labels.npy")
X_qual_features = Xq  # (N, D) motion+audio features
log_views = df_merged["log_views"].values  # (N,)

plt.figure(figsize=(7, 6))
sc = plt.scatter(
    umap_q[:, 0],
    umap_q[:, 1],
    c=virality_cluster_labels,
    cmap="tab10",
    s=45,
    alpha=0.9
)
plt.colorbar(sc, label="Virality Cluster")
plt.title("Audiovisual UMAP Colored by Virality Cluster")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.savefig("qual_umap_colored_by_virality.png", dpi=200)
plt.close()

unique_viral = np.unique(virality_cluster_labels)

for v in unique_viral:
    if v == -1:
        continue

    mask = virality_cluster_labels == v
    X_sub = X_qual_features[mask]

    if len(X_sub) < 12:
        print(f"Skipping virality cluster {v} (too small)")
        continue

    sub_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=3
    )
    sub_labels = sub_clusterer.fit_predict(X_sub)

    # Plot
    X_sub_umap = umap_q[mask]

    plt.figure(figsize=(6, 5))
    plt.scatter(
        X_sub_umap[:, 0],
        X_sub_umap[:, 1],
        c=sub_labels,
        cmap="tab10",
        s=45
    )
    plt.title(f"Audiovisual Subclusters Inside Virality Cluster {v}")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(f"qual_subclusters_inside_virality_{v}.png", dpi=200)
    plt.close()

plt.figure(figsize=(7, 6))
sc = plt.scatter(
    umap_q[:, 0],
    umap_q[:, 1],
    c=log_views,
    cmap="viridis",
    s=45
)
plt.colorbar(sc, label="Log Views")
plt.title("Audiovisual UMAP Colored by Performance")
plt.tight_layout()
plt.savefig("qual_umap_colored_by_log_views.png", dpi=200)
plt.close()

from sklearn.metrics import silhouette_score
sil_score = silhouette_score(X_qual_features, df_merged["qual_cluster"])
print("Silhouette score:", sil_score)