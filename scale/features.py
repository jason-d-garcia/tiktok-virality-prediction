"""
Tabular features for the post table, built only from what is known at publish time
(plus post age, which is a control for cumulative views, not a predictor).

Creator history uses the *observed* views of a creator's earlier posts. Those were
read at collection time, not at publish time, so they are an optimistic stand-in
for what the creator knew. Every post's history covers strictly earlier posts.
"""

import numpy as np

HASHTAG_PATTERN = r"#\w+"

TABULAR_FEATURES = [
    "log_duration", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "caption_len", "hashtag_count",
    "log_age_days", "age_known",
    "creator_prior_posts", "creator_prior_median", "creator_prior_mean",
    "creator_prior_std", "creator_has_history", "log_days_since_last",
]
FOLLOWER_FEATURES = ["log_followers", "followers_known"]


def add_features(df):
    df = df.sort_values(["creator_id", "create_time"]).reset_index(drop=True)
    df["y_log"] = np.log1p(df["views"])

    t = df["create_time"]
    df["log_duration"] = np.log1p(df["duration_s"].fillna(df["duration_s"].median()))
    df["hour_sin"] = np.sin(2 * np.pi * t.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * t.dt.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * t.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * t.dt.dayofweek / 7)
    df["caption_len"] = np.log1p(df["caption"].str.len())
    df["hashtag_count"] = df["caption"].str.count(HASHTAG_PATTERN)

    age_days = (df["collected_time"] - df["create_time"]).dt.total_seconds() / 86400
    df["age_known"] = age_days.notna().astype(float)
    df["log_age_days"] = np.log1p(age_days.clip(lower=0)).fillna(0.0)

    grp = df.groupby("creator_id")["y_log"]
    prior = lambda s: s.shift(1)
    df["creator_prior_posts"] = np.log1p(grp.cumcount())
    df["creator_prior_median"] = grp.transform(lambda s: prior(s).expanding().median())
    df["creator_prior_mean"] = grp.transform(lambda s: prior(s).expanding().mean())
    df["creator_prior_std"] = grp.transform(lambda s: prior(s).expanding().std())
    df["creator_has_history"] = df["creator_prior_median"].notna().astype(float)

    gap = df.groupby("creator_id")["create_time"].diff().dt.total_seconds() / 86400
    df["log_days_since_last"] = np.log1p(gap.clip(lower=0)).fillna(0.0)

    # Creators with no history fall back to the global median, flagged by creator_has_history.
    global_median = df["y_log"].median()
    for col in ("creator_prior_median", "creator_prior_mean"):
        df[col] = df[col].fillna(global_median)
    df["creator_prior_std"] = df["creator_prior_std"].fillna(0.0)

    df["followers_known"] = df["followers"].notna().astype(float)
    df["log_followers"] = np.log1p(df["followers"]).fillna(0.0)
    return df


def creator_baseline(df, k=10):
    """Median log views of each creator's previous k posts (NaN for a creator's first post)."""
    return (
        df.groupby("creator_id")["y_log"]
        .transform(lambda s: s.shift(1).rolling(k, min_periods=1).median())
    )
