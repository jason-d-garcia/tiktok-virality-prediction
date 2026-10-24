"""
Tabular features for the post table, built only from what is known at publish time
(plus post age, which is a control for cumulative views, not a predictor).

Creator history uses the *observed* views of a creator's earlier posts. Those were
read at collection time, not at publish time, so they are an optimistic stand-in
for what the creator knew. To limit that, a post's history covers only posts
published at least HISTORY_LAG_DAYS earlier, whose counts had mostly settled by
then. The creator baseline uses the same lag, so models and baseline see the same
history.

Creators without eligible history are left NaN here. Fill them with
impute_history() after the split, using training rows only.
"""

import numpy as np
import pandas as pd

HASHTAG_PATTERN = r"#\w+"

TABULAR_FEATURES = [
    "log_duration", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "caption_len", "hashtag_count",
    "log_age_days", "age_known",
    "creator_prior_posts", "creator_prior_median", "creator_prior_mean",
    "creator_prior_std", "creator_has_history", "log_days_since_last",
]
FOLLOWER_FEATURES = ["log_followers", "followers_known"]
HISTORY_LAG_DAYS = 7
MIN_AGE_DAYS = 7


def post_age_days(df):
    return (df["collected_time"] - df["create_time"]).dt.total_seconds() / 86400


def filter_min_age(df, min_age_days=MIN_AGE_DAYS):
    """Drop posts younger than min_age_days when views were read. Unknown ages are kept."""
    age = post_age_days(df)
    return df[~(age < min_age_days)].reset_index(drop=True)


def _eligible_counts(df, lag_days):
    """
    For each row (sorted by creator, create_time), the number of the creator's
    posts published at least lag_days before it.
    """
    lag = pd.Timedelta(days=lag_days)
    counts = np.zeros(len(df), dtype=np.int64)
    times = df["create_time"].dt.tz_convert(None).to_numpy()
    for _, idx in df.groupby("creator_id", sort=False).indices.items():
        t = times[idx]
        counts[idx] = np.searchsorted(t, t - lag.to_timedelta64(), side="right")
    return counts


def _lagged(df, stat, lag_days):
    """stat(series) gives an expanding/rolling series per creator; read it at the last eligible post."""
    n = _eligible_counts(df, lag_days)
    out = np.full(len(df), np.nan)
    for _, idx in df.groupby("creator_id", sort=False).indices.items():
        vals = stat(df["y_log"].iloc[idx].reset_index(drop=True)).to_numpy()
        k = n[idx]
        has = k > 0
        out[idx[has]] = vals[k[has] - 1]
    return out


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

    age_days = post_age_days(df)
    df["age_known"] = age_days.notna().astype(float)
    df["log_age_days"] = np.log1p(age_days.clip(lower=0)).fillna(0.0)

    df["creator_prior_posts"] = np.log1p(_eligible_counts(df, HISTORY_LAG_DAYS))
    df["creator_prior_median"] = _lagged(df, lambda s: s.expanding().median(), HISTORY_LAG_DAYS)
    df["creator_prior_mean"] = _lagged(df, lambda s: s.expanding().mean(), HISTORY_LAG_DAYS)
    df["creator_prior_std"] = _lagged(df, lambda s: s.expanding().std(), HISTORY_LAG_DAYS)
    df["creator_has_history"] = df["creator_prior_median"].notna().astype(float)

    gap = df.groupby("creator_id")["create_time"].diff().dt.total_seconds() / 86400
    df["log_days_since_last"] = np.log1p(gap.clip(lower=0)).fillna(0.0)

    df["followers_known"] = df["followers"].notna().astype(float)
    df["log_followers"] = np.log1p(df["followers"]).fillna(0.0)
    return df


def impute_history(df, fill_value):
    """Fill creators with no eligible history. fill_value should come from training rows only."""
    df = df.copy()
    for col in ("creator_prior_median", "creator_prior_mean"):
        df[col] = df[col].fillna(fill_value)
    df["creator_prior_std"] = df["creator_prior_std"].fillna(0.0)
    return df


def creator_baseline(df, k=10, lag_days=HISTORY_LAG_DAYS):
    """
    Median log views of each creator's last k posts published at least lag_days
    earlier (NaN when there are none). Expects add_features' sort order.
    """
    return pd.Series(_lagged(df, lambda s: s.rolling(k, min_periods=1).median(), lag_days), index=df.index)
