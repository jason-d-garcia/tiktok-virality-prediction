"""
One post table for every data source.

Each loader returns a DataFrame with these columns (missing values are NaN/None):

    post_id         str    unique within source
    source          str    "emily", "tiktok10m", "einzz", "smp"
    creator_id      str
    create_time     datetime64[ns, UTC]
    collected_time  datetime64[ns, UTC]   when views were read (NaT if unknown)
    views           float  cumulative views at collected_time
    followers       float  creator followers at collection (post-publication!)
    duration_s      float
    caption         str
    video_url       str    page URL used by fetch_clips.py

Field names for the Hugging Face sets come from their dataset cards. Loaders
fail loudly on a missing column instead of guessing.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing.preprocessing import parse_metric, parse_seconds

COLUMNS = [
    "post_id", "source", "creator_id", "create_time", "collected_time",
    "views", "followers", "duration_s", "caption", "video_url",
]


def _require(df, cols, source):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{source}: missing columns {missing}; got {list(df.columns)[:30]}")


def _finish(df):
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[COLUMNS].copy()
    df["post_id"] = df["post_id"].astype(str)
    df["creator_id"] = df["creator_id"].astype(str)
    df["caption"] = df["caption"].fillna("").astype(str)
    df = df.dropna(subset=["views", "create_time"])
    df = df[df["views"] >= 0]
    return df.drop_duplicates(subset=["source", "post_id"]).reset_index(drop=True)


def _read_any(path):
    path = Path(path)
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No parquet files under {path}")
    if files[0].suffix == ".parquet":
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if files[0].suffix in (".json", ".jsonl"):
        return pd.read_json(files[0], lines=files[0].suffix == ".jsonl")
    return pd.read_csv(files[0])

# =========================
# ===== SOURCES ==========
# =========================

def load_emily(csv_path, collected_time=None):
    """Emily's hand-collected Creator Studio export. Collection date was not recorded."""
    raw = pd.read_csv(csv_path)
    df = pd.DataFrame({
        "post_id": raw["url"].str.rstrip("/").str.split("/").str[-1],
        "source": "emily",
        "creator_id": "emily",
        "create_time": pd.to_datetime(
            raw["published date"] + " " + raw["published time"], format="mixed"
        ).dt.tz_localize("UTC"),
        "collected_time": pd.Timestamp(collected_time, tz="UTC") if collected_time else pd.NaT,
        "views": raw["views"].apply(parse_metric),
        "duration_s": raw["video length"].apply(parse_seconds),
        "caption": raw["description"],
        "video_url": raw["url"],
    })
    return _finish(df)


def load_tiktok10m(path):
    """The-data-company/TikTok-10M parquet shards."""
    raw = _read_any(path)
    _require(raw, ["id", "user_id", "create_time", "collected_time", "play_count", "duration", "desc", "url"], "tiktok10m")
    df = pd.DataFrame({
        "post_id": raw["id"],
        "source": "tiktok10m",
        "creator_id": raw["user_id"],
        "create_time": pd.to_datetime(raw["create_time"], unit="s", utc=True),
        "collected_time": pd.to_datetime(raw["collected_time"], utc=True, errors="coerce"),
        "views": pd.to_numeric(raw["play_count"], errors="coerce"),
        "duration_s": pd.to_numeric(raw["duration"], errors="coerce"),
        "caption": raw["desc"],
        "video_url": raw["url"],
    })
    return _finish(df)


def load_einzz(path):
    """EinzzCookie/tiktok-videos-users-info. Nested author/statistics/video structs."""
    raw = _read_any(path)
    _require(raw, ["aweme_id", "author", "create_time", "statistics", "desc", "video"], "einzz")
    author = pd.json_normalize(raw["author"].tolist())
    stats = pd.json_normalize(raw["statistics"].tolist())
    video = pd.json_normalize(raw["video"].tolist())
    _require(stats, ["play_count"], "einzz.statistics")
    creator = raw["author_user_id"] if "author_user_id" in raw else author.get("uid")
    unique_id = author.get("unique_id", pd.Series([None] * len(raw)))
    duration = video.get("duration", pd.Series([np.nan] * len(raw)))
    df = pd.DataFrame({
        "post_id": raw["aweme_id"],
        "source": "einzz",
        "creator_id": creator,
        "create_time": pd.to_datetime(raw["create_time"], unit="s", utc=True),
        "collected_time": pd.NaT,
        "views": pd.to_numeric(stats["play_count"], errors="coerce"),
        "followers": pd.to_numeric(author.get("follower_count"), errors="coerce"),
        # video.duration is in milliseconds on TikTok's API
        "duration_s": pd.to_numeric(duration, errors="coerce") / 1000.0,
        "caption": raw["desc"],
        "video_url": "https://www.tiktok.com/@" + unique_id.astype(str) + "/video/" + raw["aweme_id"].astype(str),
    })
    return _finish(df)


LOADERS = {"emily": load_emily, "tiktok10m": load_tiktok10m, "einzz": load_einzz}


def sample_creators(df, n_posts, min_posts_per_creator=5, seed=42):
    """
    Sample whole creators (not posts) until about n_posts are kept, so creator
    baselines and creator-disjoint splits stay meaningful.
    """
    counts = df.groupby("creator_id").size()
    eligible = counts[counts >= min_posts_per_creator].sample(frac=1.0, random_state=seed)
    keep = eligible.index[eligible.cumsum() <= n_posts]
    return df[df["creator_id"].isin(keep)].reset_index(drop=True)
