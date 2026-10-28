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

EinzzCookie is a test set only (6,258 unique posts, median 1 post per creator),
so it never enters pretraining; see TEST_ONLY_SOURCES.
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
    # an all-NaT collected_time (date unknown) is tz-naive; age arithmetic needs UTC
    for col in ("create_time", "collected_time"):
        df[col] = pd.to_datetime(df[col], utc=True)
    df = df.dropna(subset=["views", "create_time"])
    df = df[df["views"] >= 0]
    return df.drop_duplicates(subset=["source", "post_id"]).reset_index(drop=True)


def _to_utc(col):
    """Unix seconds or date strings to UTC timestamps."""
    if pd.api.types.is_numeric_dtype(col):
        return pd.to_datetime(col, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(col, utc=True, errors="coerce", format="mixed")


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


def load_tiktok10m(path, views_at="stats_time"):
    """
    The-data-company/TikTok-10M parquet shards.

    `play_count` was most likely read at `stats_time`, which is up to ~98 days after
    `collected_time` (memo section 4, to verify). `views_at` picks the timestamp that
    post age is measured to; it falls back to `collected_time` if the column is absent.
    A post seen more than once keeps its latest reading.
    """
    raw = _read_any(path)
    _require(raw, ["id", "user_id", "create_time", "collected_time", "play_count", "duration", "desc", "url"], "tiktok10m")
    if views_at not in raw.columns:
        print(f"tiktok10m: no {views_at} column, measuring age to collected_time")
        views_at = "collected_time"
    df = pd.DataFrame({
        "post_id": raw["id"],
        "source": "tiktok10m",
        "creator_id": raw["user_id"],
        "create_time": _to_utc(raw["create_time"]),
        "collected_time": _to_utc(raw[views_at]),
        "views": pd.to_numeric(raw["play_count"], errors="coerce"),
        "duration_s": pd.to_numeric(raw["duration"], errors="coerce"),
        "caption": raw["desc"],
        "video_url": raw["url"],
    })
    df = df.sort_values("collected_time", na_position="first").drop_duplicates("post_id", keep="last")
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
    # each post is repeated ~150 times with identical counts
    out = _finish(df)
    print(f"einzz: {len(raw):,} rows -> {len(out):,} unique posts")
    assert out["post_id"].is_unique
    return out


# SMP-Video label: s = log2(views / days) + 1, where days is post age at crawl.
def smp_label_to_views(label, days):
    return np.asarray(days, dtype=float) * np.exp2(np.asarray(label, dtype=float) - 1)


def views_to_smp_label(views, days):
    return np.log2(np.asarray(views, dtype=float) / np.asarray(days, dtype=float)) + 1


# File stems in the smpchallenge/SMP-Video release (train split; test has no labels).
SMP_FILES = {
    "posts": "SMP-Video_anonymized_posts_train",
    "labels": "SMP-Video_anonymized_popularity_train",
    "users": "SMP-Video_anonymized_users_train",
    "videos": "SMP-Video_anonymized_videos_train",
}


def _smp_table(path, stem):
    files = [f for ext in ("parquet", "jsonl", "csv") for f in sorted(Path(path).rglob(f"{stem}*.{ext}"))]
    if not files:
        raise FileNotFoundError(f"smp: no {stem}*.parquet, .jsonl or .csv under {path}")
    return _read_any(files[0])


def load_smp(path, collected_time=None):
    """
    smpchallenge/SMP-Video (gated; labeled train split only).

    Checked against the release: posts (pid, uid, post_time, post_content),
    popularity (pid, popularity), users (uid, user_follower_count) and videos
    (pid, video_duration in seconds). Raw views are not released; they are
    rebuilt from the label as days * 2**(label - 1). The release carries no
    crawl timestamp, so pass the crawl date as `collected_time`; without it the
    loader refuses to guess.
    """
    posts = _smp_table(path, SMP_FILES["posts"])
    labels = _smp_table(path, SMP_FILES["labels"])
    _require(posts, ["pid", "uid", "post_time", "post_content"], "smp.posts")
    _require(labels, ["pid", "popularity"], "smp.labels")
    raw = posts.merge(labels[["pid", "popularity"]], on="pid", how="inner")
    try:
        users = _smp_table(path, SMP_FILES["users"])
        _require(users, ["uid", "user_follower_count"], "smp.users")
        raw = raw.merge(users[["uid", "user_follower_count"]].drop_duplicates("uid"), on="uid", how="left")
    except FileNotFoundError:
        raw["user_follower_count"] = np.nan
    try:
        videos = _smp_table(path, SMP_FILES["videos"])
        _require(videos, ["pid", "video_duration"], "smp.videos")
        raw = raw.merge(videos[["pid", "video_duration"]].drop_duplicates("pid"), on="pid", how="left")
    except FileNotFoundError:
        pass

    create = _to_utc(raw["post_time"])
    crawl_col = next((c for c in ("crawl_time", "collected_time", "stats_time") if c in raw.columns), None)
    if crawl_col:
        collected = _to_utc(raw[crawl_col])
    elif collected_time:
        collected = pd.Series(pd.Timestamp(collected_time, tz="UTC"), index=raw.index)
    else:
        raise ValueError("smp: no crawl-time column; pass the crawl date (--smp-collected)")
    days = (collected - create).dt.total_seconds() / 86400

    duration = raw["video_duration"] if "video_duration" in raw else pd.Series(np.nan, index=raw.index)
    url = raw["video_url"] if "video_url" in raw else pd.Series(None, index=raw.index)
    df = pd.DataFrame({
        "post_id": raw["pid"],
        "source": "smp",
        "creator_id": raw["uid"],
        "create_time": create,
        "collected_time": collected,
        "views": np.round(smp_label_to_views(raw["popularity"], days.where(days > 0))),
        "followers": pd.to_numeric(raw["user_follower_count"], errors="coerce"),
        "duration_s": pd.to_numeric(duration, errors="coerce"),
        "caption": raw["post_content"],
        "video_url": url,
    })
    return _finish(df)


LOADERS = {"emily": load_emily, "tiktok10m": load_tiktok10m, "einzz": load_einzz, "smp": load_smp}

# Evaluated on, never pretrained on.
TEST_ONLY_SOURCES = {"einzz"}


def sample_creators(df, n_posts, min_posts_per_creator=5, seed=42):
    """
    Sample whole creators (not posts) until about n_posts are kept, so creator
    baselines and creator-disjoint splits stay meaningful.
    """
    counts = df.groupby("creator_id").size()
    eligible = counts[counts >= min_posts_per_creator].sample(frac=1.0, random_state=seed)
    keep = eligible.index[eligible.cumsum() <= n_posts]
    return df[df["creator_id"].isin(keep)].reset_index(drop=True)
