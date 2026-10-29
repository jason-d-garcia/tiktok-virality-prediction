"""
Synthetic shards in the TikTok-10M layout, for dry runs before the real download.

    python -m tests.synthetic_tiktok10m --out data/synthetic/tiktok10m --shards 3 --creators 400

Same columns as the release (plus a few the pipeline ignores), unix-second
timestamps, stats_time after collected_time, views centred near the release's
median (~4K) with no truncation (the full release shows none), and ~1% of posts
repeated in a later shard with a newer reading.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START = pd.Timestamp("2023-01-01", tz="UTC").timestamp()
SPAN_DAYS = 540


def make_posts(n_creators=400, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_creators):
        level = rng.normal(3.6, 0.8)            # creator's typical log10 views
        n = max(1, int(rng.lognormal(2.2, 0.9)))
        first = rng.uniform(0, SPAN_DAYS * 0.8)
        days = np.sort(first + rng.uniform(0, SPAN_DAYS - first, n))
        for d in days:
            dur = float(rng.integers(5, 180))
            log_v = level + 0.25 * np.log10(dur) + rng.normal(0, 0.6)
            caption = f"post {c}-{d:.0f} " + " ".join(f"#tag{rng.integers(0, 30)}" for _ in range(rng.integers(0, 5)))
            rows.append((f"7{c:05d}{int(d * 1000):08d}", f"u{c}", d, dur, 10 ** log_v, caption))
    df = pd.DataFrame(rows, columns=["id", "user_id", "day", "duration", "views", "desc"])
    df["create_time"] = (START + df["day"] * 86400).astype("int64")
    collect = START + (SPAN_DAYS + rng.uniform(0, 5, len(df))) * 86400
    df["collected_time"] = collect.astype("int64")
    df["stats_time"] = (collect + rng.uniform(0, 98, len(df)) * 86400).astype("int64")
    df["play_count"] = np.floor(df["views"]).astype("int64")
    df["digg_count"] = (df["play_count"] * 0.08).astype("int64")
    df["url"] = "https://www.tiktok.com/@" + df["user_id"] + "/video/" + df["id"]
    return df.drop(columns=["day", "views"]).reset_index(drop=True)


def write_shards(out, n_shards=3, n_creators=400, seed=0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = make_posts(n_creators, seed)
    rng = np.random.default_rng(seed + 1)
    shard = rng.integers(0, n_shards, len(df))
    # ~1% re-seen in a later shard with a newer, larger reading
    again = df.sample(frac=0.01, random_state=seed).copy()
    again["stats_time"] += 86400 * 10
    again["play_count"] = (again["play_count"] * 1.1).astype("int64")
    paths = []
    for s in range(n_shards):
        part = df[shard == s]
        if s == n_shards - 1:
            part = pd.concat([part, again])
        p = out / f"train-{s:05d}-of-{n_shards:05d}.parquet"
        part.to_parquet(p, index=False)
        paths.append(p)
    return paths, df, again


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--creators", type=int, default=400)
    args = parser.parse_args()
    paths, df, again = write_shards(args.out, args.shards, args.creators)
    print(f"Wrote {len(df) + len(again):,} rows ({len(df):,} unique posts) to {len(paths)} shards in {args.out}")


if __name__ == "__main__":
    main()
