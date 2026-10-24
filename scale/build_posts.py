"""
Load raw sources into one post table with features.

    python -m scale.build_posts --emily data/raw/tiktok_stats.csv \
        --tiktok10m data/hf/tiktok10m --einzz data/hf/einzz \
        --smp data/hf/smp --max-posts-per-source 200000 --out data/scale/posts.parquet

Posts younger than --min-age-days when their views were read are dropped (posts
of unknown age, like Emily's, are kept). EinzzCookie is never creator-sampled:
it is a test set with ~1 post per creator.
"""

import argparse
from pathlib import Path

import pandas as pd

from scale.features import MIN_AGE_DAYS, add_features, filter_min_age
from scale.posts import LOADERS, TEST_ONLY_SOURCES, sample_creators


def main():
    parser = argparse.ArgumentParser()
    for name in LOADERS:
        parser.add_argument(f"--{name}", default=None, help=f"path to {name} data")
    parser.add_argument("--emily-collected", default=None, help="date Emily's stats were read, if known")
    parser.add_argument("--smp-collected", default=None, help="SMP crawl date, if the files carry no crawl time")
    parser.add_argument("--tiktok10m-views-at", default="stats_time", choices=["stats_time", "collected_time"],
                        help="timestamp TikTok-10M play_count was read at")
    parser.add_argument("--min-age-days", type=float, default=MIN_AGE_DAYS)
    parser.add_argument("--max-posts-per-source", type=int, default=None)
    parser.add_argument("--min-posts-per-creator", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    frames = []
    for name, loader in LOADERS.items():
        path = getattr(args, name)
        if not path:
            continue
        if name == "emily":
            df = loader(path, args.emily_collected)
        elif name == "smp":
            df = loader(path, args.smp_collected)
        elif name == "tiktok10m":
            df = loader(path, views_at=args.tiktok10m_views_at)
        else:
            df = loader(path)
        n_loaded = len(df)
        df = filter_min_age(df, args.min_age_days)
        if n_loaded > len(df):
            print(f"{name}: dropped {n_loaded - len(df):,} posts younger than {args.min_age_days:g} days")
        if name != "emily" and name not in TEST_ONLY_SOURCES and args.max_posts_per_source:
            df = sample_creators(df, args.max_posts_per_source, args.min_posts_per_creator)
        print(f"{name}: {len(df):,} posts, {df['creator_id'].nunique():,} creators")
        frames.append(df)

    posts = add_features(pd.concat(frames, ignore_index=True))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    posts.to_parquet(args.out, index=False)
    print(f"Saved {len(posts):,} posts to {args.out}")


if __name__ == "__main__":
    main()
