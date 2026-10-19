"""
Load raw sources into one post table with features.

    python -m scale.build_posts --emily data/raw/tiktok_stats.csv \
        --tiktok10m data/hf/tiktok10m --einzz data/hf/einzz \
        --max-posts-per-source 200000 --out data/scale/posts.parquet
"""

import argparse
from pathlib import Path

import pandas as pd

from scale.features import add_features
from scale.posts import LOADERS, sample_creators


def main():
    parser = argparse.ArgumentParser()
    for name in LOADERS:
        parser.add_argument(f"--{name}", default=None, help=f"path to {name} data")
    parser.add_argument("--emily-collected", default=None, help="date Emily's stats were read, if known")
    parser.add_argument("--max-posts-per-source", type=int, default=None)
    parser.add_argument("--min-posts-per-creator", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    frames = []
    for name, loader in LOADERS.items():
        path = getattr(args, name)
        if not path:
            continue
        df = loader(path, args.emily_collected) if name == "emily" else loader(path)
        if name != "emily" and args.max_posts_per_source:
            df = sample_creators(df, args.max_posts_per_source, args.min_posts_per_creator)
        print(f"{name}: {len(df):,} posts, {df['creator_id'].nunique():,} creators")
        frames.append(df)

    posts = add_features(pd.concat(frames, ignore_index=True))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    posts.to_parquet(args.out, index=False)
    print(f"Saved {len(posts):,} posts to {args.out}")


if __name__ == "__main__":
    main()
