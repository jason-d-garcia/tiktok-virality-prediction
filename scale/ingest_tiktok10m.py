"""
Stream TikTok-10M shards into the post table without loading 10M rows at once.

    python -m scale.ingest_tiktok10m --src /Volumes/<drive>/hf/tiktok10m \
        --out /Volumes/<drive>/tiktok/tiktok10m_posts.parquet

Reads only the columns the post table needs, one row group at a time, maps each
batch onto the shared schema (scale.posts.map_tiktok10m), then dedups across
shards keeping each post's latest reading. Writes the post table and a data card
(<out>.card.json) with row counts, duplicates, creator sizes, the view
distribution near the ~30K truncation point and post ages.

Pass the output to scale.build_posts --tiktok10m <out>; the loader recognises it.
Every shard's mapped file is kept under <out>.parts/ so a re-run skips finished shards.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scale.posts import TIKTOK10M_COLUMNS, TIKTOK10M_TRUNCATION_VIEWS, _finish, map_tiktok10m

OPTIONAL_COLUMNS = ["stats_time"]


def shard_files(src):
    src = Path(src)
    files = sorted(src.rglob("*.parquet")) if src.is_dir() else [src]
    if not files:
        raise FileNotFoundError(f"No parquet files under {src}")
    return files


def map_shard(path, views_at, batch_rows=250_000):
    """Map one shard batch by batch. Returns the mapped frame and its raw row count."""
    pf = pq.ParquetFile(path)
    present = set(pf.schema_arrow.names)
    missing = [c for c in TIKTOK10M_COLUMNS if c not in present]
    if missing:
        raise KeyError(f"tiktok10m {path.name}: missing columns {missing}; got {sorted(present)[:30]}")
    cols = TIKTOK10M_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in present]
    parts, n_raw = [], 0
    for batch in pf.iter_batches(batch_size=batch_rows, columns=cols):
        raw = batch.to_pandas()
        n_raw += len(raw)
        parts.append(map_tiktok10m(raw, views_at))
    return pd.concat(parts, ignore_index=True), n_raw


def dedup_latest(df):
    """A post seen in several shards or snapshots keeps its latest reading."""
    df = df.sort_values("collected_time", na_position="first", kind="stable")
    return df.drop_duplicates("post_id", keep="last")


def write_deduped(parts, out):
    """
    Cross-shard dedup on narrow columns only, then filter and write one part at a
    time, so peak memory is one shard rather than all ten. Returns the narrow
    columns of the written table, for the data card.
    """
    ids = []
    for i, part in enumerate(parts):
        d = pd.read_parquet(part, columns=["post_id", "collected_time"])
        d["part"], d["row"] = i, np.arange(len(d))
        ids.append(d)
    keep = dedup_latest(pd.concat(ids, ignore_index=True))

    narrow, writer = [], None
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    for i, part in enumerate(parts):
        rows = np.sort(keep.loc[keep["part"] == i, "row"].to_numpy())
        df = _finish(pd.read_parquet(part).iloc[rows])
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(tmp, table.schema)
        writer.write_table(table.cast(writer.schema))
        narrow.append(df[["creator_id", "create_time", "collected_time", "views"]])
    writer.close()
    tmp.replace(out)
    return pd.concat(narrow, ignore_index=True)


def data_card(posts, n_raw, n_mapped, files):
    views = posts["views"].to_numpy(float)
    age = (posts["collected_time"] - posts["create_time"]).dt.total_seconds() / 86400
    per_creator = posts.groupby("creator_id").size()
    log_v = np.log10(np.maximum(views, 1))
    edges = np.arange(3.0, 6.01, 0.25)
    hist, _ = np.histogram(log_v, bins=edges)
    return {
        "shards": [f.name for f in files],
        "raw_rows": int(n_raw),
        "mapped_rows": int(n_mapped),
        "unique_posts": int(len(posts)),
        "duplicate_rows": int(n_mapped - len(posts)),
        "creators": int(per_creator.size),
        "posts_per_creator": {q: float(per_creator.quantile(float(q))) for q in ("0.5", "0.9", "0.99")},
        "creators_with_5plus_posts": int((per_creator >= 5).sum()),
        "create_time_range": [str(posts["create_time"].min()), str(posts["create_time"].max())],
        "views_quantiles": {q: float(np.quantile(views, float(q))) for q in ("0.001", "0.01", "0.1", "0.5", "0.9")},
        "truncation_views": TIKTOK10M_TRUNCATION_VIEWS,
        "share_below_truncation": float(np.mean(views < TIKTOK10M_TRUNCATION_VIEWS)),
        # a sharp cut shows as near-empty bins left of 4.5 and a jump right of it
        "log10_views_hist": {f"{lo:.2f}": int(n) for lo, n in zip(edges[:-1], hist)},
        "age_days_quantiles": {q: float(age.quantile(float(q))) for q in ("0.01", "0.1", "0.5", "0.9", "0.99")},
        "age_unknown": float(age.isna().mean()),
        "age_under_7d": float((age < 7).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="directory of TikTok-10M parquet shards (or one shard)")
    parser.add_argument("--out", required=True, help="output parquet path")
    parser.add_argument("--views-at", default="stats_time", choices=["stats_time", "collected_time"])
    parser.add_argument("--max-shards", type=int, default=None, help="for a quick partial run")
    parser.add_argument("--batch-rows", type=int, default=250_000)
    args = parser.parse_args()

    files = shard_files(args.src)[:args.max_shards]
    out = Path(args.out)
    parts_dir = out.with_suffix(".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)

    n_raw = n_mapped = 0
    for f in files:
        part = parts_dir / f"{f.stem}.parquet"
        meta = parts_dir / f"{f.stem}.json"
        if part.exists() and meta.exists():
            done = json.loads(meta.read_text())
            n_raw, n_mapped = n_raw + done["raw_rows"], n_mapped + done["mapped_rows"]
            print(f"{f.name}: done earlier, skipping")
            continue
        mapped, n = map_shard(f, args.views_at, args.batch_rows)
        # dedup inside the shard first so the cross-shard pass stays small
        dedup_latest(mapped).to_parquet(part, index=False)
        meta.write_text(json.dumps({"raw_rows": n, "mapped_rows": len(mapped)}))
        n_raw, n_mapped = n_raw + n, n_mapped + len(mapped)
        print(f"{f.name}: {n:,} rows -> {len(mapped):,} mapped")

    parts = [parts_dir / f"{f.stem}.parquet" for f in files]
    posts = write_deduped(parts, out)
    card = data_card(posts, n_raw, n_mapped, files)
    card_path = out.with_suffix(".card.json")
    card_path.write_text(json.dumps(card, indent=2))
    print(f"Saved {len(posts):,} posts from {card['creators']:,} creators to {out}")
    print(f"Data card: {card_path}  ({card['share_below_truncation']:.2%} below the 10^4.5 truncation point)")


if __name__ == "__main__":
    main()
