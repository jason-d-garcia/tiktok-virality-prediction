"""
Unattended fetch-and-embed run: pull clips in creator-sampled batches, embed each
batch while the next one downloads, delete clips once embedded, and stop cleanly
when TikTok starts blocking or the disk fills.

    caffeinate -i python -m scale.weekend_run --posts data/scale/posts.parquet \
        --root /Volumes/<drive>/tiktok --target 100000

Order of work:
  1. caption embeddings for every post (cheap, no video)        -> <root>/emb/text
  2. pilot: 200 posts per source; per-source success rate goes to the log
  3. batches of --batch posts until --target clips are embedded  -> <root>/emb/av

Batches follow a fixed queue of whole creators (Emily, then SMP, EinzzCookie, then
TikTok-10M), so any prefix of the run is a creator-sampled set and the learning
curve can be read off at 5K, 10K, 20K, ... Re-running the same command resumes.
Progress goes to <root>/run_log.jsonl, one line per batch.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scale.embed import existing_ids

SOURCE_ORDER = ["emily", "smp", "einzz", "tiktok10m"]


def build_queue(posts, seed=0):
    """Posts ordered source by source, whole creators at a time, creators shuffled."""
    rng = np.random.default_rng(seed)
    parts = []
    for source in SOURCE_ORDER + sorted(set(posts["source"]) - set(SOURCE_ORDER)):
        sub = posts[posts["source"] == source]
        creators = sub["creator_id"].unique()
        rank = dict(zip(rng.permutation(creators), range(len(creators))))
        parts.append(sub.assign(_rank=sub["creator_id"].map(rank)).sort_values(["_rank", "create_time"]))
    queue = pd.concat(parts, ignore_index=True).drop(columns="_rank")
    return queue[["post_id", "source", "creator_id", "video_url", "caption"]]


def attempted(clips_dir):
    manifest = clips_dir / "manifest.csv"
    if not manifest.exists():
        return pd.DataFrame(columns=["post_id", "source", "status"])
    return pd.read_csv(manifest, dtype={"post_id": str})


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def log(root, **row):
    row["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    print(json.dumps(row), flush=True)
    with open(root / "run_log.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def run(*args):
    return subprocess.run([sys.executable, "-m", *args], check=True)


def fetch(batch_path, clips_dir, args):
    cmd = ["scale.fetch_clips", "--posts", str(batch_path), "--out-dir", str(clips_dir),
           "--seconds", str(args.seconds), "--workers", str(args.workers), "--sleep", str(args.sleep)]
    if args.cookies:
        cmd += ["--cookies", args.cookies]
    run(*cmd)


def start_embed(batch_path, clips_dir, emb_dir, args):
    cmd = [sys.executable, "-m", "scale.embed", "--posts", str(batch_path), "--clips-dir", str(clips_dir),
           "--out-dir", str(emb_dir), "--modalities", "video,audio", "--seconds", str(args.seconds),
           "--delete-clips"]
    return subprocess.Popen(cmd)


def success_rate(manifest, ids):
    rows = manifest[manifest["post_id"].isin(ids)]
    return float((rows["status"] == "ok").mean()) if len(rows) else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posts", required=True)
    parser.add_argument("--root", required=True, help="working folder, e.g. on the external drive")
    parser.add_argument("--target", type=int, default=100_000, help="stop after this many clips are embedded")
    parser.add_argument("--batch", type=int, default=2000)
    parser.add_argument("--pilot-per-source", type=int, default=200)
    parser.add_argument("--seconds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--cookies", default=None)
    parser.add_argument("--min-success", type=float, default=0.3,
                        help="a batch below this looks like blocking: back off, then stop if it repeats")
    parser.add_argument("--backoff-min", type=float, default=45)
    parser.add_argument("--min-free-gb", type=float, default=5)
    parser.add_argument("--skip-text", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    clips_dir, batch_dir = root / "clips", root / "batches"
    text_dir, av_dir = root / "emb" / "text", root / "emb" / "av"
    for d in (clips_dir, batch_dir, text_dir, av_dir):
        d.mkdir(parents=True, exist_ok=True)

    posts = pd.read_parquet(args.posts)
    posts["post_id"] = posts["post_id"].astype(str)
    queue = build_queue(posts.dropna(subset=["video_url"]))
    queue = queue[queue["video_url"].astype(str).str.startswith("http")].reset_index(drop=True)

    if not args.skip_text:
        run("scale.embed", "--posts", args.posts, "--out-dir", str(text_dir), "--modalities", "text")

    # pilot: a slice of each source, so a failing source shows up before the long run
    pilot_path = batch_dir / "pilot.parquet"
    pilot = queue.groupby("source", sort=False).head(args.pilot_per_source)
    pilot.to_parquet(pilot_path, index=False)
    fetch(pilot_path, clips_dir, args)
    manifest = attempted(clips_dir)
    rates = {s: success_rate(manifest, g["post_id"]) for s, g in pilot.groupby("source")}
    log(root, stage="pilot", success_by_source=rates)
    weak = [s for s, r in rates.items() if r < args.min_success]
    if len(weak) == len(rates):
        log(root, stage="done", reason=f"every source failed the pilot (< {args.min_success:.0%})")
        return
    if weak:
        log(root, stage="pilot", dropped_sources=weak)
        queue = queue[~queue["source"].isin(weak)]
    embedder = start_embed(pilot_path, clips_dir, av_dir, args)

    strikes, n = 0, 0
    while True:
        manifest = attempted(clips_dir)
        n_embedded = len(existing_ids(av_dir))
        todo = queue[~queue["post_id"].isin(set(manifest["post_id"]))]
        if n_embedded >= args.target or todo.empty:
            reason = "target reached" if n_embedded >= args.target else "queue empty"
            break
        if free_gb(root) < args.min_free_gb:
            reason = f"less than {args.min_free_gb} GB free"
            break

        n += 1
        batch = todo.head(args.batch)
        batch_path = batch_dir / f"batch_{time.strftime('%Y%m%d_%H%M%S')}_{n}.parquet"
        batch.to_parquet(batch_path, index=False)
        t0 = time.time()
        fetch(batch_path, clips_dir, args)
        rate = success_rate(attempted(clips_dir), batch["post_id"])

        # embed this batch once the previous embed finishes; the next fetch overlaps it
        embedder.wait()
        embedder = start_embed(batch_path, clips_dir, av_dir, args)
        log(root, stage="batch", batch=n, sources=batch["source"].value_counts().to_dict(),
            fetched_ok=rate, fetch_minutes=round((time.time() - t0) / 60, 1),
            embedded_so_far=n_embedded, free_gb=round(free_gb(root), 1))

        if rate < args.min_success:
            strikes += 1
            if strikes >= 2:
                reason = f"two low-success batches in a row ({rate:.0%}); probably blocked"
                break
            log(root, stage="backoff", minutes=args.backoff_min)
            time.sleep(args.backoff_min * 60)
        else:
            strikes = 0

    embedder.wait()
    log(root, stage="done", reason=reason, embedded=len(existing_ids(av_dir)),
        attempted=len(attempted(clips_dir)))


if __name__ == "__main__":
    main()
