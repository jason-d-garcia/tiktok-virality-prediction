"""
Download the first few seconds of each post's video with yt-dlp.

Resumable: every attempt is appended to <out_dir>/manifest.csv, and posts already
marked ok or failed are skipped on the next run. Only the opening seconds are kept,
which is all the embedding step reads.

    python -m scale.fetch_clips --posts data/scale/posts.parquet --out-dir data/scale/clips \
        --seconds 5 --workers 2 --limit 20000
"""

import argparse
import csv
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd


def load_done(manifest):
    if not manifest.exists():
        return set()
    done = pd.read_csv(manifest)
    return set(done["post_id"].astype(str))


def fetch_one(row, out_dir, seconds, cookies, sleep_s):
    out_path = out_dir / f"{row.source}_{row.post_id}.mp4"
    command = [
        "yt-dlp",
        "--quiet", "--no-warnings", "--no-playlist",
        "-f", "worst[ext=mp4]/worst",
        "--download-sections", f"*0-{seconds}",
        "--force-keyframes-at-cuts",
        "-o", str(out_path),
        row.video_url,
    ]
    if cookies:
        command[1:1] = ["--cookies", str(cookies)]

    try:
        subprocess.run(command, check=True, timeout=120, capture_output=True)
        status = "ok" if out_path.exists() else "missing"
    except subprocess.TimeoutExpired:
        status = "timeout"
    except subprocess.CalledProcessError as e:
        lines = (e.stderr or b"").decode(errors="ignore").strip().splitlines()
        status = "failed:" + lines[-1][:120] if lines else "failed"
    time.sleep(sleep_s)
    return row.post_id, row.source, status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posts", required=True, help="parquet from scale.build_posts")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seconds", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=1.0, help="pause per download, per worker")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cookies", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.csv"

    posts = pd.read_parquet(args.posts, columns=["post_id", "source", "video_url"])
    posts = posts.dropna(subset=["video_url"])
    done = load_done(manifest)
    todo = posts[~posts["post_id"].astype(str).isin(done)]
    if args.limit:
        todo = todo.head(max(0, args.limit - len(done)))
    print(f"{len(done)} already attempted, {len(todo)} to fetch")

    lock = threading.Lock()
    new_file = not manifest.exists()
    n_ok = 0
    with open(manifest, "a", newline="") as f, ThreadPoolExecutor(args.workers) as pool:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["post_id", "source", "status"])
        futures = [
            pool.submit(fetch_one, row, out_dir, args.seconds, args.cookies, args.sleep)
            for row in todo.itertuples(index=False)
        ]
        for i, fut in enumerate(futures, 1):
            post_id, source, status = fut.result()
            n_ok += status == "ok"
            with lock:
                writer.writerow([post_id, source, status])
                f.flush()
            if i % 100 == 0:
                print(f"{i}/{len(futures)} attempted, {n_ok} ok")

    print(f"Done: {n_ok}/{len(todo)} ok")


if __name__ == "__main__":
    main()
