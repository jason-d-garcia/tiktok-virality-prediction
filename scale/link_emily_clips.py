"""
Link Emily's existing downloads (data/raw/videos/0001.mp4, ...) to the clip names
scale/embed.py expects (emily_<video id>.mp4), without re-downloading.

util/download_videos.py numbered videos in order of reversed(video_urls.txt)[:140],
which matches the row order of tiktok_stats.csv.

    python -m scale.link_emily_clips --clips-dir data/scale/clips
"""

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URL_FILE = PROJECT_ROOT / "data" / "raw" / "video_urls.txt"
VIDEO_DIR = PROJECT_ROOT / "data" / "raw" / "videos"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips-dir", required=True)
    parser.add_argument("--n", type=int, default=140)
    args = parser.parse_args()

    out = Path(args.clips_dir)
    out.mkdir(parents=True, exist_ok=True)
    urls = [line.strip() for line in URL_FILE.read_text().splitlines() if line.strip()]
    linked = 0
    for i, url in enumerate(list(reversed(urls))[: args.n], start=1):
        src = VIDEO_DIR / f"{i:04d}.mp4"
        dst = out / f"emily_{url.rstrip('/').split('/')[-1]}.mp4"
        if src.exists() and not dst.exists():
            os.symlink(src.resolve(), dst)
            linked += 1
    print(f"Linked {linked} clips into {out}")


if __name__ == "__main__":
    main()
