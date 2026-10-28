"""
End-to-end smoke test of the scale/ pipeline on synthetic posts.

Synthetic views have a creator effect, a real signal in the video and text
embeddings, and a heavy right tail, so the multimodal model should beat
metadata-only and every head should run. Embedding extraction itself (CLIP,
Whisper, yt-dlp) is not exercised here.

    python -m tests.smoke_scale
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from scale.embed import write_shard
from scale.features import add_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def synthetic_posts(n_creators=200, posts_per_creator=40, seed=0):
    rng = np.random.default_rng(seed)
    w_video, w_text = rng.normal(size=512), rng.normal(size=512)
    rows, video, text = [], [], []
    creators = [f"c{i}" for i in range(n_creators)] + ["emily"]
    for c in creators:
        n = 150 if c == "emily" else posts_per_creator
        level = 12.5 if c == "emily" else rng.normal(10.5, 1.5)
        start = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=int(rng.integers(0, 200)))
        for j in range(n):
            v = rng.normal(size=512)
            t = rng.normal(size=512)
            signal = 0.04 * v @ w_video + 0.03 * t @ w_text
            tail = rng.normal(2.5, 1.0) if rng.random() < 0.08 else 0.0
            y = level + signal + rng.normal(0, 0.6) + tail
            created = start + pd.Timedelta(days=2 * j, hours=int(rng.integers(0, 24)))
            rows.append({
                "post_id": f"{c}_{j}", "source": "emily" if c == "emily" else "synthetic",
                "creator_id": c, "create_time": created,
                "collected_time": pd.Timestamp("2025-06-01", tz="UTC"),
                "views": float(np.floor(np.expm1(y))), "followers": np.nan,
                "duration_s": float(rng.uniform(5, 90)), "caption": "a post #fyp" * int(rng.integers(1, 3)),
                "video_url": "",
            })
            video.append(v / np.linalg.norm(v))
            text.append(t / np.linalg.norm(t))
    df = pd.DataFrame(rows)
    # drop video for 40% of posts, as most public posts won't have clips
    has_video = rng.random(len(df)) > 0.4
    return df, np.array(video, dtype=np.float32), np.array(text, dtype=np.float32), has_video


def run(results, *args):
    cmd = [sys.executable, "-m", *args]
    print("$", " ".join(cmd[2:]))
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env={**os.environ, "SCALE_RESULTS": str(results)})


def main():
    tmp = Path(tempfile.mkdtemp())
    df, video, text, has_video = synthetic_posts()
    keys = (df["source"] + "_" + df["post_id"]).tolist()
    (tmp / "emb_text").mkdir()
    (tmp / "emb_av").mkdir()
    write_shard(tmp / "emb_text", keys, {"text": list(text)})
    av_keys = [k for k, h in zip(keys, has_video) if h]
    write_shard(tmp / "emb_av", av_keys, {"video": list(video[has_video])})

    posts = add_features(df)
    posts.to_parquet(tmp / "posts.parquet", index=False)

    results = tmp / "results.csv"
    emb_args = ["--emb-dirs", str(tmp / "emb_text"), str(tmp / "emb_av")]

    run(results, "scale.train", "--posts", str(tmp / "posts.parquet"), "--head", "gaussian",
        "--epochs", "15", "--out", str(tmp / "meta.pt"))
    for head in ("gaussian", "mdn", "quantile", "negbin"):
        run(results, "scale.train", "--posts", str(tmp / "posts.parquet"), *emb_args, "--modalities", "video,text",
            "--head", head, "--epochs", "15", "--out", str(tmp / f"{head}.pt"))
    run(results, "scale.finetune", "--posts", str(tmp / "posts.parquet"), "--checkpoint", str(tmp / "mdn.pt"),
        *emb_args, "--epochs", "40")

    new = pd.read_csv(results)
    pre = new[new["stage"] == "pretrain"].set_index(["head", "modalities"])
    print(pre[["mae_median", "baseline_mae", "coverage_80", "pinball"]].round(3))
    meta_mae = pre.loc[("gaussian", "tab"), "mae_median"]
    mm_mae = pre.loc[("gaussian", "tab+video+text"), "mae_median"]
    assert mm_mae < meta_mae, f"multimodal {mm_mae:.3f} should beat metadata {meta_mae:.3f} on synthetic data"
    assert new[["mae_median", "pinball"]].notna().all().all()
    print("OK")


if __name__ == "__main__":
    main()
