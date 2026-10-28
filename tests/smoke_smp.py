"""Synthetic SMP-Video release -> smp_benchmark end to end, plus a conformal coverage check. CPU, ~1 min."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from scale import conformal as cp
from scale import smp_benchmark


def check_conformal(seed=0, n=4000, alpha=0.1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = x + rng.normal(0, 0.1 + x, n)  # heteroscedastic
    cal, te = slice(0, n // 2), slice(n // 2, n)
    q = cp.conformal_quantile(cp.split_scores(y[cal], x[cal]), alpha)
    lo, hi = cp.split_interval(x[te], q)
    cov = np.mean((y[te] >= lo) & (y[te] <= hi))
    assert abs(cov - (1 - alpha)) < 0.03, cov
    assert cp.conformal_quantile([1.0, 2.0], 0.1) == np.inf


def write_fake_release(root, n_users=1500, seed=0):
    rng = np.random.default_rng(seed)
    uid = np.arange(n_users)
    fol = np.exp(rng.normal(8, 2, n_users)).round()
    users = pd.DataFrame({"uid": [f"u{u}" for u in uid], "user_follower_count": fol,
                          "user_likes_count": (fol * rng.uniform(1, 20, n_users)).round()})
    post_u = rng.choice(uid, 4000)
    words = np.array(["dance", "food", "cat", "prank", "news", "fyp", "makeup", "car"])
    w = rng.choice(len(words), 4000)
    t = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.uniform(0, 700, 4000), unit="D")
    dur = rng.uniform(5, 120, 4000)
    pop = 1 + 0.6 * np.log2(fol[post_u] + 1) + (w == 2) * 2 - 0.02 * dur + rng.normal(0, 1.5 + (w == 3), 4000)
    posts = pd.DataFrame({"pid": [f"p{i}" for i in range(4000)], "uid": [f"u{u}" for u in post_u],
                          "post_time": t.astype(str), "post_content": [f"#{x} my video" for x in words[w]]})
    for name, df in (("posts", posts), ("users", users),
                     ("labels", pd.DataFrame({"pid": posts["pid"], "popularity": pop})),
                     ("videos", pd.DataFrame({"pid": posts["pid"], "video_duration": dur}))):
        df.to_json(Path(root) / f"{smp_benchmark.SMP_FILES[name]}.jsonl", orient="records", lines=True)


def main():
    check_conformal()
    with tempfile.TemporaryDirectory() as tmp:
        write_fake_release(tmp)
        args = type("A", (), dict(smp_dir=tmp, emb_dir=None, folds=3, cal_frac=0.25, seed=0))()
        args.conformal_sets = lambda f: {f[-2], f[-1]}
        points, cov = smp_benchmark.run(args)
        smp_benchmark.write_report(points, cov, Path(tmp) / "out")
    best = points.sort_values("mae").iloc[0]
    assert best["model"] != "mean", points
    overall = cov[(cov["group"] == "all") & (cov["method"] == "cqr")]
    assert (overall["coverage"] > 1 - overall["alpha"] - 0.04).all(), overall
    print("smoke_smp OK")


if __name__ == "__main__":
    main()
