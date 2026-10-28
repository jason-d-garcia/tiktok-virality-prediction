"""
Checks for the TikTok-10M ingest, the shared splits and the leaderboard, on
synthetic shards (tests/synthetic_tiktok10m.py). No downloads, no torch.

    python -m tests.test_pipeline
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from scale import ingest_tiktok10m
from scale.features import add_features
from scale.leaderboard import (build_leaderboard, clustered_bootstrap_ci, quantile_columns, read_predictions,
                               smp_label, write_predictions)
from scale.posts import load_tiktok10m, views_to_smp_label
from scale.splits import assign_splits, check_splits
from tests.synthetic_tiktok10m import write_shards


def run_ingest(tmp):
    paths, df, again = write_shards(tmp / "shards", n_shards=3, n_creators=300)
    out = tmp / "tt10m.parquet"
    argv = sys.argv
    sys.argv = ["x", "--src", str(tmp / "shards"), "--out", str(out), "--batch-rows", "500"]
    try:
        ingest_tiktok10m.main()
    finally:
        sys.argv = argv
    return out, df, again


def test_ingest(tmp):
    out, df, again = run_ingest(tmp)
    posts = pd.read_parquet(out)
    card = json.loads(out.with_suffix(".card.json").read_text())
    assert posts["post_id"].is_unique and len(posts) == len(df)
    assert card["raw_rows"] == len(df) + len(again) and card["duplicate_rows"] == len(again)
    # a re-seen post keeps the later, larger reading
    pid = again["id"].iloc[0]
    assert posts.set_index("post_id").loc[pid, "views"] == again["play_count"].iloc[0]
    # age runs to stats_time, so it is later than collected_time-based age
    assert (posts["collected_time"] > posts["create_time"]).all()
    assert card["share_below_truncation"] == 0.0
    # build_posts hands the ingested table to load_tiktok10m, which passes it through
    assert len(load_tiktok10m(out)) == len(posts)
    # a re-run skips finished shards and gives the same table
    run_ingest(tmp)
    assert pd.read_parquet(out).equals(posts)
    return posts


def post_table(tmp):
    out, _, _ = run_ingest(tmp)
    tt = pd.read_parquet(out)
    rng = np.random.default_rng(1)
    emily = tt.iloc[:40].copy()
    emily["source"], emily["creator_id"] = "emily", "emily"
    emily["post_id"] = "e" + emily["post_id"]
    smp = tt.sample(50, random_state=0).copy()
    smp["source"], smp["post_id"] = "smp", "s" + smp["post_id"]
    smp["creator_id"] = "smp_u" + pd.Series(rng.integers(0, 45, len(smp)).astype(str), index=smp.index)
    return add_features(pd.concat([tt, emily, smp], ignore_index=True))


def test_splits(posts):
    splits, meta = assign_splits(posts)
    report = check_splits(posts, splits, meta)
    assert report["ok"], report["hard"]
    s = splits.merge(posts[["source", "post_id", "create_time"]], on=["source", "post_id"])
    for scheme in ("creator", "time", "strict"):
        col = s[f"split_{scheme}"]
        assert set(col[s["source"] == "smp"]) == {"external"}
        assert set(col[s["source"] == "emily"]) == {"emily_adapt", "emily_cal", "emily_test"}
    e = s[s["source"] == "emily"].sort_values("create_time")
    assert e["split_strict"].iloc[0] == "emily_adapt" and e["split_strict"].iloc[-1] == "emily_test"
    st = s[s["source"] == "tiktok10m"]
    assert (st["split_strict"] == "test").any() and (st["split_strict"] == "train").any()
    tr_c = set(st.loc[st["split_strict"] == "train", "creator_id"])
    assert not tr_c & set(st.loc[st["split_strict"] == "test", "creator_id"])

    # a planted leak must be caught: move one test creator's post into train
    bad = splits.copy()
    i = bad.index[bad["split_creator"] == "test"][0]
    bad.loc[i, "split_creator"] = "train"
    assert not check_splits(posts, bad, meta)["hard"]["creator:creators_disjoint"]
    # and an external post in train
    bad = splits.copy()
    bad.loc[bad.index[bad["source"] == "smp"][0], "split_time"] = "train"
    assert not check_splits(posts, bad, meta)["hard"]["time:no_protected_in_train_val"]
    return splits


def test_leaderboard(posts, splits, tmp):
    sp = tmp / "splits.parquet"
    splits.to_parquet(sp, index=False)
    qs = (0.1, 0.5, 0.9)
    perfect = np.repeat(posts["y_log"].to_numpy()[:, None], 3, axis=1) + np.array([-1, 0, 1])
    p1 = write_predictions(tmp / "perfect.parquet", posts, perfect, qs, model="perfect")
    p2 = write_predictions(tmp / "half.csv", posts.iloc[::2], perfect[::2, 1:2], (0.5,))
    preds = [read_predictions(p1), read_predictions(p2)]
    assert list(quantile_columns(preds[0])) == list(qs)
    board = build_leaderboard(posts, sp, "creator", preds, n_boot=100)
    test = board[(board["split"] == "test") & (board["source"] == "tiktok10m")].set_index("model")
    assert test.loc["perfect", "mae"] == 0 and test.loc["perfect", "cov80"] == 1.0
    assert test.loc["perfect", "mae_minus_base"] < 0 and test.loc["perfect", "mae_minus_base_hi"] < 0
    assert test.loc["creator_median10", "mae_minus_base"] == 0
    assert 0.4 < test.loc["half", "pred_share"] < 0.6
    assert {"test", "external", "emily_test"} <= set(board["split"])
    assert board.loc[board["split"] == "external", "smp_mae"].notna().any()


def test_smp_label():
    views, days = np.array([5000.0, 120.0, 1e6]), np.array([100.0, 3.0, 40.0])
    assert np.allclose(smp_label(np.log1p(views), days), views_to_smp_label(views, days))


def test_bootstrap():
    rng = np.random.default_rng(0)
    diff = rng.normal(-0.2, 1, 3000)
    clusters = rng.integers(0, 300, 3000)
    lo, hi = clustered_bootstrap_ci(diff, clusters, 400)
    assert lo < diff.mean() < hi and hi < 0.05
    lo1, hi1 = clustered_bootstrap_ci(diff[:50], np.zeros(50), 200)  # one creator: rows resampled
    assert lo1 < hi1


def main():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        test_ingest(tmp / "a")
        posts = post_table(tmp / "b")
        splits = test_splits(posts)
        test_leaderboard(posts, splits, tmp)
    test_smp_label()
    test_bootstrap()
    print("OK")


if __name__ == "__main__":
    main()
