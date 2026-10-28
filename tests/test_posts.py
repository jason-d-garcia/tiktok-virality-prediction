"""
Unit checks for the loader and history fixes (memo section 15, milestone 1).
Synthetic tables only, no downloads.

    python -m tests.test_posts
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from scale.features import add_features, creator_baseline, filter_min_age, impute_history
from scale.posts import load_einzz, load_emily, load_smp, load_tiktok10m, smp_label_to_views, views_to_smp_label


def test_smp_label_roundtrip():
    rng = np.random.default_rng(0)
    views = np.floor(10 ** rng.uniform(1, 7, 1000))
    days = rng.uniform(1, 400, 1000)
    s = views_to_smp_label(views, days)
    assert np.allclose(smp_label_to_views(s, days), views, rtol=1e-12)


def test_load_smp(tmp):
    post_time = pd.Timestamp("2024-03-01", tz="UTC")
    crawl = "2024-06-09"  # 100 days later, as passed by --smp-collected
    pd.DataFrame({"pid": ["a", "b"], "uid": ["u1", "u1"], "post_time": [post_time] * 2,
                  "post_content": ["x", "y #fyp"]}).to_json(tmp / "SMP-Video_anonymized_posts_train.jsonl",
                                                             orient="records", lines=True, date_format="iso")
    labels = views_to_smp_label(np.array([5000.0, 120.0]), 100)
    pd.DataFrame({"pid": ["a", "b"], "popularity": labels}).to_json(
        tmp / "SMP-Video_anonymized_popularity_train.jsonl", orient="records", lines=True)
    pd.DataFrame({"pid": ["a", "b"], "video_duration": [12, 60]}).to_json(
        tmp / "SMP-Video_anonymized_videos_train.jsonl", orient="records", lines=True)
    df = load_smp(tmp, collected_time=crawl)
    assert np.allclose(df.set_index("post_id")["views"].loc[["a", "b"]], [5000.0, 120.0])
    assert df.set_index("post_id")["duration_s"].to_dict() == {"a": 12.0, "b": 60.0}
    try:
        load_smp(tmp)
    except ValueError:
        pass
    else:
        raise AssertionError("load_smp should refuse to guess a crawl date")


def test_tiktok10m_age_and_dedup(tmp):
    create = pd.Timestamp("2024-01-01", tz="UTC")
    pd.DataFrame({
        "id": ["p1", "p1", "p2"], "user_id": ["c"] * 3,
        "create_time": [int(create.timestamp())] * 3,
        "collected_time": ["2024-01-11", "2024-02-01", "2024-01-11"],
        "stats_time": ["2024-03-01", "2024-04-10", "2024-03-01"],
        "play_count": [40_000, 90_000, 50_000], "duration": [10, 10, 20],
        "desc": ["", "", ""], "url": ["", "", ""],
    }).to_parquet(tmp / "t10m.parquet")
    df = load_tiktok10m(tmp / "t10m.parquet").set_index("post_id")
    assert df.loc["p1", "views"] == 90_000, "latest reading of a repeated post should win"
    age = (df["collected_time"] - df["create_time"]).dt.days
    assert age["p1"] == 100 and age["p2"] == 60, "age should run to stats_time"
    old = load_tiktok10m(tmp / "t10m.parquet", views_at="collected_time").set_index("post_id")
    assert (old.loc["p2", "collected_time"] - old.loc["p2", "create_time"]).days == 10


def test_einzz_dedup(tmp):
    row = {"aweme_id": "1", "author": {"uid": "u", "unique_id": "name", "follower_count": None},
           "create_time": 1700000000, "statistics": {"play_count": 10}, "desc": "", "video": {"duration": 9000}}
    rows = [row] * 150 + [{**row, "aweme_id": "2"}] * 150
    pd.DataFrame(rows).to_json(tmp / "einzz.jsonl", orient="records", lines=True)
    df = load_einzz(tmp / "einzz.jsonl")
    assert len(df) == 2 and df["duration_s"].iloc[0] == 9.0


def creator_posts(days, views):
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    return pd.DataFrame({
        "post_id": [str(i) for i in range(len(days))], "source": "x", "creator_id": "c",
        "create_time": [t0 + pd.Timedelta(days=d) for d in days],
        "collected_time": t0 + pd.Timedelta(days=60), "views": views, "followers": np.nan,
        "duration_s": 10.0, "caption": "", "video_url": "",
    })


def test_lagged_history():
    # posts on days 0, 1, 2, 10, 11; only posts >= 7 days older count as history
    df = add_features(creator_posts([0, 1, 2, 10, 11], [9, 99, 999, 9999, 99999]))
    y = np.log1p([9, 99, 999])
    assert df["creator_has_history"].tolist() == [0, 0, 0, 1, 1]
    # day 10 sees posts up to day 3 and day 11 up to day 4: the same three posts, not day 10's
    assert np.allclose(np.expm1(df["creator_prior_posts"]), [0, 0, 0, 3, 3])
    assert np.isclose(df.loc[3, "creator_prior_median"], np.median(y))
    base = creator_baseline(df)
    assert base.isna().tolist() == [True, True, True, False, False]
    assert np.isclose(base[4], np.median(y))
    filled = impute_history(df, 5.0)
    assert (filled.loc[:2, "creator_prior_median"] == 5.0).all() and filled["creator_prior_std"].notna().all()


def test_min_age_filter():
    df = creator_posts([0, 55, 58], [1, 2, 3])  # collected on day 60: ages 60, 5, 2
    df.loc[2, "collected_time"] = pd.NaT
    kept = filter_min_age(df, 7)
    assert kept["post_id"].tolist() == ["0", "2"], "drop known-young posts, keep unknown ages"


def test_emily_unknown_collection_date(tmp):
    pd.DataFrame({"url": ["https://www.tiktok.com/@e/video/1"], "description": ["x"],
                  "published date": ["7/29/19"], "published time": ["11:30 AM"],
                  "video length": ["8.98s"], "views": [230221]}).to_csv(tmp / "emily.csv", index=False)
    df = load_emily(tmp / "emily.csv")
    assert len(filter_min_age(df, 7)) == 1, "unknown collection date must not break age arithmetic"


def main():
    test_smp_label_roundtrip()
    for test in (test_load_smp, test_tiktok10m_age_and_dedup, test_einzz_dedup,
                 test_emily_unknown_collection_date):
        with tempfile.TemporaryDirectory() as d:
            test(Path(d))
    test_lagged_history()
    test_min_age_filter()
    print("OK")


if __name__ == "__main__":
    main()
