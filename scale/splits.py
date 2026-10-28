"""
Leak-free train/val/test splits shared by every model, written once to a table.

    python -m scale.splits --posts data/scale/posts.parquet --out data/scale/splits.parquet

The table has one row per post (source, post_id) and one column per scheme:

    split_creator  held-out creators (hash of creator_id), any date
    split_time     global date cutoffs, any creator, with an embargo gap
    split_strict   held-out creators AND later dates: the headline scheme

Values: train, val, test for pretraining sources; "external" for test-only
sources (SMP-Video, EinzzCookie), which are scored but never trained on;
emily_adapt / emily_cal / emily_test for the target creator (oldest 60/20/20%,
as in scale.finetune); "purged" for posts inside an embargo gap; "history" for
held-out creators' posts before the cutoff (usable only as creator history,
which add_features already builds from posts >= 7 days older); "excluded" for
training creators' posts after the cutoff.

Why the embargo: creator-history features and the creator baseline look back
HISTORY_LAG_DAYS, and views keep growing for days after posting, so posts right
before a cutoff are dropped rather than trained on.

check_splits() runs the leakage checks; the CLI writes them to <out>.card.json
and exits non-zero if a hard check fails.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scale.features import HISTORY_LAG_DAYS
from scale.posts import TEST_ONLY_SOURCES

SCHEMES = ("creator", "time", "strict")
TARGET_CREATOR_SOURCE = "emily"
KEY = ["source", "post_id"]


def creator_split(creator_ids, val_frac=0.1, test_frac=0.1):
    """Deterministic creator-disjoint split by hashing the creator id."""
    def bucket(c):
        return int(hashlib.md5(str(c).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    b = np.array([bucket(c) for c in creator_ids])
    return np.where(b < test_frac, "test", np.where(b < test_frac + val_frac, "val", "train"))


def time_cutoffs(create_time, val_frac=0.1, test_frac=0.1):
    """Dates where val and test start: the (1 - val - test) and (1 - test) quantiles."""
    # explicit ns: pandas 3 may store microseconds
    ns = create_time.dt.tz_convert(None).to_numpy().astype("datetime64[ns]").astype("int64")
    t_val = pd.Timestamp(int(np.quantile(ns, 1 - val_frac - test_frac)), tz="UTC")
    t_test = pd.Timestamp(int(np.quantile(ns, 1 - test_frac)), tz="UTC")
    return t_val, t_test


def _time_split(t, t_val, t_test, embargo):
    return np.select(
        [t < t_val - embargo, (t >= t_val) & (t < t_test - embargo), t >= t_test],
        ["train", "val", "test"], default="purged")


def _emily_split(t, adapt=0.6, cal=0.2):
    order = np.argsort(t.to_numpy(), kind="stable")
    n = len(t)
    a, c = int(n * adapt), int(n * (adapt + cal))
    out = np.empty(n, dtype=object)
    out[order[:a]], out[order[a:c]], out[order[c:]] = "emily_adapt", "emily_cal", "emily_test"
    return out


def assign_splits(posts, val_frac=0.1, test_frac=0.1, embargo_days=HISTORY_LAG_DAYS):
    """One row per post with a split label per scheme, plus the cutoffs used."""
    df = posts[KEY + ["creator_id", "create_time"]].copy()
    out = {f"split_{s}": np.full(len(df), "", dtype=object) for s in SCHEMES}

    ext = df["source"].isin(TEST_ONLY_SOURCES).to_numpy()
    emily = (df["source"] == TARGET_CREATOR_SOURCE).to_numpy()
    pre = ~ext & ~emily
    for col in out.values():
        col[ext] = "external"
        if emily.any():
            col[emily] = _emily_split(df.loc[emily, "create_time"])

    meta = {"val_frac": val_frac, "test_frac": test_frac, "embargo_days": embargo_days}
    if pre.any():
        t = df.loc[pre, "create_time"]
        t_val, t_test = time_cutoffs(t, val_frac, test_frac)
        embargo = pd.Timedelta(days=embargo_days)
        by_creator = creator_split(df.loc[pre, "creator_id"], val_frac, test_frac)
        by_time = _time_split(t, t_val, t_test, embargo)
        strict = np.where(by_creator == by_time, by_creator, "")
        held_out = by_creator != "train"
        strict = np.where(strict != "", strict, np.where(held_out & (t < t_val).to_numpy(), "history",
                                                         np.where(held_out, "purged", "excluded")))
        # a held-out creator's post in the other held-out window (val creator, test dates) is not scored
        out["split_creator"][pre] = by_creator
        out["split_time"][pre] = by_time
        out["split_strict"][pre] = strict
        meta.update({"t_val": str(t_val), "t_test": str(t_test)})

    for name, col in out.items():
        df[name] = col
    return df.drop(columns=["create_time"]), meta


def _caption_key(posts):
    """Same creator, same normalised caption, same length: a repost, not a new post."""
    cap = posts["caption"].fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
    dur = posts["duration_s"].round().astype("Int64").astype(str)
    key = posts["creator_id"].astype(str) + "|" + cap + "|" + dur
    return key.where(cap.str.len() >= 10)


def check_splits(posts, splits, meta):
    """
    Leakage checks per scheme. Hard checks (fail -> pipeline stops): unique keys,
    no external or Emily rows in train/val, creator-disjointness, date ordering
    with the embargo. Soft checks (reported): reposts across train/test and
    training labels read after the test window opened.
    """
    df = posts.merge(splits.drop(columns="creator_id"), on=KEY, how="inner", validate="one_to_one")
    report = {"hard": {}, "soft": {}, "counts": {}}
    report["hard"]["unique_keys"] = bool(not splits.duplicated(KEY).any())
    report["hard"]["all_posts_assigned"] = bool(len(df) == len(posts))
    embargo = pd.Timedelta(days=meta["embargo_days"])
    key = _caption_key(df)

    for s in SCHEMES:
        col = df[f"split_{s}"]
        tr, va, te = col == "train", col == "val", col == "test"
        protected = df["source"].isin(TEST_ONLY_SOURCES) | (df["source"] == TARGET_CREATOR_SOURCE)
        report["hard"][f"{s}:no_protected_in_train_val"] = bool(not (protected & (tr | va)).any())
        if s in ("creator", "strict"):
            ctr = set(df.loc[tr, "creator_id"])
            report["hard"][f"{s}:creators_disjoint"] = bool(
                not (ctr & set(df.loc[te, "creator_id"])) and not (ctr & set(df.loc[va, "creator_id"])))
        if s in ("time", "strict") and tr.any() and te.any():
            report["hard"][f"{s}:train_before_val_with_embargo"] = bool(
                not va.any() or df.loc[tr, "create_time"].max() + embargo <= df.loc[va, "create_time"].min())
            report["hard"][f"{s}:val_before_test_with_embargo"] = bool(
                not va.any() or df.loc[va, "create_time"].max() + embargo <= df.loc[te, "create_time"].min())
        if tr.any() and te.any():
            shared = set(key[tr].dropna()) & set(key[te].dropna())
            report["soft"][f"{s}:test_reposts_of_train"] = int(key[te].isin(shared).sum())
            t_test = df.loc[te, "create_time"].min()
            late = df.loc[tr, "collected_time"] >= t_test
            report["soft"][f"{s}:train_labels_read_after_test_start"] = float(late.mean())
        report["counts"][s] = (df.groupby(["source", f"split_{s}"]).size()
                               .unstack(fill_value=0).to_dict(orient="index"))
    report["ok"] = all(report["hard"].values())
    return report


def load_split(splits_path, scheme):
    """(source, post_id, split) for one scheme, ready to merge onto a post table."""
    s = pd.read_parquet(splits_path)
    return s[KEY + [f"split_{scheme}"]].rename(columns={f"split_{scheme}": "split"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--embargo-days", type=float, default=HISTORY_LAG_DAYS)
    args = parser.parse_args()

    posts = pd.read_parquet(args.posts)
    splits, meta = assign_splits(posts, args.val_frac, args.test_frac, args.embargo_days)
    report = check_splits(posts, splits, meta)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(out, index=False)
    out.with_suffix(".card.json").write_text(json.dumps({"meta": meta, **report}, indent=2, default=str))

    print(json.dumps(meta, indent=2))
    for s in SCHEMES:
        print(f"\n{s}:")
        print(splits.groupby(["source", f"split_{s}"]).size().unstack(fill_value=0).to_string())
    failed = [k for k, v in report["hard"].items() if not v]
    print("\nsoft checks:", json.dumps(report["soft"], indent=2))
    if failed:
        print("FAILED hard checks:", failed)
        sys.exit(1)
    print(f"All hard leakage checks passed. Saved {out}")


if __name__ == "__main__":
    main()
