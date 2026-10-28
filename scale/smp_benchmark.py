"""
SMP-Video benchmark: tabular + text models on the challenge label, with conformal intervals.

The target is the challenge label itself, s = log2(views / days) + 1, so no
crawl date is needed. Post time enters as calendar features; because the crawl
date is one constant, a monotone "days since 2022-01-01" covariate carries the
same information as post age.

Protocol: 5-fold creator-disjoint CV on the labeled train split (the challenge
test labels are not public, so scores are not directly comparable to the
leaderboard). Inside each outer fold, 25% of training creators are held out to
calibrate conformal intervals. Metrics match the challenge: MAE, MSE, Spearman.

Feature sets (each adds to the previous):
  meta  calendar time, caption statistics, video metadata (numeric columns)
  text  + TF-IDF caption embedding (SVD, fit inside the fold)
  clip  + CLIP caption embeddings from scale.embed shards (--emb-dir), PCA in fold
  user  + creator stats (followers, likes, video count...). Read at crawl time,
        so they include post-publication growth; challenge entries use them.

    python -m scale.smp_benchmark --smp-dir data/hf/SMP-Video [--emb-dir data/scale/emb/text]
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scale import conformal as cp
from scale.posts import SMP_FILES, _smp_table

ALPHAS = (0.1, 0.2)
TEXT_COLS = re.compile(r"content|suggest|hashtag|title|desc|keyword|music", re.I)
ID_COLS = re.compile(r"(^|_)(pid|uid|id|url)$", re.I)
EPOCH = pd.Timestamp("2022-01-01", tz="UTC")


# ---------------------------------------------------------------- data

def _numeric(df, prefix):
    keep = [c for c in df.columns if not ID_COLS.search(c) and pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])]
    return df[keep].astype(float).add_prefix(prefix)


def _text(df):
    cols = [c for c in df.columns if TEXT_COLS.search(c) and df[c].dtype == object]
    if not cols:
        return pd.Series("", index=df.index)
    return df[cols].apply(lambda r: " ".join(" ".join(map(str, v)) if isinstance(v, (list, np.ndarray))
                                             else str(v) for v in r if v is not None and v == v), axis=1)


def load_smp_frame(smp_dir):
    posts = _smp_table(smp_dir, SMP_FILES["posts"])
    labels = _smp_table(smp_dir, SMP_FILES["labels"])[["pid", "popularity"]]
    df = posts.merge(labels, on="pid", how="inner").reset_index(drop=True)
    text = _text(posts.set_index("pid").loc[df["pid"]].reset_index())

    video = pd.DataFrame(index=df.index)
    try:
        videos = _smp_table(smp_dir, SMP_FILES["videos"]).drop_duplicates("pid")
        joined = df[["pid"]].merge(videos, on="pid", how="left")
        video = _numeric(joined, "video_")
        text = text + " " + _text(joined)
    except FileNotFoundError:
        pass
    user = pd.DataFrame(index=df.index)
    try:
        users = _smp_table(smp_dir, SMP_FILES["users"]).drop_duplicates("uid")
        user = np.log1p(_numeric(df[["uid"]].merge(users, on="uid", how="left"), "user_").clip(lower=0))
    except FileNotFoundError:
        pass

    t = pd.to_datetime(df["post_time"], utc=True, errors="coerce")
    if t.isna().mean() > 0.5:  # epoch seconds
        t = pd.to_datetime(pd.to_numeric(df["post_time"], errors="coerce"), unit="s", utc=True)
    cap = df["post_content"].fillna("").astype(str)
    meta = pd.DataFrame({
        "days_since_2022": (t - EPOCH).dt.total_seconds() / 86400,
        "hour_sin": np.sin(2 * np.pi * t.dt.hour / 24), "hour_cos": np.cos(2 * np.pi * t.dt.hour / 24),
        "dow_sin": np.sin(2 * np.pi * t.dt.dayofweek / 7), "dow_cos": np.cos(2 * np.pi * t.dt.dayofweek / 7),
        "weekend": (t.dt.dayofweek >= 5).astype(float),
        "cap_len": cap.str.len(), "cap_words": cap.str.split().str.len(),
        "n_hashtags": cap.str.count("#"), "n_mentions": cap.str.count("@"),
        "n_nonascii": cap.map(lambda s: sum(ord(ch) > 127 for ch in s)),
    })
    meta = pd.concat([meta, video], axis=1)
    print(f"smp: {len(df):,} labeled posts, {df['uid'].nunique():,} creators; "
          f"meta {meta.shape[1]} cols, user {user.shape[1]} cols ({', '.join(user.columns)})")
    return df, meta, user, (cap + " " + text).str.lower()


# ---------------------------------------------------------------- features

class FoldFeatures:
    """Fit text/embedding reducers on training rows only, then build matrices for any rows."""

    def __init__(self, meta, user, text, clip, fset, n_text=64, n_clip=32, seed=0):
        self.meta, self.user, self.text, self.clip, self.fset = meta, user, text, clip, fset
        self.n_text, self.n_clip, self.seed = n_text, n_clip, seed

    def fit(self, idx):
        if "text" in self.fset:
            self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)
            X = self.tfidf.fit_transform(self.text.iloc[idx])
            self.svd = TruncatedSVD(min(self.n_text, X.shape[1] - 1), random_state=self.seed).fit(X)
        if "clip" in self.fset:
            ok = idx[~np.isnan(self.clip[idx]).any(1)]
            self.pca = PCA(self.n_clip, random_state=self.seed).fit(self.clip[ok])
        return self

    def transform(self, idx):
        blocks = [self.meta.iloc[idx].to_numpy(float)]
        if "text" in self.fset:
            blocks.append(self.svd.transform(self.tfidf.transform(self.text.iloc[idx])))
        if "clip" in self.fset:
            E = self.clip[idx]
            out = np.full((len(idx), self.n_clip), np.nan)
            ok = ~np.isnan(E).any(1)
            if ok.any():
                out[ok] = self.pca.transform(E[ok])
            blocks.append(out)
        if "user" in self.fset:
            blocks.append(self.user.iloc[idx].to_numpy(float))
        return np.hstack(blocks)


def hgb(loss="squared_error", quantile=None, seed=0):
    kw = {"quantile": quantile} if loss == "quantile" else {}
    return HistGradientBoostingRegressor(loss=loss, learning_rate=0.05, max_iter=400, max_leaf_nodes=15,
                                         min_samples_leaf=30, l2_regularization=1.0, early_stopping=True,
                                         validation_fraction=0.15, random_state=seed, **kw)


def ridge():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))


def point_scores(y, yhat):
    return {"mae": float(np.mean(np.abs(y - yhat))), "mse": float(np.mean((y - yhat) ** 2)),
            "src": float(spearmanr(y, yhat).correlation)}


# ---------------------------------------------------------------- run

def creator_folds(uids, k, seed):
    uniq = pd.unique(uids)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(rng.permutation(uniq), np.arange(len(uniq)) % k))
    return np.array([fold_of[u] for u in uids])


def size_bins(followers, edges):
    labels = ["Q1 small", "Q2", "Q3", "Q4 large"]
    b = np.digitize(followers, edges)
    return np.where(np.isnan(followers), "unknown", np.array(labels)[np.clip(b, 0, 3)])


def run(args):
    df, meta, user, text = load_smp_frame(args.smp_dir)
    y = df["popularity"].to_numpy(float)
    clip = None
    fsets = ["meta", "meta+text"]
    if args.emb_dir:
        from scale.embed import load_embeddings
        clip = load_embeddings(args.emb_dir, ("smp_" + df["pid"].astype(str)).tolist(), ("text",))["text"]
        print(f"clip text embeddings for {np.mean(~np.isnan(clip).any(1)):.0%} of posts")
        fsets.append("meta+text+clip")
    fsets += [fsets[-1] + "+user"]
    fol_col = next((c for c in user.columns if "follower" in c), None)
    followers = user[fol_col].to_numpy(float) if fol_col else np.full(len(df), np.nan)

    folds = creator_folds(df["uid"].to_numpy(), args.folds, args.seed)
    point_rows, pred = [], {}
    cov_rows = []
    for k in range(args.folds):
        tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
        rng = np.random.default_rng(args.seed + k)
        tr_uids = pd.unique(df["uid"].iloc[tr])
        cal_uids = set(rng.choice(tr_uids, int(len(tr_uids) * args.cal_frac), replace=False))
        is_cal = df["uid"].iloc[tr].isin(cal_uids).to_numpy()
        fit, cal = tr[~is_cal], tr[is_cal]

        # baselines
        preds = {("mean", "-"): np.full(len(te), y[tr].mean())}
        if fol_col:
            m = ridge().fit(user[[fol_col]].iloc[tr], y[tr])
            preds[("ridge", "followers only")] = m.predict(user[[fol_col]].iloc[te])

        for fset in fsets:
            ff = FoldFeatures(meta, user, text, clip, fset, seed=args.seed).fit(tr)
            Xtr, Xte = ff.transform(tr), ff.transform(te)
            preds[("ridge", fset)] = ridge().fit(Xtr, y[tr]).predict(Xte)
            preds[("hgb", fset)] = hgb(seed=args.seed).fit(Xtr, y[tr]).predict(Xte)

            if fset not in args.conformal_sets(fsets):
                continue
            # conformal: models on `fit`, scores on `cal`, intervals on `te`
            ff = FoldFeatures(meta, user, text, clip, fset, seed=args.seed).fit(fit)
            Xf, Xc, Xt = ff.transform(fit), ff.transform(cal), ff.transform(te)
            point = hgb(seed=args.seed).fit(Xf, y[fit])
            yc, yt = point.predict(Xc), point.predict(Xt)
            edges = np.nanquantile(followers[fit], [0.25, 0.5, 0.75]) if fol_col else [0, 0, 0]
            g_cal, g_te = size_bins(followers[cal], edges), size_bins(followers[te], edges)
            # predicted-level bins: does coverage hold for posts the model thinks will flop or pop?
            lvl_edges = np.quantile(yc, [1 / 3, 2 / 3])
            lvl_te = np.array(["pred low", "pred mid", "pred high"])[np.digitize(yt, lvl_edges)]
            for alpha in ALPHAS:
                lo_m = hgb("quantile", alpha / 2, args.seed).fit(Xf, y[fit])
                hi_m = hgb("quantile", 1 - alpha / 2, args.seed).fit(Xf, y[fit])
                lc, hc, lt, ht = lo_m.predict(Xc), hi_m.predict(Xc), lo_m.predict(Xt), hi_m.predict(Xt)
                s_split, s_cqr = cp.split_scores(y[cal], yc), cp.cqr_scores(y[cal], lc, hc)
                methods = {
                    "raw quantile (no conformal)": (lt, ht),
                    "split": cp.split_interval(yt, cp.conformal_quantile(s_split, alpha)),
                    "cqr": cp.cqr_interval(lt, ht, cp.conformal_quantile(s_cqr, alpha)),
                }
                qm, pooled = cp.mondrian_quantiles(s_split, g_cal, alpha)
                methods["mondrian split"] = cp.split_interval(yt, cp.per_row_quantile(g_te, qm, pooled))
                qm, pooled = cp.mondrian_quantiles(s_cqr, g_cal, alpha)
                methods["mondrian cqr"] = cp.cqr_interval(lt, ht, cp.per_row_quantile(g_te, qm, pooled))
                for name, (lo, hi) in methods.items():
                    for by, groups in (("creator size", g_te), ("predicted level", lvl_te)):
                        t = cp.coverage_table(y[te], lo, hi, alpha, groups)
                        t = t if by == "creator size" else t[t["group"] != "all"]
                        cov_rows.append(t.assign(fold=k, fset=fset, alpha=alpha, method=name, by=by))

        for key, p in preds.items():
            pred.setdefault(key, np.full(len(y), np.nan))[te] = p
        print(f"fold {k + 1}/{args.folds} done")

    for (model, fset), p in pred.items():
        per_fold = [point_scores(y[folds == k], p[folds == k]) for k in range(args.folds)]
        row = {"model": model, "features": fset, **point_scores(y, p)}
        row["src_fold_sd"] = float(np.std([s["src"] for s in per_fold]))
        point_rows.append(row)
    points = pd.DataFrame(point_rows)
    cov = pd.concat(cov_rows, ignore_index=True)
    # pool folds: weight each fold's group by its row count
    cov["covered_n"], cov["width_n"], cov["is_n"] = (cov["coverage"] * cov["n"], cov["width"] * cov["n"],
                                                     cov["interval_score"] * cov["n"])
    cov = (cov.groupby(["fset", "alpha", "method", "by", "group"], sort=False)
           .agg(n=("n", "sum"), covered_n=("covered_n", "sum"), width_n=("width_n", "sum"), is_n=("is_n", "sum"))
           .reset_index())
    cov["coverage"], cov["width"], cov["interval_score"] = (cov["covered_n"] / cov["n"], cov["width_n"] / cov["n"],
                                                            cov["is_n"] / cov["n"])
    cov = cov.drop(columns=["covered_n", "width_n", "is_n"])
    return points, cov


def fmt(d):
    """Markdown table without the tabulate dependency."""
    cell = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)  # noqa: E731
    rows = [list(map(str, d.columns))] + [[cell(v) for v in r] for r in d.itertuples(index=False)]
    return "\n".join("| " + " | ".join(r) + " |" for r in rows[:1] + [["---"] * len(d.columns)] + rows[1:])


def write_report(points, cov, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    points.to_csv(out_dir / "smp_benchmark.csv", index=False)
    cov.to_csv(out_dir / "smp_conformal.csv", index=False)
    lines = ["# SMP-Video benchmark", "",
             "5-fold creator-disjoint CV on the labeled train split; target is the challenge label "
             "log2(views/day) + 1. Lower MAE/MSE and higher SRC are better.", "",
             fmt(points.sort_values("mae")), ""]
    for fset in cov["fset"].unique():
        c = cov[cov["fset"] == fset]
        lines += [f"## Conformal intervals: {fset}", ""]
        summ = c[(c["by"] == "creator size")]
        wide = summ.pivot_table(index=["alpha", "method"], columns="group", values="coverage", sort=False)
        width = summ[summ["group"] == "all"].set_index(["alpha", "method"])[["width", "interval_score"]]
        lines += ["Coverage overall and by creator-size quartile, with mean width (label units):", "",
                  fmt(wide.join(width).reset_index()), ""]
        lvl = c[c["by"] == "predicted level"].pivot_table(index=["alpha", "method"], columns="group",
                                                         values="coverage", sort=False)
        lines += ["Coverage by predicted level (tercile of the point prediction):", "",
                  fmt(lvl.reset_index()), ""]
    (out_dir / "smp_benchmark.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smp-dir", required=True)
    ap.add_argument("--emb-dir", default=None, help="scale.embed text shards (CLIP caption embeddings)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--cal-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()
    # intervals for the best prepublication set and the same set with creator stats
    args.conformal_sets = lambda fsets: {fsets[-2], fsets[-1]}
    points, cov = run(args)
    write_report(points, cov, args.out_dir)


if __name__ == "__main__":
    main()
