"""
One leaderboard for every model, on the shared splits.

    python -m scale.leaderboard --posts data/scale/posts.parquet \
        --splits data/scale/splits.parquet --scheme strict \
        --preds runs/*.parquet --out reports/leaderboard

Prediction files (parquet or csv), one per model, on the log1p(views) scale:

    source, post_id     keys into the post table
    q0.5                the median; required
    q0.05 ... q0.95     any other quantile levels (optional; enables pinball,
                        coverage, interval score and the quantile CRPS)
    model               optional; defaults to the file stem

Use write_predictions() to produce one. Predict every row of the evaluation
splits (test, external, emily_test); pred_share in the table says how many
rows a model covered, and models below 99% are marked.

Built-in baselines, computed here from the post table so nobody has to ship them:

    global_train        training-split empirical quantiles for every post
    creator_median10    median log views of the creator's last 10 posts published
                        >= 7 days earlier (the honest baseline from PR #1);
                        training median when a creator has no such history
    creator_median10+q  the same median plus training-split residual quantiles,
                        so the baseline has intervals too

Every model is scored on the same rows per (split, source): median MAE,
Spearman, pinball, CRPS from the quantiles, 80/90% coverage, width and interval
score, MAE on the SMP-Video label scale (log2(views/day) + 1) when post age is
known, and the MAE gap to creator_median10 on rows where the creator has
history, with a creator-clustered bootstrap 95% CI.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scale.features import creator_baseline
from scale.splits import KEY, load_split

EVAL_SPLITS = ("test", "external", "emily_test")
DEFAULT_QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
BASELINE = "creator_median10"
QCOL = re.compile(r"^q(0?\.\d+)$")


def qcol(q):
    return f"q{q:g}"


def quantile_columns(df):
    """{level: column} for every qX column, sorted by level."""
    found = {float(m.group(1)): c for c in df.columns if (m := QCOL.match(c))}
    return dict(sorted(found.items()))


def write_predictions(path, keys, q_pred, quantiles=DEFAULT_QUANTILES, model=None):
    """Save predictions in the leaderboard format. keys holds source and post_id, row-aligned with q_pred."""
    q_pred = np.asarray(q_pred, dtype=float).reshape(len(keys), -1)
    out = keys[KEY].reset_index(drop=True).copy()
    for i, q in enumerate(quantiles):
        out[qcol(q)] = q_pred[:, i]
    if model:
        out["model"] = model
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    (out.to_csv(path, index=False) if path.suffix == ".csv" else out.to_parquet(path, index=False))
    return path


def read_predictions(path):
    path = Path(path)
    df = pd.read_csv(path, dtype={"post_id": str}) if path.suffix == ".csv" else pd.read_parquet(path)
    df["post_id"] = df["post_id"].astype(str)
    if "q0.5" not in df.columns:
        raise KeyError(f"{path}: needs a q0.5 column; got {list(df.columns)}")
    if "model" not in df.columns:
        df["model"] = path.stem
    if df.duplicated(KEY + ["model"]).any():
        raise ValueError(f"{path}: duplicate (source, post_id) rows")
    return df


def builtin_baselines(posts, quantiles=DEFAULT_QUANTILES):
    """Prediction frames for the built-in baselines. posts must carry y_log and a split column."""
    posts = posts.sort_values(["creator_id", "create_time"]).reset_index(drop=True)
    train = posts["split"] == "train"
    if not train.any():
        raise ValueError("no training rows in this scheme; baselines need them")
    y_tr = posts.loc[train, "y_log"]
    qs = np.asarray(quantiles)

    glob = posts[KEY].copy()
    for q, v in zip(qs, np.quantile(y_tr, qs)):
        glob[qcol(q)] = v
    glob["model"] = "global_train"

    base = creator_baseline(posts)
    point = base.fillna(float(y_tr.median()))
    med = posts[KEY].copy()
    med["q0.5"] = point.to_numpy()
    med["model"] = BASELINE

    resid = (posts["y_log"] - base)[train & base.notna()]
    offsets = np.quantile(resid, qs) if len(resid) else np.zeros(len(qs))
    offsets -= offsets[list(qs).index(0.5)] if 0.5 in qs else 0.0  # keep the median on the point baseline
    band = posts[KEY].copy()
    for q, o in zip(qs, offsets):
        band[qcol(q)] = point.to_numpy() + o
    band["model"] = BASELINE + "+q"
    return [glob, med, band], base.to_numpy(), posts


def smp_label(y_log, age_days):
    """SMP-Video popularity label, log2(views / days) + 1, from log1p(views)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.log2(np.expm1(y_log) / age_days) + 1
    return np.where((age_days > 0) & (np.expm1(y_log) > 0), s, np.nan)


def pinball(y, q_pred, levels):
    diff = y[:, None] - q_pred
    return np.maximum(levels * diff, (levels - 1) * diff)


def clustered_bootstrap_ci(diff, clusters, n_boot=1000, seed=0, chunk=100):
    """
    95% CI for mean(diff) resampling whole creators (Poisson weights). With a
    single creator (Emily) there is nothing to cluster, so rows are resampled.
    """
    codes, uniq = pd.factorize(clusters)
    if len(uniq) < 2:
        codes, uniq = np.arange(len(diff)), np.arange(len(diff))
    s = np.bincount(codes, weights=diff, minlength=len(uniq))
    n = np.bincount(codes, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    stats = []
    for b in range(0, n_boot, chunk):
        w = rng.poisson(1.0, size=(min(chunk, n_boot - b), len(uniq)))
        tot = w @ n
        stats.append((w @ s)[tot > 0] / tot[tot > 0])
    lo, hi = np.percentile(np.concatenate(stats), [2.5, 97.5])
    return float(lo), float(hi)


def score_rows(y, preds, levels, age_days):
    """Metrics for one model on aligned rows. preds is an (n, len(levels)) array."""
    med = preds[:, list(levels).index(0.5)]
    out = {
        "n": len(y),
        "mae": float(np.mean(np.abs(y - med))),
        "spearman": float(spearmanr(y, med).correlation) if len(y) > 2 and np.std(med) > 0 and np.std(y) > 0 else np.nan,
    }
    if len(levels) > 1:
        lv = np.asarray(levels)
        pl = pinball(y, preds, lv)
        out["pinball"] = float(pl.mean())
        if len(levels) >= 5:
            # CRPS = 2 * integral of pinball over levels; a mean over the levels approximates it
            out["crps_q"] = float(2 * pl.mean())
        for lo_q, hi_q, name in ((0.10, 0.90, "80"), (0.05, 0.95, "90")):
            if lo_q in levels and hi_q in levels:
                lo, hi = preds[:, list(levels).index(lo_q)], preds[:, list(levels).index(hi_q)]
                alpha = 1 - int(name) / 100
                out[f"cov{name}"] = float(np.mean((y >= lo) & (y <= hi)))
                out[f"width{name}"] = float(np.mean(hi - lo))
                out[f"iscore{name}"] = float(np.mean(
                    (hi - lo) + (2 / alpha) * ((lo - y) * (y < lo) + (y - hi) * (y > hi))))
    s_true, s_pred = smp_label(y, age_days), smp_label(med, age_days)
    ok = np.isfinite(s_true) & np.isfinite(s_pred)
    if ok.sum() > 2 and np.std(s_pred[ok]) > 0:
        out["smp_mae"] = float(np.mean(np.abs(s_true[ok] - s_pred[ok])))
        out["smp_spearman"] = float(spearmanr(s_true[ok], s_pred[ok]).correlation)
    return out


def build_leaderboard(posts, splits_path, scheme, pred_frames, n_boot=1000, eval_splits=EVAL_SPLITS):
    posts = posts.copy()
    posts["post_id"] = posts["post_id"].astype(str)
    posts = posts.merge(load_split(splits_path, scheme), on=KEY, how="inner", validate="one_to_one")
    baselines, base_raw, posts = builtin_baselines(posts)
    posts["baseline_raw"] = base_raw
    age = (posts["collected_time"] - posts["create_time"]).dt.total_seconds() / 86400

    ev = posts["split"].isin(eval_splits).to_numpy()
    rows_all = posts.loc[ev, KEY + ["creator_id", "split", "y_log", "baseline_raw"]].assign(
        age_days=age[ev].to_numpy()).reset_index(drop=True)
    base_err = np.abs(rows_all["y_log"] - rows_all["baseline_raw"]).to_numpy()

    results = []
    for frame in baselines + list(pred_frames):
        for model, pf in frame.groupby("model", sort=False):
            qmap = quantile_columns(pf)
            levels = list(qmap)
            m = rows_all.merge(pf[KEY + list(qmap.values())], on=KEY, how="left")
            preds = m[list(qmap.values())].to_numpy(float)
            have = ~np.isnan(preds).any(1)
            for (split, source), g in rows_all.groupby(["split", "source"]):
                idx = g.index.to_numpy()
                ok = idx[have[idx]]
                row = {"model": model, "scheme": scheme, "split": split, "source": source,
                       "pred_share": float(len(ok) / len(idx))}
                if len(ok):
                    row.update(score_rows(rows_all.loc[ok, "y_log"].to_numpy(), preds[ok], levels,
                                          rows_all.loc[ok, "age_days"].to_numpy()))
                    b = ok[~np.isnan(base_err[ok])]
                    if len(b) > 2:
                        med = preds[b, levels.index(0.5)]
                        diff = np.abs(rows_all.loc[b, "y_log"].to_numpy() - med) - base_err[b]
                        row["n_vs_base"] = len(b)
                        row["mae_minus_base"] = float(diff.mean())
                        if n_boot:
                            lo, hi = clustered_bootstrap_ci(diff, rows_all.loc[b, "creator_id"].to_numpy(), n_boot)
                            row["mae_minus_base_lo"], row["mae_minus_base_hi"] = lo, hi
                results.append(row)
    board = pd.DataFrame(results)
    sort_key = "crps_q" if "crps_q" in board else "mae"
    return board.sort_values(["split", "source", sort_key, "mae"], na_position="last").reset_index(drop=True)


def to_markdown(board):
    lines = []
    show = [c for c in ["model", "n", "pred_share", "mae", "spearman", "crps_q", "cov80", "cov90", "width80",
                        "iscore80", "smp_mae", "mae_minus_base", "mae_minus_base_lo", "mae_minus_base_hi"]
            if c in board]
    for (scheme, split, source), g in board.groupby(["scheme", "split", "source"], sort=False):
        lines += [f"### {scheme} / {split} / {source}", ""]
        g = g[show].copy()
        g["model"] = np.where(g["pred_share"] < 0.99, g["model"] + " (partial)", g["model"])
        lines += ["| " + " | ".join(show) + " |", "|" + "---|" * len(show)]
        for _, r in g.iterrows():
            cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in r]
            lines.append("| " + " | ".join(c.replace("nan", "") for c in cells) + " |")
        lines.append("")
    lines.append("Lower is better for mae, crps_q, width, iscore, smp_mae and mae_minus_base (negative beats "
                 "the creator baseline; the CI is creator-clustered). Coverage should match its nominal level.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--posts", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--scheme", default="strict", choices=["creator", "time", "strict"])
    parser.add_argument("--preds", nargs="*", default=[])
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out", default="reports/leaderboard", help="writes <out>_<scheme>.csv and .md")
    args = parser.parse_args()

    posts = pd.read_parquet(args.posts)
    board = build_leaderboard(posts, args.splits, args.scheme, [read_predictions(p) for p in args.preds],
                              args.n_boot)
    out = Path(f"{args.out}_{args.scheme}")
    out.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(out.with_suffix(".csv"), index=False)
    out.with_suffix(".md").write_text(to_markdown(board))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(board[[c for c in ["split", "source", "model", "n", "mae", "spearman", "crps_q", "cov80",
                                 "mae_minus_base"] if c in board]].to_string(index=False))
    print(f"Saved {out.with_suffix('.csv')} and {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
