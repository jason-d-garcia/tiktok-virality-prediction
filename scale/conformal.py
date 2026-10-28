"""
Conformal prediction intervals and coverage diagnostics.

All methods take scores from a held-out calibration set and return intervals
with finite-sample marginal coverage >= 1 - alpha (Mondrian: per group).

- split:   |y - yhat| residuals around a point model (constant width).
- cqr:     conformalized quantile regression (Romano et al., 2019): widens or
           narrows a quantile model's [lo, hi] by one calibrated constant.
- mondrian: any of the above calibrated separately inside each group
           (e.g. creator-size bins), so coverage holds per group, not just on average.
"""

import numpy as np
import pandas as pd


def conformal_quantile(scores, alpha):
    """The ceil((n + 1)(1 - alpha))/n empirical quantile; inf when n is too small."""
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else scores[k - 1]


def split_scores(y, yhat):
    return np.abs(y - yhat)


def cqr_scores(y, lo, hi):
    return np.maximum(lo - y, y - hi)


def split_interval(yhat, q):
    return yhat - q, yhat + q


def cqr_interval(lo, hi, q):
    return lo - q, hi + q


def mondrian_quantiles(scores, groups, alpha, min_n=30):
    """Per-group conformal quantile; groups with fewer than min_n calibration rows use the pooled one."""
    pooled = conformal_quantile(scores, alpha)
    out = {}
    for g in pd.unique(groups):
        s = scores[groups == g]
        out[g] = conformal_quantile(s, alpha) if len(s) >= min_n else pooled
    return out, pooled


def per_row_quantile(groups, qmap, pooled):
    return np.array([qmap.get(g, pooled) for g in groups], dtype=float)


def interval_score(y, lo, hi, alpha):
    """Gneiting & Raftery interval score: width plus 2/alpha times the miss distance."""
    return (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)


def coverage_table(y, lo, hi, alpha, groups=None):
    """Coverage, mean width and interval score overall and per group."""
    covered = (y >= lo) & (y <= hi)
    width = hi - lo
    iscore = interval_score(y, lo, hi, alpha)
    frame = pd.DataFrame({"group": "all" if groups is None else groups,
                          "covered": covered, "width": width, "iscore": iscore})
    rows = [("all", frame)] + ([] if groups is None else list(frame.groupby("group", sort=True)))
    return pd.DataFrame([{
        "group": str(g), "n": len(f), "coverage": f["covered"].mean(),
        "width": f["width"].mean(), "interval_score": f["iscore"].mean(),
    } for g, f in rows])


def worst_group_gap(table, alpha):
    """Largest shortfall below 1 - alpha across groups (0 when every group is covered)."""
    groups = table[table["group"] != "all"]
    return float(max(0.0, (1 - alpha) - groups["coverage"].min())) if len(groups) else 0.0
