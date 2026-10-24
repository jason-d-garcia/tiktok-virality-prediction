"""Scoring for distributional forecasts on the log1p(views) scale."""

import numpy as np
from scipy.stats import spearmanr

from scale.model import QUANTILES


def pinball_loss(q_pred, y, qs=QUANTILES):
    qs = np.asarray(qs)
    diff = y[:, None] - q_pred
    return np.maximum(qs * diff, (qs - 1) * diff).mean()


def interval(q_pred, lo_q, hi_q, qs=QUANTILES):
    qs = list(qs)
    return q_pred[:, qs.index(lo_q)], q_pred[:, qs.index(hi_q)]


def interval_score(y, lo, hi, alpha):
    """Gneiting & Raftery interval score: width plus 2/alpha times the miss distance."""
    return (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)


def score_forecast(y_log, q_pred, nll_count=None, baseline=None, qs=QUANTILES):
    median = q_pred[:, list(qs).index(0.5)]
    out = {
        "n": len(y_log),
        "mae_median": float(np.mean(np.abs(y_log - median))),
        "spearman": float(spearmanr(y_log, median).correlation) if len(y_log) > 2 else np.nan,
        "pinball": float(pinball_loss(q_pred, y_log, qs)),
    }
    for lo_q, hi_q, name in ((0.10, 0.90, "80"), (0.05, 0.95, "90")):
        lo, hi = interval(q_pred, lo_q, hi_q, qs)
        out[f"coverage_{name}"] = float(np.mean((y_log >= lo) & (y_log <= hi)))
        out[f"width_{name}"] = float(np.mean(hi - lo))
        out[f"interval_score_{name}"] = float(np.mean(interval_score(y_log, lo, hi, 1 - int(name) / 100)))
    if nll_count is not None and np.isfinite(nll_count).any():
        out["nll_count"] = float(np.nanmean(nll_count))
    if baseline is not None:
        ok = ~np.isnan(baseline)
        out["baseline_mae"] = float(np.mean(np.abs(y_log[ok] - baseline[ok])))
        out["mae_median_same_rows"] = float(np.mean(np.abs(y_log[ok] - median[ok])))
    return out
