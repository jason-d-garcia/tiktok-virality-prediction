"""Predictors for Creator Studio.

The real model plugs in by implementing `Predictor.predict`. Until then,
`StubPredictor` gives honest-but-simple numbers:

  median   = creator baseline (median log10 views of the previous K posts)
             + small hand-set heuristic adjustments (hook, hashtags, length,
             posting hour). The adjustments are placeholders, not learned.
  interval = split conformal on the stub's own residuals over the creator's
             history, in time order.
"""

import math
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import pandas as pd

from creator_studio.schema import PostPlan, Prediction, Variant

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = PROJECT_ROOT / "data" / "raw" / "tiktok_stats.csv"


class Predictor(Protocol):
    def predict(self, plan: PostPlan) -> list[Prediction]: ...


# =========================
# ===== History ===========
# =========================

def _parse_count(val) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower().replace(",", "")
    mult = 1.0
    if s.endswith("k"):
        mult, s = 1e3, s[:-1]
    elif s.endswith("m"):
        mult, s = 1e6, s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return np.nan


def _parse_seconds(val) -> float:
    if pd.isna(val):
        return np.nan
    m = re.match(r"([\d.]+)", str(val))
    return float(m.group(1)) if m else np.nan


def load_history(path: Path = DEFAULT_HISTORY) -> pd.DataFrame:
    """Creator history as posted_at, caption, length_s, views, log10_views."""
    raw = pd.read_csv(path)
    df = pd.DataFrame({
        "posted_at": pd.to_datetime(
            raw["published date"].astype(str) + " " + raw["published time"].astype(str),
            format="mixed", errors="coerce",
        ),
        "caption": raw["description"].fillna("").astype(str),
        "length_s": raw["video length"].apply(_parse_seconds),
        "views": raw["views"].apply(_parse_count),
    })
    df = df.dropna(subset=["posted_at", "views"]).sort_values("posted_at")
    df["log10_views"] = np.log10(df["views"].clip(lower=1))
    return df.reset_index(drop=True)


# =========================
# ===== Heuristics ========
# =========================

CURIOSITY_WORDS = {"why", "how", "what", "secret", "never", "nobody", "wait", "watch", "pov", "you"}


def heuristic_adjustments(hook: str, caption: str, length_s: float, hour: int) -> list[tuple[str, float]]:
    """Signed log10 nudges. Placeholders until the real model supplies SHAP."""
    out = []
    words = hook.lower().split()
    if not words:
        out.append(("No hook", -0.10))
    else:
        if "?" in hook or CURIOSITY_WORDS & {w.strip("?!.,") for w in words} or any(c.isdigit() for c in hook):
            out.append(("Hook creates curiosity", +0.05))
        if len(words) > 12:
            out.append(("Hook is long (>12 words)", -0.05))

    n_tags = caption.count("#")
    if n_tags == 0:
        out.append(("No hashtags", -0.05))
    elif 3 <= n_tags <= 6:
        out.append(("3 to 6 hashtags", +0.03))
    elif n_tags > 8:
        out.append(("Too many hashtags (>8)", -0.05))

    if not math.isnan(length_s):
        if 7 <= length_s <= 30:
            out.append(("Length 7 to 30 s", +0.03))
        elif length_s > 60:
            out.append(("Over 60 s", -0.05))

    if 17 <= hour <= 22:
        out.append(("Evening posting slot", +0.03))
    elif 1 <= hour <= 6:
        out.append(("Late-night posting slot", -0.03))
    return out


def _first_sentence(text: str) -> str:
    text = re.sub(r"#\w+", "", text).strip()
    return re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0] if text else ""


def conformal_quantile(residuals: np.ndarray, coverage: float) -> float:
    """Split-conformal quantile of |residual| with the finite-sample correction."""
    r = np.sort(np.abs(residuals))
    n = len(r)
    k = math.ceil((n + 1) * coverage)
    return float(r[min(k, n) - 1])


# =========================
# ===== Stub predictor ====
# =========================

class StubPredictor:
    def __init__(self, history: Optional[pd.DataFrame] = None, k: int = 10,
                 coverage: float = 0.8, calib_frac: float = 0.6):
        self.history = load_history() if history is None else history
        self.k = k
        self.coverage = coverage
        self._calibrate(calib_frac)

    def _point(self, prior_log_views: np.ndarray, hook: str, caption: str,
               length_s: float, hour: int) -> tuple[float, float, list[tuple[str, float]]]:
        baseline = float(np.median(prior_log_views[-self.k:]))
        drivers = heuristic_adjustments(hook, caption, length_s, hour)
        return baseline + sum(d for _, d in drivers), baseline, drivers

    def _calibrate(self, calib_frac: float) -> None:
        """Residuals of one-step-ahead predictions over history, in time order."""
        h = self.history
        y = h["log10_views"].to_numpy()
        res = []
        for i in range(self.k, len(h)):
            row = h.iloc[i]
            pred, _, _ = self._point(y[:i], _first_sentence(row.caption), row.caption,
                                     row.length_s, row.posted_at.hour)
            res.append(y[i] - pred)
        res = np.array(res)
        if len(res) < 10:
            raise ValueError(f"need at least {self.k + 10} history posts, got {len(h)}")

        n_cal = int(len(res) * calib_frac)
        q_cal = conformal_quantile(res[:n_cal], self.coverage)
        self.coverage_empirical = float(np.mean(np.abs(res[n_cal:]) <= q_cal))
        # Deploy with every residual; the held-out check above is the honesty test.
        self.q = conformal_quantile(res, self.coverage)
        self.residuals = res

    def predict(self, plan: PostPlan) -> list[Prediction]:
        h = self.history
        out = []
        for v in plan.variants:
            t = plan.time_for(v)
            prior = h.loc[h["posted_at"] < t, "log10_views"].to_numpy()
            if len(prior) == 0:
                prior = h["log10_views"].to_numpy()
            med, base, drivers = self._point(prior, v.hook, v.caption, v.length_s, t.hour)
            out.append(Prediction(
                variant=v.name, median=med, lower=med - self.q, upper=med + self.q,
                baseline=base, coverage_target=self.coverage,
                coverage_empirical=self.coverage_empirical, drivers=drivers,
            ))
        return out
