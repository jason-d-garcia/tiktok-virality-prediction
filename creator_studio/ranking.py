"""Rank variants into tiers that reflect the model's resolution."""

from dataclasses import dataclass

from creator_studio.schema import Prediction

# Differences below this (log10 views) are reported as ties: 0.15 ~ 1.4x.
DEFAULT_DELTA = 0.15


@dataclass
class Ranked:
    prediction: Prediction
    rank: int        # 1 = highest median
    tier: int        # 1 = top tier; same tier = effectively tied
    gap_to_best: float  # log10 views below the top variant


def assign_tiers(preds: list[Prediction], delta: float = DEFAULT_DELTA) -> list[Ranked]:
    """Sort by median; open a new tier when a variant falls more than `delta`
    below the first variant of the current tier."""
    order = sorted(preds, key=lambda p: p.median, reverse=True)
    out, tier, anchor = [], 0, None
    for i, p in enumerate(order, start=1):
        if anchor is None or anchor - p.median > delta:
            tier += 1
            anchor = p.median
        out.append(Ranked(p, i, tier, order[0].median - p.median))
    return out
