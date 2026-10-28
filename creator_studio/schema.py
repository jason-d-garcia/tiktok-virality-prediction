"""Data types shared by the predictor, ranking, feedback and UI."""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

HASHTAG_RE = re.compile(r"#(\w+)")


@dataclass
class Variant:
    name: str
    caption: str
    hook: str
    length_s: float
    posted_at: Optional[datetime] = None   # None -> use the plan's time
    thumbnail: Optional[bytes] = None      # raw image bytes (png/jpeg)
    thumbnail_mime: str = "image/jpeg"

    @property
    def hashtags(self) -> list[str]:
        return [h.lower() for h in HASHTAG_RE.findall(self.caption)]

    @property
    def caption_text(self) -> str:
        """Caption with hashtags removed."""
        return HASHTAG_RE.sub("", self.caption).strip()


@dataclass
class PostPlan:
    creator: str
    posted_at: datetime
    variants: list[Variant]

    def time_for(self, v: Variant) -> datetime:
        return v.posted_at or self.posted_at


@dataclass
class Prediction:
    """Predicted views on a log10 scale with a conformal interval."""
    variant: str
    median: float               # log10 views
    lower: float                # log10 views
    upper: float                # log10 views
    baseline: float             # creator's typical post, log10 views
    coverage_target: float      # e.g. 0.8
    coverage_empirical: Optional[float] = None  # on held-out history
    drivers: list[tuple[str, float]] = field(default_factory=list)  # (label, log10 delta)

    @property
    def views(self) -> tuple[float, float, float]:
        return 10 ** self.lower, 10 ** self.median, 10 ** self.upper

    @property
    def vs_baseline(self) -> float:
        """Multiplier vs the creator's typical post."""
        return 10 ** (self.median - self.baseline)


def fmt_views(x: float) -> str:
    if x >= 1e6:
        return f"{x / 1e6:.1f}M"
    if x >= 1e3:
        return f"{x / 1e3:.0f}K"
    return f"{x:.0f}"
