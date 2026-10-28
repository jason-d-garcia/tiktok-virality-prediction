"""Feedback on post variants: Claude when ANTHROPIC_API_KEY is set, rules otherwise.

Both paths use the same rubric (hook, caption, hashtags, length, thumbnail)
and are grounded in the predictor's drivers so the text agrees with the numbers.
"""

import base64
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from creator_studio.ranking import Ranked
from creator_studio.schema import PostPlan, Variant, fmt_views

MODEL = os.environ.get("CREATOR_STUDIO_MODEL", "claude-sonnet-5-5")


@dataclass
class Feedback:
    variant: str
    notes: list[str] = field(default_factory=list)   # what works / what doesn't
    edits: list[str] = field(default_factory=list)   # concrete changes to try


@dataclass
class Review:
    per_variant: dict[str, Feedback]
    comparison: str
    source: str  # "claude" or "rules"


# =========================
# ===== Rule-based ========
# =========================

def rule_feedback(v: Variant) -> Feedback:
    fb = Feedback(v.name)
    words = v.hook.split()
    if not words:
        fb.notes.append("No hook: the first seconds give viewers no reason to stay.")
        fb.edits.append("Open with a question or a promise of what they'll see.")
    elif len(words) > 12:
        fb.notes.append(f"Hook is {len(words)} words; viewers decide in 1 to 3 seconds.")
        fb.edits.append("Cut the hook to under 10 words, leading with the most surprising part.")
    elif "?" not in v.hook:
        fb.notes.append("Hook is short, which is good.")
        fb.edits.append("Try phrasing the hook as a question to create curiosity.")
    else:
        fb.notes.append("Hook is short and poses a question.")

    n = len(v.hashtags)
    if n == 0:
        fb.notes.append("No hashtags, so the video is harder to place for new viewers.")
        fb.edits.append("Add 3 to 6 hashtags mixing niche (#ballroomhair) and broad (#dance).")
    elif n > 8:
        fb.notes.append(f"{n} hashtags reads as spammy.")
        fb.edits.append("Trim to the 3 to 6 most relevant hashtags.")

    if not v.caption_text:
        fb.edits.append("Add a one-line caption that echoes the hook or invites comments.")
    elif "?" not in v.caption_text and "comment" not in v.caption_text.lower():
        fb.edits.append("End the caption with a light call to action, such as a question for comments.")

    if v.length_s > 60:
        fb.notes.append(f"{v.length_s:.0f} s is long; retention usually drops past a minute.")
        fb.edits.append("Test a cut under 30 s.")
    if v.thumbnail is None:
        fb.notes.append("No cover uploaded, so it wasn't reviewed.")
    return fb


def _comparison_text(ranked: list[Ranked]) -> str:
    top = [r for r in ranked if r.tier == 1]
    best = top[0].prediction
    lo, mid, hi = best.views
    vs = [r.prediction.variant for r in top]
    names = vs[0] if len(vs) == 1 else ", ".join(vs[:-1]) + " and " + vs[-1]
    if len(top) > 1:
        lead = f"{names} are effectively tied at the top"
    else:
        lead = f"{names} is the model's pick"
    return (f"{lead} (about {fmt_views(mid)} views, likely {fmt_views(lo)} to {fmt_views(hi)}). "
            "Differences smaller than about 1.4x are within the model's noise, "
            "so choose among tied versions on taste.")


def rule_review(plan: PostPlan, ranked: list[Ranked]) -> Review:
    return Review({v.name: rule_feedback(v) for v in plan.variants},
                  _comparison_text(ranked), "rules")


# =========================
# ===== Claude ============
# =========================

SYSTEM = """You are a short-form video coach reviewing draft TikTok posts for one creator.
Rubric: hook (first 1 to 3 seconds: curiosity, under ~10 words), caption (clear,
matches hook, light call to action), hashtags (3 to 6, niche plus broad),
length, and the cover image if one is attached.
Rules: never invent statistics. Attribute numbers to "the model". Variants in
the same tier are effectively tied; do not call a winner among them. Do not
suggest changes to what the video shows, only to how it is packaged.
Reply with JSON only:
{"variants": {"<name>": {"notes": [..2-4 short strings..], "edits": [..2-3 concrete edits..]}},
 "comparison": "<one short paragraph on which to post and why>"}"""


def _variant_block(v: Variant, r: Ranked) -> str:
    p = r.prediction
    lo, mid, hi = p.views
    drivers = "; ".join(f"{d} ({s:+.2f})" for d, s in p.drivers) or "none"
    return (f"Variant {v.name} (tier {r.tier}, rank {r.rank})\n"
            f"Hook: {v.hook!r}\nCaption: {v.caption!r}\nLength: {v.length_s:.0f} s\n"
            f"Model: ~{fmt_views(mid)} views, {int(p.coverage_target * 100)}% range "
            f"{fmt_views(lo)} to {fmt_views(hi)}, {p.vs_baseline:.2f}x the creator's typical post\n"
            f"Model drivers (log10 views): {drivers}")


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


def claude_review(plan: PostPlan, ranked: list[Ranked], client=None) -> Review:
    import anthropic
    client = client or anthropic.Anthropic()
    by_name = {r.prediction.variant: r for r in ranked}
    content = []
    for v in plan.variants:
        content.append({"type": "text", "text": _variant_block(v, by_name[v.name])})
        if v.thumbnail:
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": v.thumbnail_mime,
                "data": base64.b64encode(v.thumbnail).decode()}})
            content.append({"type": "text", "text": f"(cover image for variant {v.name})"})
    msg = client.messages.create(model=MODEL, max_tokens=2000, system=SYSTEM,
                                 messages=[{"role": "user", "content": content}])
    data = _parse_json("".join(b.text for b in msg.content if b.type == "text"))
    per = {name: Feedback(name, d.get("notes", []), d.get("edits", []))
           for name, d in data.get("variants", {}).items()}
    for v in plan.variants:  # fill any variant Claude skipped
        per.setdefault(v.name, rule_feedback(v))
    return Review(per, data.get("comparison", _comparison_text(ranked)), "claude")


def review(plan: PostPlan, ranked: list[Ranked], use_claude: Optional[bool] = None) -> Review:
    if use_claude is None:
        use_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if use_claude:
        try:
            return claude_review(plan, ranked)
        except Exception as e:  # network, auth, bad JSON: fall back, never break the app
            r = rule_review(plan, ranked)
            r.source = f"rules (Claude unavailable: {type(e).__name__})"
            return r
    return rule_review(plan, ranked)


# =========================
# ===== Rewrites ==========
# =========================

def suggest_hooks(v: Variant, n: int = 3, use_claude: Optional[bool] = None) -> list[str]:
    """Alternative hooks to add as new variants and score like the rest."""
    if use_claude is None:
        use_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if use_claude:
        try:
            import anthropic
            msg = anthropic.Anthropic().messages.create(
                model=MODEL, max_tokens=400,
                messages=[{"role": "user", "content":
                    f"Write {n} alternative TikTok hooks (under 10 words each) for this post. "
                    f"Same video, different packaging. Hook: {v.hook!r}. Caption: {v.caption!r}. "
                    'Reply with a JSON list of strings only.'}])
            text = "".join(b.text for b in msg.content if b.type == "text")
            hooks = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
            return [str(h) for h in hooks][:n]
        except Exception:
            pass
    topic = v.caption_text.split(".")[0][:60] or v.hook[:60] or "this"
    return [f"Why does nobody talk about {topic.lower()}?",
            f"Wait for the end: {topic.lower()}",
            f"3 things I wish I knew about {topic.lower()}"][:n]
