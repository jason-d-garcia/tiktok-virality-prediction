"""Creator Studio prototype UI.

    pip install streamlit anthropic
    streamlit run app/creator_studio.py

Set ANTHROPIC_API_KEY for Claude feedback; without it, rule-based tips are used.
Predictions come from StubPredictor until the real model is plugged in.
"""

import sys
from datetime import datetime, time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from creator_studio import PostPlan, StubPredictor, Variant, assign_tiers  # noqa: E402
from creator_studio.feedback import review, suggest_hooks  # noqa: E402
from creator_studio.schema import fmt_views  # noqa: E402

MAX_VARIANTS = 6

st.set_page_config(page_title="Creator Studio", layout="wide")


@st.cache_resource
def get_predictor():
    return StubPredictor()


def blank(name: str) -> dict:
    return {"name": name, "hook": "", "caption": "", "length_s": 20.0, "thumb": None}


if "variants" not in st.session_state:
    st.session_state.variants = [blank("A"), blank("B")]
variants = st.session_state.variants

st.title("Creator Studio")
st.caption("Compare drafts of a post before you publish. Predictions are a **prototype stub** "
           "(your recent median plus simple heuristics) until the trained model is connected.")

c1, c2 = st.columns(2)
post_date = c1.date_input("Planned posting date", datetime.now().date())
post_time = c2.time_input("Planned posting time", time(19, 0))

for i, v in enumerate(variants):
    with st.expander(f"Variant {v['name']}", expanded=True):
        v["hook"] = st.text_input("Hook (first line said or shown)", v["hook"], key=f"hook{i}")
        v["caption"] = st.text_area("Caption with hashtags", v["caption"], key=f"cap{i}", height=80)
        a, b = st.columns(2)
        v["length_s"] = a.number_input("Length (s)", 1.0, 600.0, float(v["length_s"]), key=f"len{i}")
        up = b.file_uploader("Cover / thumbnail (optional)", ["png", "jpg", "jpeg"], key=f"img{i}")
        v["thumb"] = (up.getvalue(), up.type) if up else None
        if up:
            b.image(up, width=120)

b1, b2, _ = st.columns([1, 1, 4])
if b1.button("Add variant", disabled=len(variants) >= MAX_VARIANTS):
    src = dict(variants[-1], name=chr(ord("A") + len(variants)), thumb=None)
    variants.append(src)
    st.rerun()
if b2.button("Remove last", disabled=len(variants) <= 1):
    variants.pop()
    st.rerun()

if st.button("Score drafts", type="primary"):
    plan = PostPlan("emily", datetime.combine(post_date, post_time), [
        Variant(v["name"], v["caption"], v["hook"], v["length_s"],
                thumbnail=v["thumb"][0] if v["thumb"] else None,
                thumbnail_mime=v["thumb"][1] if v["thumb"] else "image/jpeg")
        for v in variants])
    pred = get_predictor()
    ranked = assign_tiers(pred.predict(plan))
    with st.spinner("Writing feedback..."):
        rev = review(plan, ranked)
    st.session_state.result = (plan, ranked, rev, pred.coverage_empirical)

if "result" in st.session_state:
    plan, ranked, rev, cov = st.session_state.result
    st.subheader("Ranking")
    st.info(rev.comparison)
    rows = []
    for r in ranked:
        p = r.prediction
        lo, mid, hi = p.views
        rows.append({"Rank": r.rank, "Tier": r.tier, "Variant": p.variant,
                     "Predicted views": fmt_views(mid),
                     f"{int(p.coverage_target * 100)}% range": f"{fmt_views(lo)} to {fmt_views(hi)}",
                     "vs your typical post": f"{p.vs_baseline:.2f}x"})
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(f"Variants in the same tier are effectively tied. Ranges are split-conformal "
               f"intervals targeting {int(ranked[0].prediction.coverage_target * 100)}% coverage; "
               f"on your older posts held out in time order they covered {cov:.0%}.")

    st.subheader("Feedback")
    st.caption(f"Source: {rev.source}")
    by_name = {v.name: v for v in plan.variants}
    for r in ranked:
        name = r.prediction.variant
        fb = rev.per_variant[name]
        with st.expander(f"Variant {name}", expanded=r.rank == 1):
            d = r.prediction.drivers
            if d:
                st.markdown("**What moves the prediction:** " +
                            ", ".join(f"{lbl} ({s:+.2f})" for lbl, s in d))
            for n in fb.notes:
                st.markdown(f"- {n}")
            if fb.edits:
                st.markdown("**Try:**")
                for e in fb.edits:
                    st.markdown(f"- {e}")
            if st.button("Suggest new hooks as variants", key=f"sug{name}",
                         disabled=len(variants) >= MAX_VARIANTS):
                for h in suggest_hooks(by_name[name]):
                    if len(variants) >= MAX_VARIANTS:
                        break
                    src = next(v for v in variants if v["name"] == name)
                    variants.append(dict(src, name=chr(ord("A") + len(variants)), hook=h, thumb=None))
                st.rerun()
