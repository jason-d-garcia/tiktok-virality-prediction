"""Creator Studio tests. Run: python -m pytest tests/test_creator_studio.py"""

from datetime import datetime

import numpy as np

from creator_studio import PostPlan, StubPredictor, Variant, assign_tiers
from creator_studio.feedback import review, suggest_hooks
from creator_studio.predictor import conformal_quantile
from creator_studio.schema import Prediction


def _pred(name, m):
    return Prediction(name, m, m - 0.5, m + 0.5, 5.0, 0.8)


def test_tiers_group_close_variants():
    ranked = assign_tiers([_pred("a", 5.0), _pred("b", 5.1), _pred("c", 4.7)], delta=0.15)
    assert [r.prediction.variant for r in ranked] == ["b", "a", "c"]
    assert [r.tier for r in ranked] == [1, 1, 2]


def test_conformal_quantile_covers():
    rng = np.random.default_rng(0)
    q = conformal_quantile(rng.normal(size=500), 0.8)
    assert np.mean(np.abs(rng.normal(size=5000)) <= q) > 0.75


def _plan():
    return PostPlan("emily", datetime(2026, 11, 1, 19), [
        Variant("A", "Hair routine #ballroomdance #hair #asmr", "Why do dancers soak their hair?", 25),
        Variant("B", "Hair routine", "Here is my whole competition day routine from start to finish today", 75),
    ])


def test_stub_predicts_and_ranks():
    sp = StubPredictor()
    preds = sp.predict(_plan())
    assert all(p.lower < p.median < p.upper for p in preds)
    assert 0.5 < sp.coverage_empirical <= 1.0
    assert assign_tiers(preds)[0].prediction.variant == "A"


def test_rule_review_without_api_key():
    plan = _plan()
    r = review(plan, assign_tiers(StubPredictor().predict(plan)), use_claude=False)
    assert r.source == "rules"
    assert set(r.per_variant) == {"A", "B"}
    assert r.per_variant["B"].edits
    assert len(suggest_hooks(plan.variants[0], use_claude=False)) == 3
