"""Proposer plumbing: parsing, the restricted namespace, and the rule."""

import json
from pathlib import Path

import numpy as np
import pytest

from hallmark.inventor import (SuccessRule, TransformRejected, evaluate,
                               extract_proposals, hypergeom_p_at_least,
                               run_transform)

ROOT = Path(__file__).resolve().parents[1]

GOOD = "def score(time, flux):\n    return float(np.std(flux))\n"


def test_run_transform_basic():
    t = np.linspace(0, 10, 100)
    assert run_transform(GOOD, t, np.sin(t)) > 0


@pytest.mark.parametrize("code", [
    "import os\ndef score(time, flux):\n    return 0.0",
    "def score(time, flux):\n    return open('x').read()",
    "def score(time, flux):\n    return ().__class__",
    "def score(time, flux):\n    return getattr(np, 'std')(flux)",
])
def test_forbidden_code_is_rejected(code):
    with pytest.raises(TransformRejected):
        run_transform(code, np.zeros(3), np.zeros(3))


def test_non_whitelisted_import_fails():
    code = "def score(time, flux):\n    import json\n    return 0.0"
    with pytest.raises(ImportError):
        run_transform(code, np.zeros(3), np.zeros(3))


def test_wall_clock_limit():
    code = "def score(time, flux):\n    while True:\n        pass\n"
    with pytest.raises(TimeoutError):
        run_transform(code, np.zeros(3), np.zeros(3), seconds=0.2)


def test_non_finite_score_is_an_error():
    code = "def score(time, flux):\n    return float('nan')\n"
    with pytest.raises(ValueError):
        run_transform(code, np.zeros(3), np.zeros(3))


def test_extract_from_json_and_fences():
    payload = {"proposals": [{"name": "a", "principle": "p", "code": GOOD}]}
    assert extract_proposals(json.dumps(payload))[0]["name"] == "a"
    fenced = "here:\n```json\n" + json.dumps(payload) + "\n```"
    assert len(extract_proposals(fenced)) == 1
    bare = "```python\n" + GOOD + "```"
    assert extract_proposals(bare)[0]["code"].startswith("def score")
    assert extract_proposals("no code here") == []


def test_shipped_example_proposals_parse():
    text = (ROOT / "examples" / "first_round_proposals.json").read_text()
    props = extract_proposals(text)
    assert [p["name"] for p in props] == [
        "lomb_scargle_peak", "string_length_min", "acf_secondary_peak"]


def test_hypergeom_matches_reported_values():
    # 60 curves, 20 positives, top-10: the thresholds quoted in the README.
    assert hypergeom_p_at_least(7, 60, 20, 10) == pytest.approx(0.01155, abs=1e-4)
    assert hypergeom_p_at_least(8, 60, 20, 10) == pytest.approx(0.001395, abs=1e-5)
    assert hypergeom_p_at_least(9, 60, 20, 10) == pytest.approx(9.2e-5, rel=0.05)
    assert hypergeom_p_at_least(0, 60, 20, 10) == pytest.approx(1.0)


def _toy_set(n_pos=10, n_neg=20):
    """Positives are sinusoids, negatives are white noise (seeded)."""
    rng = np.random.default_rng(0)
    t = np.linspace(0, 20, 2000)
    curves, labels = {}, {}
    for i in range(n_pos):
        curves[f"p{i}"] = (t, 1 + 0.01 * np.sin(2 * np.pi * t / (1 + i * 0.3))
                           + rng.normal(0, 0.002, t.size))
        labels[f"p{i}"] = "eb"
    for i in range(n_neg):
        curves[f"n{i}"] = (t, 1 + rng.normal(0, 0.002, t.size))
        labels[f"n{i}"] = "ctl"
    return curves, labels


def test_rule_accepts_a_real_signal_and_rejects_noise():
    curves, labels = _toy_set()
    good = {"name": "std", "code": GOOD}
    noise = {"name": "const", "code": "def score(time, flux):\n    return 1.0\n"}
    crash = {"name": "crash", "code": "def score(time, flux):\n    return 1/0\n"}
    res = evaluate([good, noise, crash], curves, labels, baseline_scores=None,
                   rule=SuccessRule(top_k=10, min_hits=7))
    by = {r["name"]: r for r in res["proposals"]}
    assert by["std"]["success"] is True and by["std"]["hits"] == 10
    assert by["crash"]["disqualified"] is True and by["crash"]["success"] is False


def test_rule_requires_strictly_beating_the_baseline():
    curves, labels = _toy_set()
    perfect = {k: (1.0 if labels[k] == "eb" else 0.0) for k in curves}
    res = evaluate([{"name": "std", "code": GOOD}], curves, labels,
                   baseline_scores=perfect, rule=SuccessRule(top_k=10, min_hits=7))
    r = res["proposals"][0]
    assert r["hits"] == res["baseline"]["hits"] == 10
    assert r["beats_baseline"] is False and r["success"] is False
