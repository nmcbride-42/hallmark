"""The absorbed detector library ranks; it does not decide."""

import numpy as np

from hallmark.detectors import (TRANSFORMS, combine_scores,
                                composite_eb_score, estimate_period)


def _load(frozen, key):
    d = np.load(frozen / f"{key}.npz")
    return d["time"], d["flux"]


def test_composite_ranks_positive_above_quiet_null(frozen):
    pos = composite_eb_score(*_load(frozen, "14069011_s92"))
    null = composite_eb_score(*_load(frozen, "6286534_s94"))
    assert pos > 3 * null


def test_estimate_period_on_a_detached_binary(frozen):
    est = estimate_period(*_load(frozen, "14069011_s92"))
    assert abs(est["period_d"] - 2.0894) / 2.0894 < 0.01


def test_transforms_survive_degenerate_input():
    t = np.linspace(0, 1, 10)
    for fn in TRANSFORMS:
        assert fn(t, np.ones(10)) == 0.0          # too short / flat -> 0
        assert np.isfinite(fn(t, np.full(10, np.nan)))


def test_combine_scores_takes_max_z():
    per = {"a": {"x": 1.0, "y": 2.0, "z": 3.0},
           "b": {"x": 10.0, "y": 0.0, "z": 0.0}}
    comb = combine_scores(per)
    assert max(comb, key=comb.get) == "x"
