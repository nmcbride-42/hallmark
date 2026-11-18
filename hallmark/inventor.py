"""The proposer ("inventor") half: an LLM writes executable detectors, and a
predeclared rule decides whether any of them earned their place.

The LLM never sees probe identities, labels, or any per-target information.
It sees a description of what the current detectors miss, in aggregate, and
returns `score(time, flux) -> float` functions. Each runs in a restricted
namespace over every curve in a frozen, labelled probe set. Labels are applied
only after every score is in.

The success rule is declared BEFORE the model is called:

  (a) at least `min_hits` labelled positives in the transform's top-k, with an
      exact hypergeometric p-value reported, and
  (b) strictly more positives in its top-k than the baseline achieves on the
      identical probe set.

A transform that passes is not trusted because it passed; it is reviewed and
absorbed into `hallmark.detectors` as ordinary versioned code, and the next
proposer has to beat a baseline that contains it.

Sandbox caveat: `run_transform` restricts builtins and imports and applies a
wall-clock limit. That is a guard against accidents, not a security boundary.
Python `exec` cannot be made safe against adversarial code; run proposals from
a model you do not control inside a container or VM.
"""

from __future__ import annotations

import json
import math
import re
import signal
from dataclasses import dataclass
from math import comb

import numpy as np

# ── Restricted execution ───────────────────────────────────────────────────

FORBIDDEN_TOKENS = ("__", "open(", "eval(", "exec(", "compile(", "globals(",
                    "locals(", "getattr", "setattr", "delattr", "input(",
                    "exit(", "quit(", "import os", "import sys", "subprocess",
                    "socket", "urllib", "requests", "pathlib")
IMPORT_WHITELIST = {"math", "numpy"}


def _safe_import(name, *args, **kwargs):
    if name.split(".")[0] in IMPORT_WHITELIST:
        return __import__(name, *args, **kwargs)
    raise ImportError(f"import of {name!r} not allowed")


_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum, "pow": pow,
    "round": round, "len": len, "range": range, "float": float,
    "int": int, "bool": bool, "sorted": sorted, "map": map,
    "filter": filter, "zip": zip, "enumerate": enumerate, "all": all,
    "any": any, "list": list, "tuple": tuple, "set": set, "dict": dict,
    "reversed": reversed, "divmod": divmod, "isinstance": isinstance,
    "slice": slice, "print": lambda *a, **k: None,
    "Exception": Exception, "ValueError": ValueError,
    "ZeroDivisionError": ZeroDivisionError, "IndexError": IndexError,
    "__import__": _safe_import,
}


class TransformRejected(ValueError):
    pass


def compile_transform(code: str):
    """Static screen + compile. Returns the `score` callable."""
    for bad in FORBIDDEN_TOKENS:
        if bad in code:
            raise TransformRejected(f"forbidden token {bad!r}")
    env = {"__builtins__": dict(_BUILTINS), "np": np, "numpy": np,
           "math": math}
    exec(compile(code, "<proposal>", "exec"), env)          # noqa: S102
    fn = env.get("score")
    if not callable(fn):
        raise TransformRejected("no score(time, flux) defined")
    return fn


def run_transform(code: str, time_arr, flux_arr, seconds: float = 10.0) -> float:
    """Compile and run `score(time, flux)` under restrictions + wall clock.

    Unix only (uses SIGALRM). Raises on any failure; callers decide how a
    failure scores."""
    def _handler(signum, frame):
        raise TimeoutError("transform exceeded wall clock")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        fn = compile_transform(code)
        v = float(fn(np.asarray(time_arr, float).copy(),
                     np.asarray(flux_arr, float).copy()))
        if not math.isfinite(v):
            raise ValueError("non-finite score")
        return v
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


# ── Parsing model output ───────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json|python)?\s*\n(.*?)```", re.DOTALL)


def extract_proposals(text: str) -> list[dict]:
    """Accept either the requested JSON object or bare fenced code blocks.

    Returns [{"name", "principle", "code"}, ...]."""
    text = (text or "").strip()
    candidates = [text] + _FENCE_RE.findall(text)
    for c in candidates:
        try:
            payload = json.loads(c)
        except (json.JSONDecodeError, TypeError):
            continue
        props = payload.get("proposals") if isinstance(payload, dict) else None
        if isinstance(props, list):
            return [p for p in props
                    if isinstance(p, dict) and "def score" in str(p.get("code", ""))]
    out = []
    for i, block in enumerate(_FENCE_RE.findall(text)):
        if "def score" in block:
            out.append({"name": f"proposal_{i + 1}", "principle": "",
                        "code": block.strip()})
    return out


# ── Scoring against the predeclared rule ───────────────────────────────────

def hypergeom_p_at_least(k: int, N: int, K: int, n: int) -> float:
    """P(X >= k) when drawing n of N items, K of which are positives."""
    return sum(comb(K, i) * comb(N - K, n - i)
               for i in range(k, min(K, n) + 1)) / comb(N, n)


def rank_hits(scores: dict, labels: dict, k: int, positive: str = "eb"):
    order = sorted(scores, key=lambda t: -scores[t])
    top = order[:k]
    return sum(1 for t in top if labels[str(t)] == positive), top


@dataclass(frozen=True)
class SuccessRule:
    """Written down before the proposer is called. Changing it after seeing
    results is the one thing the design forbids."""
    top_k: int = 10
    min_hits: int = 7
    must_beat_baseline: bool = True
    min_scored_frac: float = 0.8     # a transform that crashes on >20% is out


def evaluate(proposals: list[dict], curves: dict, labels: dict,
             baseline_scores: dict | None, rule: SuccessRule,
             seconds: float = 10.0, positive: str = "eb") -> dict:
    """Score every proposal over every curve, then apply the rule.

    `curves`: {id: (time, flux)}; `labels`: {str(id): label}. Labels are read
    only after all scores for a proposal are computed."""
    ids = list(curves)
    n_pos = sum(1 for t in ids if labels[str(t)] == positive)

    base = None
    if baseline_scores is not None:
        b_hits, _ = rank_hits(baseline_scores, labels, rule.top_k, positive)
        base = {"hits": b_hits,
                "p": hypergeom_p_at_least(b_hits, len(ids), n_pos, rule.top_k)}

    results = []
    for prop in proposals:
        scores, errors = {}, []
        for t in ids:
            try:
                scores[t] = run_transform(prop["code"], *curves[t], seconds=seconds)
            except Exception as e:                       # noqa: BLE001
                scores[t] = float("-inf")
                errors.append(f"{type(e).__name__}: {str(e)[:60]}")
        n_ok = sum(1 for v in scores.values() if math.isfinite(v))
        row = {"name": prop.get("name", "?"),
               "principle": prop.get("principle", ""),
               "n_scored": n_ok, "n_errors": len(errors),
               "first_errors": errors[:2]}
        if n_ok < rule.min_scored_frac * len(ids):
            row.update(disqualified=True, success=False)
            results.append(row)
            continue
        hits, _ = rank_hits(scores, labels, rule.top_k, positive)
        beats = base is None or hits > base["hits"]
        row.update(
            hits=hits,
            p=hypergeom_p_at_least(hits, len(ids), n_pos, rule.top_k),
            beats_baseline=beats,
            success=bool(hits >= rule.min_hits
                         and (beats or not rule.must_beat_baseline)),
        )
        results.append(row)

    return {"n_curves": len(ids), "n_positive": n_pos, "rule": rule.__dict__,
            "baseline": base, "proposals": results,
            "any_success": any(r.get("success") for r in results)}


# ── Prompt ─────────────────────────────────────────────────────────────────

def build_prompt(n_proposals: int = 3) -> str:
    """The task, described in aggregate. No targets, no labels."""
    return f"""You are the proposer in a discovery-pipeline experiment over \
TESS light curves.

The current detector library already contains: a Lomb-Scargle peak, a
Lafler-Kinman string-length statistic, an autocorrelation secondary-peak
statistic, and a composite gap-masked box-dip SNR plus phase-folded ANOVA.

The target class is eclipsing binaries: PERIODIC variability, often SHALLOW
(well under 7 sigma per point), periods from about 0.05 to 100+ days,
including contact systems whose light varies continuously rather than in
discrete deep dips. Light curves contain data gaps, and dips adjacent to gaps
are usually spacecraft artifacts, not stars.

Write {n_proposals} DIFFERENT executable Python transforms. Each must define
exactly:

    def score(time, flux):
        # time: np.ndarray in days (may contain gaps); flux: np.ndarray
        # return one float; HIGHER = more likely a periodic variable
        ...

Rules: numpy (as np) and math only. No I/O, no other imports, no dunder
access. Each call must finish within ~10 seconds on ~30,000 points. Be robust
to NaNs (mask with np.isfinite) and to normalization (divide by the median).
Use genuinely different detection principles, not variants of one idea.

Return only JSON: {{"proposals": [{{"name": str, "principle": str (one line),
"code": str}}, ...]}}"""
