# Design notes

Decisions that shaped Hallmark, most of them learned by watching something fail.

## The proposer and the verifier never share a channel

The proposer (an LLM) writes code and commits to claims, such as a period. The
verifier is declared and hashed before the candidate is computed, and its
`verify()` takes only a frozen dataset path plus what the proposer committed
to. It never sees suite labels, catalogue status, or anything the proposer
said about why the candidate is interesting. Labels enter the validation run
only after every verdict has been written.

LLMs were also used as adversarial reviewers, two model lineages critiquing
each candidate's evidence for aliasing, spacecraft artifacts, and blending.
Those reviews were useful for deciding what to check next by hand. They were
recorded and never fed into promotion. If deciding a candidate needs a model's
opinion, the candidate is not promotable by definition (clause 6).

## The verifier tests recurrence, not depth

"There is a dip" is not the claim. "Eclipses recur on a fixed ephemeris" is.
Dips caused by data gaps, momentum dumps, or scattered light are tied to the
spacecraft's clock, not to an orbit, so they do not appear where a fitted
ephemeris predicts. All six adversarial nulls in the shipped suite are real
light curves that a detector flagged and that turned out to be gap artifacts.
A mechanical gap guard (drop events near gaps and segment edges) reproduces,
deterministically, what had previously been done by a vision model reading
plots.

## Period aliasing needs two constraints, not one

Phase dispersion alone cannot break the `P / k` degeneracy: if every eclipse
falls on a multiple of `P`, it also falls on a multiple of `P / 2`. Adding
cycle *completeness* (predicted epochs inside observed coverage that actually
have an event) kills the half-period alias but admits the double-period one,
because `2P` predicts perfectly while leaving half the real eclipses
unexplained. Scoring `min(completeness, explanation)` pins `P`. Both failure
modes were observed before the fix.

That applies to the leave-one-out verifier, which sees enough events to
score completeness. The cross-visit verifier (`southern-archival-ephemeris`)
does not: with two primary eclipses per visit it accepts 2P, and at some
precisions P/2, of a period that replicates. It confirms recurrence, not the
exact period. `tests/test_verifiers.py` pins that limit so it cannot quietly
change.

## INDETERMINATE fails closed

With fewer than four clean events, an ephemeris is underdetermined: any two
events admit `P = interval / k` for every integer `k`. The verifier says so
and the candidate does not promote. This is why the 52-day candidate in the
worked example could not be certified from one TESS visit. It needed a second
visit, not a looser verifier.

## A panel must be symmetric

Several verifiers run on every candidate, and a candidate replicates if any
one confirms. That is only safe if nulls face exactly the same panel, plus a
sweep over every period proposed anywhere in the suite. Otherwise "any of N
verifiers" becomes a soft path for a favoured object. The secondary-eclipse
exclusion in the southern-archival verifier follows the same rule: the
secondary's phase is removed from the control set only when the secondary
itself replicates in held-out data, so a null never earns the exclusion.

## One hash function

The first validation run rejected everything, including known positives,
because the freezer hashed arrays in sorted key order and the contract hashed
them in literal order. Two correct implementations that disagree are worse
than one, because the failure looks exactly like a fail-closed contract doing
its job. `hallmark/hashing.py` is the only place that decides.

## "We could not check" is not "it is not there"

Clause 3 needs a named catalogue, a query time, and a cone radius. A network
error produces a record without a hit count, which clause 3 treats as
incomplete. Catalogues move: VSX absorbed a large bulk load of Gaia DR3
variables, which raised the bar for "uncatalogued" considerably, and at least
one earlier candidate was catalogued by another team between detection and
submission. Novelty is re-queried immediately before any submission, never
reused from an earlier check.

## Check feasibility before declaring a verifier

The contract requires a predeclared verifier but does not ask whether that
verifier can resolve the claim. `hallmark/feasibility.py` is a deterministic
gate run first: are the readouts measured, does the dataset's regime overlap
the claim's, is there enough power, is the stratifier clean. Its example comes
from a different domain (a pharmacology hypothesis against LINCS L1000) to
make the point that the gate is domain-agnostic. A parked claim keeps a
generated description of the dataset that would settle it.

## Absorb, then raise the bar

When an LLM-written detector meets the predeclared rule, it is not trusted
because it passed. It is reviewed and versioned into `hallmark/detectors.py`
as ordinary code, and the next proposer has to beat a baseline that already
contains it. `BASELINE_VERSION` changes whenever the library does, the same
discipline as verifier spec hashes.

## Ground coverage depends on the sky

For targets far south (the worked example sits at Dec −58°), northern surveys
such as ZTF return no data at all. Long-period southern candidates need either
repeated TESS visits or southern ground surveys (ASAS-SN, ATLAS, SkyMapper).
The archival module queries Gaia DR3 and ASAS-SN for context, but only the
TESS multi-visit test is used as a verifier.
