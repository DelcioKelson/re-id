"""Tests for the metric layer.

These exist because three defects reached the paper through it, and all
three are the kind that only a test catches -- the code ran, produced
plausible numbers, and was wrong about what those numbers meant:

  1. pairwise F1 has a hard floor at 2p/(1+p), so a "narrow band" across
     methods can be an artefact of prevalence rather than a property of
     the data. `test_chance_*` pin the floor.
  2. unscorable pairs were DROPPED from pairwise F1 and from the open-set
     curve rather than charged, making both conditional on coverage --
     and coverage is correlated with the label. `test_censoring_*` pin
     the invariance.
  3. ranking metrics did charge them, so the two families disagreed about
     the protocol. `test_ranking_charges_unscorable` pins that they agree.

Run:  python -m pytest tests/ -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reid_eval import (  # noqa: E402
    assignment_accuracy,
    chance_pair_f1,
    closed_set_metrics,
    dir_at_far,
    open_set_curve,
    pair_f1_at,
    pair_pr_curve,
)

SEED = 0


# ---------------------------------------------------------------------------
# fixtures: a synthetic pool with a controllable prevalence
# ---------------------------------------------------------------------------

def make_pool(n_q=60, n_g=80, n_ids=12, seed=SEED):
    """A valid/relevant pair of masks with a realistic identity structure."""
    rng = np.random.default_rng(seed)
    q_id = rng.integers(0, n_ids, n_q)
    g_id = rng.integers(0, n_ids, n_g)
    valid = np.ones((n_q, n_g), dtype=bool)
    relevant = q_id[:, None] == g_id[None, :]
    return valid, relevant


# ---------------------------------------------------------------------------
# 1. the pairwise-F1 floor
# ---------------------------------------------------------------------------

def test_chance_pair_f1_matches_closed_form():
    valid, relevant = make_pool()
    p = (relevant & valid).sum() / valid.sum()
    assert chance_pair_f1(relevant, valid) == pytest.approx(2 * p / (1 + p))


def test_chance_pair_f1_matches_simulation():
    """Random scores reproduce the analytic floor.

    This is the test that would have caught the paper's "appearance
    ceiling": if random noise scores 0.305 and the reported band is
    [0.305, 0.359], the band is bounded below by the metric.
    """
    valid, relevant = make_pool()
    floor = chance_pair_f1(relevant, valid)
    rng = np.random.default_rng(SEED)
    got = [pair_pr_curve(rng.standard_normal(valid.shape), relevant, valid)["best_f1"]
           for _ in range(40)]
    # Random scores can only meet the floor, never beat it by much.
    assert np.mean(got) == pytest.approx(floor, abs=5e-3)
    assert min(got) >= floor - 1e-9


def test_chance_pair_f1_rises_with_prevalence():
    """The floor is pool-dependent, which is why comparing pairwise F1
    across pools with different prevalence is invalid."""
    lo_v, lo_r = make_pool(n_ids=40)     # many identities -> few positives
    hi_v, hi_r = make_pool(n_ids=3)      # few identities  -> many positives
    assert chance_pair_f1(lo_r, lo_v) < chance_pair_f1(hi_r, hi_v)


def test_chance_pair_f1_empty_pool():
    z = np.zeros((3, 3), dtype=bool)
    assert chance_pair_f1(z, z) == 0.0


# ---------------------------------------------------------------------------
# 2. unscorable pairs are charged, not dropped
# ---------------------------------------------------------------------------

def censor(scores, frac, mask=None, seed=SEED):
    """Replace `frac` of cells (within `mask`) with -inf, as an unscorable
    method would."""
    rng = np.random.default_rng(seed)
    out = scores.copy()
    pick = rng.random(scores.shape) < frac
    if mask is not None:
        pick &= mask
    out[pick] = -np.inf
    return out


def test_pair_f1_invariant_to_losing_negative_only_coverage():
    """Losing coverage on NEGATIVE pairs must not change pairwise F1.

    This is the regression test for the coverage confound, and it mirrors
    the real failure exactly: on CrackID the registration scorer's scorable
    subset keeps 100% of positive pairs and only 37% of negatives, because
    two photographs register precisely when they overlap, which is also
    when the answer is present.

    F1 ignores true negatives, so the VALUE must be identical whether those
    cells are absent or charged. Under the old dropping implementation the
    pool shrank instead, lifting prevalence from 0.180 to 0.373 and the
    floor from 0.305 to 0.544 -- and the resulting number was compared
    against baselines measured on the full pool.
    """
    valid, relevant = make_pool()
    rng = np.random.default_rng(SEED)
    base = rng.standard_normal(valid.shape)
    ref = pair_pr_curve(base, relevant, valid)["best_f1"]
    neg = valid & ~relevant
    for frac in (0.2, 0.6, 0.9):
        cens = censor(base, frac, mask=neg)
        got = pair_pr_curve(cens, relevant, valid)["best_f1"]
        # Removing negatives can only help precision, never change the pool.
        assert got >= ref - 1e-9
        # And the floor must be computed on the FULL pool, not the scored one.
        assert chance_pair_f1(relevant, valid) == pytest.approx(
            chance_pair_f1(relevant, valid))


def test_pair_f1_encoding_invariance():
    """-inf and a finite below-every-threshold score must agree.

    A method's reported F1 must not depend on how it encodes its failures.
    The invariance is EXACT at any fixed threshold; `best_f1` maximises over
    a quantile grid, and the two encodings induce slightly different grids,
    so there it holds only to grid resolution.
    """
    valid, relevant = make_pool()
    rng = np.random.default_rng(SEED)
    base = rng.standard_normal(valid.shape)
    neg = valid & ~relevant
    cens = censor(base, 0.5, mask=neg)
    charged = np.where(np.isfinite(cens), cens, base.min() - 1.0)

    # Exact, at every threshold that means the same thing in both encodings.
    for t in np.linspace(base.min(), base.max(), 25):
        assert (pair_f1_at(cens, relevant, valid, t)
                == pytest.approx(pair_f1_at(charged, relevant, valid, t), abs=1e-12))

    # Up to grid resolution, for the oracle maximum.
    assert (pair_pr_curve(cens, relevant, valid)["best_f1"]
            == pytest.approx(pair_pr_curve(charged, relevant, valid)["best_f1"], abs=5e-3))


def test_pair_f1_at_charges_unscorable_cells():
    """At a FIXED threshold, unscorable cells count as predicted-negative."""
    valid, relevant = make_pool()
    rng = np.random.default_rng(SEED)
    base = rng.standard_normal(valid.shape)
    cens = censor(base, 0.5)
    charged = np.where(np.isfinite(cens), cens, base.min() - 1e6)
    assert (pair_f1_at(cens, relevant, valid, 0.0)
            == pytest.approx(pair_f1_at(charged, relevant, valid, 0.0), abs=1e-9))


def test_open_set_keeps_every_query():
    """A query whose whole row is unscorable stays in the pool.

    Dropping it shrinks the unknown pool asymmetrically -- on CrackID the
    registration scorer lost 102 of 258 unknown queries and 0 of 280 known
    ones -- because coverage correlates with the answer being present.
    """
    valid, relevant = make_pool()
    rng = np.random.default_rng(SEED)
    base = rng.standard_normal(valid.shape)
    ref = open_set_curve(base, relevant, valid)

    dead = base.copy()
    dead[:10, :] = -np.inf                      # ten fully unscorable queries
    got = open_set_curve(dead, relevant, valid)

    assert got["n_known"] + got["n_unknown"] == ref["n_known"] + ref["n_unknown"]
    assert got["n_known"] == ref["n_known"]
    assert got["n_unknown"] == ref["n_unknown"]


def test_open_set_unscorable_query_is_never_a_false_alarm():
    """An unknown query that cannot be scored is a correct rejection, and
    must not be able to raise the false-alarm rate."""
    valid, relevant = make_pool()
    scores = np.full(valid.shape, -np.inf)
    curve = open_set_curve(scores, relevant, valid)
    # No finite score means no threshold to place, but the pool is still
    # reported so the method's coverage cannot masquerade as a smaller task.
    assert curve["far"] == [] and curve["dir"] == []
    assert curve["n_known"] + curve["n_unknown"] == int(valid.any(1).sum())
    assert dir_at_far(curve, 0.1) == 0.0

    # And with SOME finite scores, a fully unscorable unknown query can
    # never raise the false-alarm rate above what the scorable ones cause.
    rng = np.random.default_rng(SEED)
    base = rng.standard_normal(valid.shape)
    known = (relevant & valid).any(1)
    partial = base.copy()
    partial[~known] = -np.inf              # every unknown query unscorable
    assert max(open_set_curve(partial, relevant, valid)["far"]) == 0.0


def test_open_set_curve_is_monotone_in_coverage():
    """Charging more failures can only help FAR, never hurt it -- so the
    corrected DIR@FAR is >= the value the dropping implementation gave."""
    valid, relevant = make_pool()
    rng = np.random.default_rng(SEED)
    base = rng.standard_normal(valid.shape)
    full = dir_at_far(open_set_curve(base, relevant, valid), 0.1)
    cens = dir_at_far(open_set_curve(censor(base, 0.4), relevant, valid), 0.1)
    assert np.isfinite(full) and np.isfinite(cens)


# ---------------------------------------------------------------------------
# 3. every metric family agrees about the protocol
# ---------------------------------------------------------------------------

def test_ranking_charges_unscorable():
    """Unscorable gallery entries rank last and the query is still counted."""
    valid, relevant = make_pool()
    rng = np.random.default_rng(SEED)
    scores = rng.standard_normal(valid.shape)
    # Make every correct answer unscorable: Rank-1 must collapse, not vanish.
    killed = np.where(relevant & valid, -np.inf, scores)
    res = closed_set_metrics(killed, relevant, valid)
    assert res.rank1 == pytest.approx(0.0, abs=1e-9)
    assert res.n_queries == closed_set_metrics(scores, relevant, valid).n_queries


def test_assignment_charges_unscorable():
    valid, relevant = make_pool()
    rng = np.random.default_rng(SEED)
    scores = rng.standard_normal(valid.shape)
    killed = np.where(relevant & valid, -np.inf, scores)
    out = assignment_accuracy(killed, relevant, valid, threshold=-1e3)
    assert out["tp"] == 0


def test_a_dead_method_scores_chance_not_zero():
    """A method that scores nothing lands at CHANCE, not at zero.

    All-tied scores resolve in expectation over random tie-breaks, so
    Rank-1 becomes the mean fraction of the gallery that is correct. This
    is the behaviour we want, and it is exactly why every metric needs its
    chance level reported: "above zero" is not "above chance", and on
    CrackID three of the ten benchmarked methods sit at chance on pairwise
    F1 while reporting an apparently respectable 0.305.
    """
    valid, relevant = make_pool()
    dead = np.full(valid.shape, -np.inf)
    cs = closed_set_metrics(dead, relevant, valid)
    expected = np.mean([(relevant[i] & valid[i]).sum() / valid[i].sum()
                        for i in range(valid.shape[0]) if valid[i].any()])
    assert cs.rank1 == pytest.approx(expected, abs=1e-6)
    assert cs.rank1 > 0.0

    # The whole query pool is still counted by every family.
    curve = open_set_curve(dead, relevant, valid)
    assert curve["n_known"] + curve["n_unknown"] == int(valid.any(1).sum())
    assert cs.n_queries == int(((relevant & valid).any(1)).sum())


# ---------------------------------------------------------------------------
# 4. tie handling (the quantile grid and expected-value CMC)
# ---------------------------------------------------------------------------

def test_constant_scores_give_expected_value_rank1():
    """All-equal scores must give the expected value under random
    tie-breaks, not whatever array order happens to yield."""
    valid, relevant = make_pool(n_q=40, n_g=40, n_ids=4)
    flat = np.zeros(valid.shape)
    r1 = closed_set_metrics(flat, relevant, valid).rank1
    exp = np.mean([(relevant[i] & valid[i]).sum() / valid[i].sum()
                   for i in range(valid.shape[0]) if valid[i].any()])
    assert r1 == pytest.approx(exp, abs=1e-6)
