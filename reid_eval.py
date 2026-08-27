"""
Evaluation harness for crack re-identification.

The point of this file is that every method - registration-based,
keypoint-based, or embedding-based - is forced through the SAME
interface and scored by the SAME code. Each method only has to produce
a (n_query, n_gallery) matrix of similarity scores, higher = more
likely to be the same physical crack. Everything downstream (ranking,
CMC, mAP, open-set rejection, threshold calibration) is shared, so no
method can accidentally get a favourable protocol.

Two things this handles that a naive harness gets wrong:

  * MANY COMPONENTS -> ONE IDENTITY. A crack that is continuous in one
    photo often splits into several connected components in another. A
    query is correct if it retrieves ANY component of the right
    identity, and mAP counts all of them as relevant.

  * OPEN SET. Cracks appear and disappear between sessions. Queries with
    no correct answer in the gallery are kept, not discarded, and scored
    on whether the method correctly rejects them.

Requires: numpy, scipy
"""

from __future__ import annotations

import json
import time
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from collections import defaultdict


# ===========================================================================
# Data model
# ===========================================================================

@dataclass(frozen=True)
class InstanceRef:
    """One connected component in one photo."""
    instance_id: str       # e.g. "wall03_s1_img012_c02"
    image_id: str          # e.g. "wall03_s1_img012"
    wall_id: str           # e.g. "wall03"
    session: str           # e.g. "2026-08-10-morning"
    identity: str | None   # e.g. "wall03_crack01"; None = unlabelled


def load_instances(path: str) -> list[InstanceRef]:
    with open(path) as f:
        return [InstanceRef(**row) for row in json.load(f)]


# ===========================================================================
# The interface every method must implement
# ===========================================================================

class ReIDScorer(ABC):
    """Subclass once per method. Only score_matrix is required."""

    name: str = "unnamed"

    #: Declared so the paper can report it honestly rather than burying it.
    #: "crop"       - sees only the cropped instance
    #: "crop+ctx"   - sees the crop plus surrounding wall
    #: "full-image" - sees the entire photo (registration methods)
    input_scope: str = "crop"

    def prepare(self, refs: list[InstanceRef], data) -> None:
        """Optional one-off precomputation (e.g. embed every crop once)."""
        return None

    @abstractmethod
    def score_matrix(self, queries: list[InstanceRef],
                     gallery: list[InstanceRef], data) -> np.ndarray:
        """Return (n_query, n_gallery) scores. Higher = more similar.

        Use -np.inf for pairs the method structurally cannot score (e.g.
        registration failed). Do NOT drop those pairs: a method that
        silently skips its hard cases will look better than it is.
        """
        ...


class DistanceScorer(ReIDScorer):
    """Convenience base for methods that produce a distance, not a score.

    Ranking metrics need a total order over the gallery, so a method that
    natively outputs a hard assignment (Hungarian) must expose the
    underlying distance here and keep the assignment for the separate
    one-to-one evaluation below.
    """

    def score_matrix(self, queries, gallery, data) -> np.ndarray:
        d = self.distance_matrix(queries, gallery, data)
        with np.errstate(divide="ignore"):
            return -d

    @abstractmethod
    def distance_matrix(self, queries, gallery, data) -> np.ndarray:
        ...


# ===========================================================================
# Protocol: which gallery entries are valid for which query
# ===========================================================================

def frame_index(image_id: str) -> int | None:
    """Position of a photo within its wall's capture sequence.

    Photos are named <wall>_<s?>_<NNNN> in capture order, so the trailing
    number is the frame index. Returns None if the name does not carry one.
    """
    tail = image_id.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else None


def build_validity_mask(queries: list[InstanceRef], gallery: list[InstanceRef],
                        exclude_same_session: bool = True,
                        same_wall_only: bool = True,
                        min_frame_gap: int = 0) -> np.ndarray:
    """Boolean (n_query, n_gallery): True where the pair may be compared.

    exclude_same_session: a query must be matched against a DIFFERENT
        capture session. Comparing two components from the same session
        (let alone the same photo) is a trivially easy pair and inflates
        every metric.

    same_wall_only: restrict the gallery to the query's own wall. This is
        the honest setting - the hard negatives are cracks on the same
        wall in the same material under the same light. Set False to also
        report the easier cross-wall gallery.

    min_frame_gap: drop gallery photos within this many frames of the
        query's own frame. This exists because each wall was captured in
        one continuous pass, and measurement showed that for 100% of
        answerable queries -- in BOTH splits -- the nearest correct answer
        is the immediately adjacent frame, a median of one frame away.
        At min_frame_gap=0 the benchmark therefore reports near-duplicate
        retrieval across roughly three seconds, not re-identification.

        Sweeping 0, 1, 2, 3 turns that confound into a controlled
        difficulty axis: R@1 against baseline separation is a far more
        informative result than a single number, and it is the first thing
        a reviewer will ask for. Costs nothing to compute -- the saved
        score matrices already carry image_id.
    """
    valid = np.ones((len(queries), len(gallery)), dtype=bool)
    q_frame = [frame_index(q.image_id) for q in queries]
    g_frame = [frame_index(g.image_id) for g in gallery]
    for i, q in enumerate(queries):
        for j, g in enumerate(gallery):
            if g.image_id == q.image_id:
                valid[i, j] = False
            elif exclude_same_session and g.session == q.session:
                valid[i, j] = False
            elif same_wall_only and g.wall_id != q.wall_id:
                valid[i, j] = False
            elif (min_frame_gap > 0 and q.wall_id == g.wall_id
                  and q_frame[i] is not None and g_frame[j] is not None
                  and abs(q_frame[i] - g_frame[j]) <= min_frame_gap):
                valid[i, j] = False
    return valid


def build_relevance(queries: list[InstanceRef], gallery: list[InstanceRef]) -> np.ndarray:
    """Boolean (n_query, n_gallery): True where the pair is the same identity."""
    rel = np.zeros((len(queries), len(gallery)), dtype=bool)
    for i, q in enumerate(queries):
        if q.identity is None:
            continue
        for j, g in enumerate(gallery):
            rel[i, j] = (g.identity == q.identity)
    return rel


# ===========================================================================
# Metrics
# ===========================================================================

@dataclass
class ClosedSetResult:
    n_queries: int
    rank1: float
    rank5: float
    mAP: float
    cmc: list


def _average_ranks(s_desc: np.ndarray) -> np.ndarray:
    """1-based ranks for an already descending-sorted score vector, with
    exact ties sharing the mean of the ranks they span.

    [9, 5, 5, 5, 1] -> [1, 3, 3, 3, 5]

    Non-finite scores (a method that structurally cannot answer a pair)
    are all mutually tied at the bottom, which is the honest treatment:
    they carry no ordering information between themselves.
    """
    n = len(s_desc)
    ranks = np.arange(1, n + 1, dtype=float)
    if n == 0:
        return ranks
    key = np.where(np.isfinite(s_desc), s_desc, -np.inf)
    # group boundaries: a new group starts wherever the score changes
    new = np.empty(n, dtype=bool)
    new[0] = True
    new[1:] = key[1:] != key[:-1]
    start = np.flatnonzero(new)
    stop = np.append(start[1:], n)
    out = np.empty(n, dtype=float)
    for a, b in zip(start, stop):
        out[a:b] = ranks[a:b].mean()
    return out


def _expected_cmc(s_desc: np.ndarray, r_desc: np.ndarray, max_rank: int) -> np.ndarray:
    """CMC contribution of one query, in expectation over random tie-breaks.

    A method that scores several gallery entries EXACTLY equal has not
    ranked them; it has declined to. Crediting or denying it rank-1 based
    on their list order measures the list, not the method. The honest
    quantity is the probability that a correct entry lands within the top
    k once the tie is broken uniformly at random.

    Let the first tie group containing a correct entry span ranks
    (a, a+m], with c of its m members correct (every entry above it is,
    by construction, incorrect). Drawing without replacement,

        P(no correct entry in the first k of the group)
            = prod_{i<k} (m - c - i) / (m - i)

    so P(hit by rank a+k) is one minus that. Exact, and computed as a
    running product so it stays stable for large tie groups.
    """
    out = np.zeros(max_rank)
    key = np.where(np.isfinite(s_desc), s_desc, -np.inf)
    n = len(key)
    start = 0
    while start < n:
        stop = start + 1
        while stop < n and key[stop] == key[start]:
            stop += 1
        grp = r_desc[start:stop]
        if grp.any():
            a, m, c = start, stop - start, int(grp.sum())
            miss = 1.0
            for k in range(1, min(max_rank - a, m) + 1):
                if k <= m - c:
                    miss *= (m - c - (k - 1)) / (m - (k - 1))
                else:
                    miss = 0.0
                out[a + k - 1] = 1.0 - miss
            if a + m < max_rank:
                out[a + m:] = 1.0
            break
        start = stop
    return out


def closed_set_metrics(scores: np.ndarray, relevant: np.ndarray,
                       valid: np.ndarray, max_rank: int = 20) -> ClosedSetResult:
    """CMC and mAP over queries that have at least one correct answer.

    Handles multiple relevant gallery entries per query, which is the
    normal case here because one physical crack can be split across
    several components in the gallery photo.
    """
    n_q, n_g = scores.shape
    cmc_hits = np.zeros(min(max_rank, n_g))
    aps, counted = [], 0

    for i in range(n_q):
        mask = valid[i]
        if not mask.any():
            continue
        rel_i = relevant[i] & mask
        if not rel_i.any():
            continue                      # open-set query, handled separately

        s = scores[i][mask]
        r = rel_i[mask]

        order = np.argsort(-s, kind="stable")
        r_sorted = r[order]
        s_sorted = s[order]

        # TIE HANDLING. Several methods score in integer inlier counts, so a
        # query's row is mostly exact ties (measured: 94% of ORB's valid pairs
        # score exactly 0.0, and the median query has 30 gallery entries tied
        # for first). A stable argsort then ranks those ties by gallery index,
        # which is not a measurement -- it is an artefact of instance ordering,
        # and a deterministic one, so it does not average out.
        #
        # Instead every member of a tie group is charged the group's AVERAGE
        # rank, which is what that method has actually earned: it cannot
        # separate them, so it gets the expected result over a random
        # tie-break rather than whatever the list order happens to give.
        eff_rank = _average_ranks(s_sorted)

        if r_sorted.any():
            cmc_hits += _expected_cmc(s_sorted, r_sorted, len(cmc_hits))

        # Average precision with multiple relevant items. Precision is
        # evaluated at the tie-averaged rank for the same reason.
        cum_hits = np.cumsum(r_sorted)
        precision_at_hits = (cum_hits / eff_rank)[r_sorted]
        aps.append(float(precision_at_hits.mean()) if len(precision_at_hits) else 0.0)
        counted += 1

    if counted == 0:
        return ClosedSetResult(0, 0.0, 0.0, 0.0, [])

    cmc = (cmc_hits / counted).tolist()
    return ClosedSetResult(
        n_queries=counted,
        rank1=float(cmc[0]),
        rank5=float(cmc[min(4, len(cmc) - 1)]),
        mAP=float(np.mean(aps)),
        cmc=cmc,
    )


def _threshold_grid(s: np.ndarray, n: int = 200) -> np.ndarray:
    """Thresholds placed by QUANTILE of the observed scores, not linearly
    across their range.

    A linear grid assumes the scores are roughly uniform over [min, max].
    They never are here, and for one method the assumption is fatal:
    registration+chamfer scores are negated Chamfer distance capped at
    1e4 px, while all of its discriminative power lives in 0-20 px. Two
    hundred thresholds spread linearly over a span of several thousand
    then put the entire useful range inside one or two bins, which is why
    it reported DIR@FAR.1 = 0.006 next to R@1 = 0.727 -- a curve sampled
    too coarsely to see the operating point, read as a detection failure.

    The inlier-count methods have the mirror problem: 200 thresholds over
    an integer range of 0-30 mostly land between achievable values.

    Quantile placement fixes both, because it puts thresholds where the
    scores actually are. Duplicates are dropped, so a heavily tied score
    distribution simply yields fewer, well-placed thresholds.
    """
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return np.asarray([], dtype=float)
    q = np.quantile(s, np.linspace(0.0, 1.0, n))
    grid = np.unique(np.concatenate([[s.min()], q, [s.max()]]))
    # Nudge the top threshold above the max so "predict nothing" is reachable.
    span = float(grid[-1] - grid[0]) or 1.0
    return np.append(grid, grid[-1] + span * 1e-6)


def open_set_curve(scores: np.ndarray, relevant: np.ndarray, valid: np.ndarray,
                   n_thresholds: int = 200) -> dict:
    """Detection-and-identification rate vs false alarm rate.

    Known queries (identity present in the gallery) count as correct only
    if the top-ranked gallery entry is both above threshold AND correct.
    Unknown queries (identity absent - the crack was repaired, or is new)
    count as a false alarm whenever anything scores above threshold.
    """
    known_best, known_top_correct, unknown_best = [], [], []

    for i in range(scores.shape[0]):
        mask = valid[i]
        if not mask.any():
            continue
        s = scores[i][mask]
        r = (relevant[i] & mask)[mask]
        if not np.isfinite(s).any():
            continue
        top = int(np.argmax(s))
        if r.any():
            known_best.append(s[top])
            known_top_correct.append(bool(r[top]))
        else:
            unknown_best.append(s[top])

    known_best = np.asarray(known_best, dtype=float)
    known_top_correct = np.asarray(known_top_correct, dtype=bool)
    unknown_best = np.asarray(unknown_best, dtype=float)

    pool = np.concatenate([known_best, unknown_best]) if len(unknown_best) else known_best
    pool = pool[np.isfinite(pool)]
    if len(pool) == 0:
        return {"thresholds": [], "dir": [], "far": []}

    thresholds = _threshold_grid(pool, n_thresholds)
    dir_rate, far_rate = [], []
    for t in thresholds:
        if len(known_best):
            dir_rate.append(float(((known_best >= t) & known_top_correct).mean()))
        else:
            dir_rate.append(0.0)
        if len(unknown_best):
            far_rate.append(float((unknown_best >= t).mean()))
        else:
            far_rate.append(0.0)

    return {"thresholds": thresholds.tolist(), "dir": dir_rate, "far": far_rate,
            "n_known": int(len(known_best)), "n_unknown": int(len(unknown_best))}


def dir_at_far(curve: dict, target_far: float = 0.1) -> float:
    """Single headline number for the open-set case."""
    if not curve["thresholds"]:
        return 0.0
    far = np.asarray(curve["far"])
    dr = np.asarray(curve["dir"])
    ok = far <= target_far
    return float(dr[ok].max()) if ok.any() else 0.0


def pair_pr_curve(scores: np.ndarray, relevant: np.ndarray, valid: np.ndarray,
                  n_thresholds: int = 200) -> dict:
    """Precision-recall over all valid PAIRS, independent of ranking.

    This is the curve that tells an engineer where to set the operating
    threshold, and it is comparable across methods in a way that a single
    accuracy number is not.
    """
    s = scores[valid]
    y = relevant[valid]
    finite = np.isfinite(s)
    s, y = s[finite], y[finite]
    if len(s) == 0 or not y.any():
        return {"thresholds": [], "precision": [], "recall": [], "f1": []}

    thresholds = _threshold_grid(s, n_thresholds)
    precision, recall, f1 = [], [], []
    for t in thresholds:
        pred = s >= t
        tp = int((pred & y).sum())
        fp = int((pred & ~y).sum())
        fn = int((~pred & y).sum())
        p = tp / (tp + fp) if (tp + fp) else 1.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if (p + r) else 0.0)

    return {"thresholds": thresholds.tolist(), "precision": precision,
            "recall": recall, "f1": f1, "best_f1": float(max(f1))}


def pair_f1_at(scores: np.ndarray, relevant: np.ndarray, valid: np.ndarray,
               threshold: float) -> float:
    """Pair F1 at ONE given threshold -- the deployable counterpart of
    pair_pr_curve's best_f1.

    best_f1 maximises over thresholds chosen on the matrix being reported.
    Quoting it beside a validation-calibrated assignment F1 invites the
    reader to compare an oracle with an honest number. This evaluates the
    same curve at the threshold calibration actually chose.
    """
    s = scores[valid]
    y = relevant[valid]
    finite = np.isfinite(s)
    s, y = s[finite], y[finite]
    if len(s) == 0 or not y.any():
        return 0.0
    pred = s >= threshold
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    p = tp / (tp + fp) if (tp + fp) else 1.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return float(2 * p * r / (p + r)) if (p + r) else 0.0


# ===========================================================================
# One-to-one assignment (the operational setting)
# ===========================================================================

def assignment_accuracy(scores: np.ndarray, relevant: np.ndarray,
                        valid: np.ndarray, threshold: float,
                        queries: list[InstanceRef] | None = None,
                        gallery: list[InstanceRef] | None = None,
                        q_image=None, g_image=None) -> dict:
    """One-to-one matching solved PER IMAGE PAIR, the operational setting.

    Ranking metrics let two queries claim the same gallery entry.
    Deployment does not. Any method can be evaluated here by thresholding
    its score matrix and solving the assignment - so a method that
    natively produces assignments gets no special treatment.

    The assignment is solved independently for each (query image, gallery
    image) block. Solving it once GLOBALLY -- which this used to do, in
    contradiction of its own docstring -- forces one-to-one across the
    entire split, so a crack correctly visible in five gallery photos can
    be claimed exactly once and the other four become false negatives no
    method can avoid. That penalises hardest the methods that find the
    most correct matches, which is precisely backwards, and it is why
    every reported assF1 sat at or below 0.51.

    Pass either queries/gallery (InstanceRefs) or q_image/g_image (arrays
    of image ids) to say which cells belong to which photo pair. With
    neither, this falls back to the old global solve so old callers keep
    working - but the number they get is not the one the paper should
    quote.
    """
    if q_image is None and queries is not None:
        q_image = [r.image_id for r in queries]
    if g_image is None and gallery is not None:
        g_image = [r.image_id for r in gallery]
    from scipy.optimize import linear_sum_assignment

    work = np.where(valid & np.isfinite(scores), scores, -np.inf)

    def solve(rows_idx, cols_idx):
        """Hungarian over one block; yields accepted (i, j) pairs."""
        sub = work[np.ix_(rows_idx, cols_idx)]
        finite = sub[np.isfinite(sub)]
        if finite.size == 0:
            return []
        floor = finite.min() - 1.0
        r, c = linear_sum_assignment(-np.where(np.isfinite(sub), sub, floor))
        out = []
        for a, b in zip(r, c):
            i, j = rows_idx[a], cols_idx[b]
            if valid[i, j] and np.isfinite(scores[i, j]) and scores[i, j] >= threshold:
                out.append((i, j))
        return out

    accepted = []
    if q_image is not None and g_image is not None:
        blocks = defaultdict(lambda: (set(), set()))
        qi, gj = np.nonzero(valid)
        for i, j in zip(qi.tolist(), gj.tolist()):
            qs, gs = blocks[(q_image[i], g_image[j])]
            qs.add(i)
            gs.add(j)
        for (qs, gs) in blocks.values():
            accepted.extend(solve(sorted(qs), sorted(gs)))
    else:
        accepted = solve(list(range(scores.shape[0])), list(range(scores.shape[1])))

    tp = fp = 0
    matched_q = set()
    for i, j in accepted:
        matched_q.add(i)
        if relevant[i, j]:
            tp += 1
        else:
            fp += 1

    has_answer = np.array([(relevant[i] & valid[i]).any() for i in range(scores.shape[0])])
    fn = int(sum(1 for i in np.flatnonzero(has_answer) if i not in matched_q))
    correct_reject = int(sum(1 for i in np.flatnonzero(~has_answer) if i not in matched_q))
    n_unknown = int((~has_answer).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0,
        "correct_rejection_rate": correct_reject / n_unknown if n_unknown else float("nan"),
    }


# ===========================================================================
# Runner
# ===========================================================================

def evaluate(scorer: ReIDScorer, queries: list[InstanceRef], gallery: list[InstanceRef],
             data, threshold: float | None = None,
             scores: np.ndarray | None = None,
             timing: tuple[float, float] | None = None, **protocol) -> dict:
    """Run one method end to end and return every metric in one dict.

    scores/timing: pass a precomputed matrix to skip scoring entirely. The
    matrix is the only expensive artefact here (hours for the pairwise
    matchers); every metric below is milliseconds. Persisting it and
    re-evaluating is what makes a new metric, a bootstrap CI or a
    coverage-matched comparison free instead of an overnight re-run.
    """
    valid = build_validity_mask(queries, gallery, **protocol)
    relevant = build_relevance(queries, gallery)

    if scores is None:
        t0 = time.perf_counter()
        scorer.prepare(list(dict.fromkeys(queries + gallery)), data)
        t_prepare = time.perf_counter() - t0

        t0 = time.perf_counter()
        scores = scorer.score_matrix(queries, gallery, data)
        t_score = time.perf_counter() - t0
    else:
        t_prepare, t_score = timing or (0.0, 0.0)

    scores = np.asarray(scores, dtype=float)
    assert scores.shape == (len(queries), len(gallery)), "score matrix has the wrong shape"

    closed = closed_set_metrics(scores, relevant, valid)
    openset = open_set_curve(scores, relevant, valid)
    pr = pair_pr_curve(scores, relevant, valid)

    threshold_source = "validation"
    if threshold is None:                       # fall back to best-F1 operating point
        idx = int(np.argmax(pr["f1"])) if pr["f1"] else 0
        threshold = pr["thresholds"][idx] if pr["thresholds"] else 0.0
        threshold_source = "TEST (oracle)"

    assign = assignment_accuracy(scores, relevant, valid, threshold,
                                 queries=queries, gallery=gallery)

    n_pairs = int(valid.sum())
    return {
        "method": scorer.name,
        "input_scope": scorer.input_scope,
        "closed_set": asdict(closed),
        "open_set_dir_at_far10": dir_at_far(openset, 0.1),
        # pair_best_f1 is max(F1) swept over thresholds ON THE TEST MATRIX:
        # the best operating point achievable in hindsight, which no
        # deployment can pick. It sits in a table of honestly calibrated
        # numbers and must be labelled as the oracle it is -- hence the
        # separate pair_f1_at_threshold, which uses the SAME
        # validation-calibrated threshold as the assignment metric and is
        # the number a deployed system would actually get.
        "pair_best_f1": pr.get("best_f1", 0.0),
        "pair_f1_at_threshold": pair_f1_at(scores, relevant, valid, threshold),
        "assignment": assign,
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "scoreable_pair_rate": float(np.isfinite(scores[valid]).mean()) if n_pairs else 0.0,
        "prepare_seconds": t_prepare,
        "score_seconds": t_score,
        "ms_per_pair": 1000 * t_score / max(n_pairs, 1),
        # Embedding methods do all their work in prepare() and none in
        # score_matrix() (a cached matmul), so ms_per_pair alone reports
        # them as free next to a pairwise matcher. Total cost is the only
        # comparable number.
        "total_seconds": t_prepare + t_score,
        "ms_per_pair_total": 1000 * (t_prepare + t_score) / max(n_pairs, 1),
        "n_valid_pairs": n_pairs,
        "_curves": {"cmc": closed.cmc, "open_set": openset, "pr": pr},
        "_scores": scores,
    }


def calibrate_threshold(scorer: ReIDScorer, val_queries, val_gallery, data, **protocol) -> float:
    """Pick the operating threshold on VALIDATION data only.

    Tuning a threshold on the test split and reporting the result is the
    single most common way a paper like this gets rejected. Run this on
    the validation split, then pass the number into evaluate() on test.
    """
    valid = build_validity_mask(val_queries, val_gallery, **protocol)
    relevant = build_relevance(val_queries, val_gallery)
    scorer.prepare(list(dict.fromkeys(val_queries + val_gallery)), data)
    scores = np.asarray(scorer.score_matrix(val_queries, val_gallery, data), dtype=float)
    pr = pair_pr_curve(scores, relevant, valid)
    if not pr["f1"]:
        return 0.0
    return float(pr["thresholds"][int(np.argmax(pr["f1"]))])


def results_table(results: list[dict]) -> str:
    """Plain-text table, in roughly the shape the paper's main table needs."""
    hdr = (f"{'method':<24}{'scope':<12}{'nQ':>5}{'R@1':>7}{'R@5':>7}{'mAP':>7}"
           f"{'DIR@FAR.1':>11}{'pairF1*':>9}{'pairF1@v':>9}{'assF1':>8}"
           f"{'scored':>8}{'total_s':>9}")
    lines = [hdr, "-" * len(hdr)]
    for r in sorted(results, key=lambda x: -x["closed_set"]["mAP"]):
        lines.append(
            f"{r['method']:<24}{r['input_scope']:<12}"
            f"{r['closed_set']['n_queries']:>5d}"
            f"{r['closed_set']['rank1']:>7.3f}{r['closed_set']['rank5']:>7.3f}"
            f"{r['closed_set']['mAP']:>7.3f}{r['open_set_dir_at_far10']:>11.3f}"
            f"{r['pair_best_f1']:>9.3f}"
            f"{r.get('pair_f1_at_threshold', float('nan')):>9.3f}"
            f"{r['assignment']['f1']:>8.3f}"
            f"{r['scoreable_pair_rate']:>8.2f}"
            f"{r.get('total_seconds', r.get('score_seconds', 0.0)):>9.1f}"
        )
    lines.append("")
    lines.append("nQ = queries with a correct answer (the only ones R@1/mAP average over).")
    lines.append("pairF1*  = ORACLE: max F1 swept over thresholds on THIS (test) matrix. It is an")
    lines.append("           upper bound no deployment can pick, and is not comparable with assF1.")
    lines.append("pairF1@v = the same curve read at the threshold calibrated on VALIDATION -- the")
    lines.append("           number a deployed system gets. Quote this one; quote pairF1* as a ceiling.")
    lines.append("assF1  = one-to-one assignment F1 at the validation-calibrated threshold.")
    lines.append("scored = fraction of valid pairs the method returned a finite score for.")
    lines.append("         Below 1.00 is a PENALTY, not an easier subset: an unscored pair takes the")
    lines.append("         fail distance, ranks last, and its query is still counted. See the")
    lines.append("         coverage breakdown in results.json for what the missing fraction is made of.")
    lines.append("total_s = prepare + score. Embedding methods do all their work in prepare.")
    srcs = {r.get("threshold_source") for r in results}
    if "TEST (oracle)" in srcs:
        oracle = [r["method"] for r in results if r.get("threshold_source") == "TEST (oracle)"]
        lines.append("")
        lines.append(f"!! no validation threshold for {oracle}: their assF1 and pairF1@v are")
        lines.append("   test-tuned and optimistic. Fix the val split before quoting them.")
    return "\n".join(lines)


# ===========================================================================
# Example adapters
# ===========================================================================

class EmbeddingScorer(ReIDScorer):
    """Wraps any of the OSNet / DeiT / ViT / CLIP / YOLO embedders.

    `embed_fn` takes a list of BGR crops and returns (N, D) L2-normalized
    vectors. Every crop is embedded exactly once, which is also what makes
    the runtime column meaningful.
    """

    input_scope = "crop"

    def __init__(self, name: str, embed_fn, input_scope: str = "crop"):
        self.name = name
        self.embed_fn = embed_fn
        self.input_scope = input_scope
        self._cache = {}

    def prepare(self, refs, data):
        todo = [r for r in refs if r.instance_id not in self._cache]
        if not todo:
            return
        embs = self.embed_fn([data.crop(r) for r in todo])
        for r, e in zip(todo, embs):
            self._cache[r.instance_id] = e

    def score_matrix(self, queries, gallery, data):
        q = np.stack([self._cache[r.instance_id] for r in queries])
        g = np.stack([self._cache[r.instance_id] for r in gallery])
        return q @ g.T


class RegistrationScorer(DistanceScorer):
    """Wraps the registration pipeline.

    Registration is per IMAGE PAIR, so homographies are cached and reused
    across all instance pairs from the same two photos. Pairs whose
    registration fails score -inf and are counted in scoreable_pair_rate
    rather than being dropped.
    """

    name = "registration+chamfer"
    input_scope = "full-image"

    def __init__(self, register_fn, chamfer_fn, fail_distance: float = np.inf):
        self.register_fn = register_fn
        self.chamfer_fn = chamfer_fn
        self.fail_distance = fail_distance
        self._H = {}

    def _homography(self, img_a_id, img_b_id, data):
        key = (img_a_id, img_b_id)
        if key not in self._H:
            self._H[key] = self.register_fn(img_a_id, img_b_id, data)
        return self._H[key]

    def distance_matrix(self, queries, gallery, data):
        d = np.full((len(queries), len(gallery)), self.fail_distance, dtype=float)
        by_pair = defaultdict(list)
        for i, q in enumerate(queries):
            for j, g in enumerate(gallery):
                by_pair[(q.image_id, g.image_id)].append((i, j))

        for (a_id, b_id), cells in by_pair.items():
            H = self._homography(a_id, b_id, data)
            if H is None:
                continue
            for i, j in cells:
                d[i, j] = self.chamfer_fn(queries[i], gallery[j], H, data)
        return d
