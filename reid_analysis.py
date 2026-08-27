"""
Analysis over saved score matrices.

benchmark.py writes one .npz per (method, split) holding the raw score
matrix plus the identity/wall/session arrays needed to rebuild the
protocol masks. Everything here works off those files, so:

  * runs done on different days, in different Colab sessions, one method
    at a time, merge into a single table;
  * a new metric costs milliseconds instead of an overnight re-run;
  * every method is re-scored by identical code from identical inputs.

What it adds over the plain results table:

  BOOTSTRAP CONFIDENCE INTERVALS
      The test split has ~161 queries with an answer, spread unevenly
      over 11 walls. A bare "R@1 = 0.727" invites a reviewer to ask what
      the error bar is. Queries are resampled with replacement; the
      reported interval is the 2.5/97.5 percentile.

  COVERAGE-MATCHED COMPARISON
      registration+chamfer returns a finite score for only ~34% of valid
      pairs (registration fails on the rest).

      This does NOT flatter its ranking metrics, which is the intuitive
      but wrong reading. A pair it cannot score becomes -inf and ranks
      LAST; the query is still counted. Measured by holding a real score
      matrix fixed and censoring it: at 34% coverage R@1 fell 0.422 ->
      0.27, and censoring whole photo-pair blocks (registration's actual
      failure structure) was no kinder than censoring cells uniformly.
      Partial coverage is a penalty, not a subsidy.

      What the matched view is actually for: answering "when registration
      CAN answer, how good is it compared to the others on exactly those
      pairs?" That separates the quality of its answers from how often it
      produces one -- two different claims a paper should not merge.

  SCOPE GROUPING
      Methods are grouped by input_scope so a crop-only method is never
      silently compared against a full-image one.

Usage:
    python reid_analysis.py benchmark_out --split test
    python reid_analysis.py benchmark_out --split test --bootstrap 2000
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np

from reid_eval import (
    closed_set_metrics, open_set_curve, dir_at_far, pair_pr_curve,
    assignment_accuracy,
)


# ===========================================================================
# Loading
# ===========================================================================

@dataclass
class ScoreRun:
    """One saved (method, split) score matrix and its protocol arrays."""
    method: str
    input_scope: str
    split: str
    scores: np.ndarray
    prepare_seconds: float
    score_seconds: float
    query: dict
    gallery: dict
    path: str
    #: the three-way split of `scored`, saved alongside the matrix by
    #: benchmark.py. Empty for methods that score every cell.
    coverage: dict | None = None

    @property
    def key(self) -> str:
        return f"{self.method}|{self.split}"


def _cols(z, prefix: str) -> dict:
    return {f: z[f"{prefix}_{f}"] for f in
            ("instance_id", "image_id", "wall_id", "session", "identity")}


def load_runs(out_dir: str, split: str | None = None) -> list[ScoreRun]:
    """Load every saved matrix under out_dir/scores/, newest wins on ties."""
    runs: dict[str, ScoreRun] = {}
    pattern = os.path.join(out_dir, "scores", "*.npz")
    for path in sorted(glob.glob(pattern), key=os.path.getmtime):
        z = np.load(path, allow_pickle=False)
        r = ScoreRun(
            method=str(z["method"]),
            input_scope=str(z["input_scope"]),
            split=str(z["split"]),
            scores=z["scores"],
            prepare_seconds=float(z["prepare_seconds"]),
            score_seconds=float(z["score_seconds"]),
            query=_cols(z, "query"),
            gallery=_cols(z, "gallery"),
            path=path,
            coverage=json.loads(str(z["coverage_json"])) if "coverage_json" in z else None,
        )
        if split and r.split != split:
            continue
        runs[r.key] = r          # later mtime overwrites an earlier re-run
    return list(runs.values())


# ===========================================================================
# Protocol masks, rebuilt from the saved arrays
# ===========================================================================

def _frames(image_ids: np.ndarray) -> np.ndarray:
    """Frame index per photo, or -1 where the name carries none."""
    out = np.full(len(image_ids), -1, dtype=np.int64)
    for i, iid in enumerate(image_ids):
        tail = str(iid).rsplit("_", 1)[-1]
        if tail.isdigit():
            out[i] = int(tail)
    return out


def masks_for(run: ScoreRun, same_wall_only: bool = True,
              exclude_same_session: bool = True,
              min_frame_gap: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(valid, relevant), matching reid_eval.build_validity_mask exactly but
    vectorised, since the analysis rebuilds these for every bootstrap.

    min_frame_gap drops gallery photos within N frames of the query's own.
    Every answerable query's nearest correct answer is the ADJACENT frame,
    so at 0 this measures near-duplicate retrieval across ~3 seconds. The
    sweep costs nothing here: image_id is already in the saved matrix, so
    no method has to be re-scored to produce the whole curve.
    """
    q, g = run.query, run.gallery
    same_image = q["image_id"][:, None] == g["image_id"][None, :]
    valid = ~same_image
    if exclude_same_session:
        valid &= q["session"][:, None] != g["session"][None, :]
    if same_wall_only:
        valid &= q["wall_id"][:, None] == g["wall_id"][None, :]
    if min_frame_gap > 0:
        qf, gf = _frames(q["image_id"]), _frames(g["image_id"])
        known = (qf[:, None] >= 0) & (gf[None, :] >= 0)
        near = np.abs(qf[:, None] - gf[None, :]) <= min_frame_gap
        same_wall = q["wall_id"][:, None] == g["wall_id"][None, :]
        valid &= ~(known & near & same_wall)

    q_id, g_id = q["identity"], g["identity"]
    relevant = (q_id[:, None] == g_id[None, :]) & (q_id[:, None] != "")
    return valid, relevant


def _metrics(scores, relevant, valid, threshold=None,
             q_image=None, g_image=None) -> dict:
    closed = closed_set_metrics(scores, relevant, valid)
    openset = open_set_curve(scores, relevant, valid)
    pr = pair_pr_curve(scores, relevant, valid)
    if threshold is None:
        idx = int(np.argmax(pr["f1"])) if pr["f1"] else 0
        threshold = pr["thresholds"][idx] if pr["thresholds"] else 0.0
    # per-image-pair assignment, matching reid_eval/benchmark
    assign = assignment_accuracy(scores, relevant, valid, threshold,
                                 q_image=q_image, g_image=g_image)
    n_pairs = int(valid.sum())
    return {
        "n_queries": closed.n_queries,
        "rank1": closed.rank1,
        "rank5": closed.rank5,
        "mAP": closed.mAP,
        "dir_at_far10": dir_at_far(openset, 0.1),
        "pair_best_f1": pr.get("best_f1", 0.0),
        "assignment_f1": assign["f1"],
        "scoreable_pair_rate": float(np.isfinite(scores[valid]).mean()) if n_pairs else 0.0,
        "n_valid_pairs": n_pairs,
    }


# ===========================================================================
# Bootstrap
# ===========================================================================

def bootstrap_ci(scores, relevant, valid, n_boot: int = 1000,
                 seed: int = 0, alpha: float = 0.05,
                 cluster: np.ndarray | None = None) -> dict:
    """Percentile CIs for the ranking metrics.

    cluster=None  resample QUERIES independently. Narrow, and too
        optimistic here: many queries come from the same wall, share its
        lighting, material and camera pass, and succeed or fail together.

    cluster=<wall id per query>  resample WALLS with replacement, taking
        all of a wall's queries each time. This is the honest interval for
        this dataset, because the wall is the real unit of replication --
        and with 11 test walls (several contributing no matchable
        identities at all) it is much wider than the query-level one.
        Report this if the paper claims one method beats another.
    """
    rng = np.random.default_rng(seed)
    n_q = scores.shape[0]
    if not (relevant & valid).any():
        return {}

    if cluster is not None:
        groups = {}
        for i, c in enumerate(cluster):
            groups.setdefault(c, []).append(i)
        units = [np.array(v) for v in groups.values()]
    else:
        units = None

    draws = {k: [] for k in ("rank1", "mAP", "dir_at_far10")}
    for _ in range(n_boot):
        if units is None:
            pick = rng.integers(0, n_q, size=n_q)
        else:
            chosen = rng.integers(0, len(units), size=len(units))
            pick = np.concatenate([units[c] for c in chosen])
        s, r, v = scores[pick], relevant[pick], valid[pick]
        closed = closed_set_metrics(s, r, v)
        if closed.n_queries == 0:
            continue
        draws["rank1"].append(closed.rank1)
        draws["mAP"].append(closed.mAP)
        draws["dir_at_far10"].append(dir_at_far(open_set_curve(s, r, v), 0.1))

    out = {}
    lo_p, hi_p = 100 * alpha / 2, 100 * (1 - alpha / 2)
    for k, vals in draws.items():
        if vals:
            out[k] = (float(np.percentile(vals, lo_p)),
                      float(np.percentile(vals, hi_p)))
    return out


# ===========================================================================
# Score normalisation
# ===========================================================================

def normalize_scores(scores: np.ndarray, valid: np.ndarray,
                     mode: str = "none") -> np.ndarray:
    """Re-scale a score matrix so scores are comparable ACROSS queries.

    Why this matters here: the ranking metrics (R@1, mAP) only ever
    compare scores WITHIN one query's row, so any per-row monotone
    transform leaves them untouched. But DIR@FAR, pair-F1 and the
    Hungarian assignment all apply ONE GLOBAL THRESHOLD across every row.
    Those metrics are therefore measuring calibration, not just accuracy.

    registration+chamfer is the extreme case: each photo pair gets its own
    independently solved homography, so its chamfer distances live on a
    per-pair scale. Excellent within a row (R@1 = 0.73) and meaningless
    across rows (DIR@FAR.1 = 0.006). That is a calibration failure being
    reported as a detection failure.

    modes:
      none : raw scores.
      row  : z-score each query against its own valid finite candidates.
             The top-1 score becomes "how many sigma ahead of this query's
             own alternatives", which is comparable across queries even
             when the raw scale is not.
      rank : replace each score by its within-row percentile. Scale-free
             and outlier-proof, but discards margin information.
    """
    if mode == "none":
        return scores
    out = np.full_like(scores, -np.inf, dtype=float)
    for i in range(scores.shape[0]):
        m = valid[i] & np.isfinite(scores[i])
        if m.sum() < 2:
            out[i][m] = 0.0
            continue
        row = scores[i][m]
        if mode == "row":
            sd = row.std()
            out[i][m] = (row - row.mean()) / sd if sd > 0 else 0.0
        elif mode == "rank":
            order = row.argsort().argsort().astype(float)
            out[i][m] = order / max(len(row) - 1, 1)
        else:
            raise ValueError(f"unknown normalisation mode {mode!r}")
    return out


# ===========================================================================
# Coverage-matched comparison
# ===========================================================================

def common_finite_mask(runs: list[ScoreRun]) -> np.ndarray:
    """Cells every run produced a finite score for.

    Requires the runs share a query/gallery ordering, which they do when
    they came from the same dataset and split; verified by caller.
    """
    m = np.ones_like(runs[0].scores, dtype=bool)
    for r in runs:
        m &= np.isfinite(r.scores)
    return m


def check_aligned(runs: list[ScoreRun]) -> list[str]:
    """Return a list of complaints; empty means the runs are comparable."""
    problems = []
    ref = runs[0]
    for r in runs[1:]:
        for side, a, b in (("query", ref.query, r.query),
                           ("gallery", ref.gallery, r.gallery)):
            if len(a["instance_id"]) != len(b["instance_id"]):
                problems.append(f"{r.method}: {side} count "
                                f"{len(b['instance_id'])} != {len(a['instance_id'])}")
            elif not np.array_equal(a["instance_id"], b["instance_id"]):
                problems.append(f"{r.method}: {side} ordering differs from {ref.method}")
    return problems


def _calibrate(val_run: "ScoreRun | None", norm_mode: str = "none") -> float | None:
    """Best-F1 threshold on the validation matrix, under the same
    normalisation the test scores will get. None if no val matrix exists,
    which makes _metrics fall back to a test-tuned point -- reported
    explicitly in the header so it is never mistaken for a clean number."""
    if val_run is None:
        return None
    valid, relevant = masks_for(val_run)
    sc = normalize_scores(val_run.scores, valid, mode=norm_mode)
    pr = pair_pr_curve(sc, relevant, valid)
    if not pr["f1"]:
        return None
    return float(pr["thresholds"][int(np.argmax(pr["f1"]))])


# ===========================================================================
# Fusion  (finding 16: free, from matrices already on disk)
# ===========================================================================

def fuse(primary: ScoreRun, backup: ScoreRun, valid: np.ndarray) -> np.ndarray:
    """Cascade two saved score matrices: use `primary` wherever it answered,
    fall back to `backup` elsewhere.

    registration+chamfer answers a minority of pairs very well; the
    appearance methods answer all of them moderately. Neither alone is the
    best available system, and the combination needs no re-scoring at all --
    check_aligned() guarantees the two matrices share a query/gallery
    ordering, so this is a masked combination of two arrays already on disk.

    The two score scales are unrelated, so the backup is mapped onto the
    primary's within-row scale before substitution: primary answers keep
    their ordering and always outrank a fallback, which is the intended
    semantics of a cascade.
    """
    P, B = primary.scores, backup.scores
    out = np.full_like(P, -np.inf, dtype=float)
    for i in range(P.shape[0]):
        m = valid[i]
        if not m.any():
            continue
        p_ok = m & np.isfinite(P[i])
        b_ok = m & np.isfinite(B[i]) & ~p_ok
        if p_ok.any():
            out[i][p_ok] = P[i][p_ok]
        if b_ok.any():
            bb = B[i][b_ok]
            rng = bb.max() - bb.min()
            unit = (bb - bb.min()) / rng if rng > 0 else np.zeros_like(bb)
            if p_ok.any():
                lo = P[i][p_ok].min()
                span = (P[i][p_ok].max() - lo) or 1.0
                # strictly below every primary answer
                out[i][b_ok] = lo - span * (1.0 + (1.0 - unit))
            else:
                out[i][b_ok] = unit
    return out


def frame_gap_sweep(out_dir: str, split: str = "test",
                    gaps=(0, 1, 2, 3)) -> dict:
    """R@1 / mAP for every method as adjacent frames are excluded.

    WHAT THIS IS: a near-duplicate control over CAPTURE SEPARATION. Each
    wall was shot in one continuous pass and for 100% of answerable
    queries the nearest correct answer is the immediately adjacent frame,
    so gap 0 measures near-duplicate retrieval across roughly three
    seconds. Widening the excluded neighbourhood shows how much of the
    headline number is near-duplicate matching. That is worth reporting.

    WHAT THIS IS NOT: the viewpoint axis. Two earlier turns of review
    called this the experiment that decides the paper; that was wrong.
    Frame index measures elapsed capture order, and over the pairs where
    geometry can be measured it tracks in-plane rotation while leaving
    scale and perspective tilt essentially uncorrelated -- so a frame-gap
    sweep is a CAMERA-ROLL sweep, and roll is the one viewpoint component
    a homography absorbs exactly and any rotation-invariant descriptor
    handles. Reporting a flat curve here and calling it viewpoint
    invariance would be demonstrating almost nothing while appearing to
    demonstrate the paper's central claim.

    `python viewpoint.py <root> --out <dir> --report-only` prints the three
    rank correlations measured on THIS dataset. Quote those, not a
    remembered figure.

    For the viewpoint axis use viewpoint_sweep() below, which stratifies
    on a covariate measured by a front end the scoring pipeline does not
    use.
    """
    runs = load_runs(out_dir, split=split)
    table = {}
    for r in runs:
        row = []
        for k in gaps:
            valid, relevant = masks_for(r, min_frame_gap=k)
            m = _metrics(r.scores, relevant, valid)
            row.append({"gap": k, "n_queries": m["n_queries"],
                        "rank1": m["rank1"], "mAP": m["mAP"]})
        table[r.method] = row
    return table


def format_frame_gap(table: dict, gaps=(0, 1, 2, 3)) -> str:
    L = ["", "=" * 92,
         "CAPTURE-SEPARATION SWEEP (near-duplicate control, NOT the viewpoint axis)",
         "gallery photos within +/-N frames of the query are excluded. gap 0 is the published",
         "setting; every query's nearest correct answer is at gap 1, so gap 0 measures",
         "near-duplicate retrieval (~3 s apart), not re-identification.",
         "Frame gap tracks camera ROLL and not scale or tilt, so it is not a viewpoint",
         "stratifier -- see --viewpoint-sweep, and `viewpoint.py --report-only` for the",
         "three measured rank correlations.",
         "=" * 92]
    hdr = f"{'method':<24}" + "".join(f"{'gap' + str(k):>18}" for k in gaps)
    L += [hdr, "-" * len(hdr)]
    for meth, row in sorted(table.items(), key=lambda kv: -kv[1][0]["mAP"]):
        cells = "".join(f"{r['rank1']:>9.3f}/{r['n_queries']:<8d}" for r in row)
        L.append(f"{meth:<24}{cells}")
    L.append("")
    L.append("cells are R@1 / number of queries that still have an answer at that gap.")
    return "\n".join(L)


# ===========================================================================
# Viewpoint stratification  (S2: the axis the paper actually claims)
# ===========================================================================

def _pair_key(a: str, b: str) -> str:
    return f"{a}|{b}" if a <= b else f"{b}|{a}"


def viewpoint_sweep(out_dir: str, split: str = "test",
                    key: str = "scale_change",
                    bins: tuple = (1.0, 1.25, 1.6, 2.5, 1e9)) -> dict:
    """R@1 / mAP per method, stratified by MEASURED viewpoint change.

    The covariate comes from viewpoint.py, which estimates the pairwise
    geometry with front ends the scoring pipeline does not use, so a cell
    exists for pairs registration cannot score. That is what makes this a
    stratification rather than a restatement of where the method works.

    Cells are dropped, not filled, where no covariate could be measured --
    and the report says how many pairs that was. A stratifier that quietly
    keeps only its own successes is the circularity this replaces.
    """
    from viewpoint import load
    pairs = load(out_dir)
    cov = {}
    for k, rec in pairs.items():
        vp = rec.get("viewpoint", {})
        if vp.get("ok") and np.isfinite(vp.get(key, np.nan)):
            cov[k] = float(vp[key])

    runs = load_runs(out_dir, split=split)
    edges = list(bins)
    table, coverage = {}, {}
    for r in runs:
        valid, relevant = masks_for(r)
        qi, gi = r.query["image_id"], r.gallery["image_id"]
        # per-cell covariate, NaN where the pair has none
        cvals = np.full(valid.shape, np.nan)
        for i in range(valid.shape[0]):
            for j in np.nonzero(valid[i])[0]:
                v = cov.get(_pair_key(str(qi[i]), str(gi[j])))
                if v is not None:
                    cvals[i, j] = v
        have = np.isfinite(cvals) & valid
        coverage = {"pairs_with_covariate": int(have.sum()),
                    "pairs_valid": int(valid.sum()),
                    "fraction": float(have.sum() / max(valid.sum(), 1))}
        row = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = valid & np.isfinite(cvals) & (cvals >= lo) & (cvals < hi)
            if not sel.any():
                row.append({"lo": lo, "hi": hi, "n_queries": 0,
                            "rank1": float("nan"), "mAP": float("nan"),
                            "n_pairs": 0})
                continue
            m = _metrics(r.scores, relevant, sel)
            row.append({"lo": lo, "hi": hi, "n_pairs": int(sel.sum()), **{
                k2: m[k2] for k2 in ("n_queries", "rank1", "mAP")}})
        table[r.method] = row
    return {"key": key, "table": table, "coverage": coverage}


def format_viewpoint(sweep: dict) -> str:
    key = sweep["key"]
    cov = sweep.get("coverage", {})
    L = ["", "=" * 100,
         f"VIEWPOINT SWEEP on {key} -- a covariate measured by a front end the scoring",
         "pipeline does not use (viewpoint.py), so cells exist where registration fails.",
         f"covariate available on {cov.get('pairs_with_covariate', 0)}/"
         f"{cov.get('pairs_valid', 0)} valid pairs ({cov.get('fraction', 0):.0%})",
         "=" * 100]
    any_row = next(iter(sweep["table"].values()), [])

    def band(b):
        hi = "inf" if b["hi"] >= 1e8 else f"{b['hi']:g}"
        return f"[{b['lo']:g},{hi})"

    hdr = f"{'method':<24}" + "".join(f"{band(b):>18}" for b in any_row)
    L += [hdr, "-" * max(len(hdr), 1)]
    for meth, row in sorted(sweep["table"].items(),
                            key=lambda kv: -max((c["mAP"] for c in kv[1]
                                                 if np.isfinite(c["mAP"])), default=0)):
        cells = "".join(
            (f"{c['rank1']:>9.3f}/{c['n_queries']:<8d}" if c["n_queries"]
             else f"{'-':>18}") for c in row)
        L.append(f"{meth:<24}{cells}")
    L += ["", "cells are R@1 / queries with an answer inside that band of viewpoint change.",
          "A FLAT row is the invariance claim. A row that decays names the envelope's edge.",
          "Empty cells mean the dataset does not cover that band -- which is itself a result,",
          "and the reason for re-shooting a subset at marked distances and angles."]
    return "\n".join(L)


# ===========================================================================
# Per-wall coverage  (S4: coverage is a property of the surface)
# ===========================================================================

def per_wall_coverage(out_dir: str, split: str = "test") -> dict:
    """Scored-pair rate per wall per method.

    Registration failures are not distributed at random -- they are
    concentrated on the walls where masking the crack out of keypoint
    detection leaves no features, and on the walls shot out of focus. A
    query-level bootstrap therefore understates the uncertainty badly, and
    a single `scored` number hides a scope condition. Both are fixed by
    printing the breakdown.
    """
    runs = load_runs(out_dir, split=split)
    out = {}
    for r in runs:
        valid, _ = masks_for(r)
        finite = np.isfinite(r.scores)
        rows = {}
        for w in sorted(set(map(str, r.query["wall_id"]))):
            sel = valid & (r.query["wall_id"][:, None] == w)
            n = int(sel.sum())
            rows[w] = {"pairs": n,
                       "scored": int((sel & finite).sum()),
                       "rate": float((sel & finite).sum() / n) if n else float("nan")}
        out[r.method] = rows
    return out


def format_per_wall_coverage(table: dict) -> str:
    walls = sorted({w for rows in table.values() for w in rows})
    L = ["", "=" * 100,
         "COVERAGE BY WALL -- where each method can answer at all",
         "=" * 100,
         f"{'method':<24}" + "".join(f"{w[-2:]:>6}" for w in walls) + f"{'all':>8}"]
    L.append("-" * len(L[-1]))
    for meth, rows in sorted(table.items(),
                             key=lambda kv: -sum(v["scored"] for v in kv[1].values())):
        cells = "".join((f"{rows[w]['rate']:>6.2f}" if w in rows and rows[w]["pairs"]
                         else f"{'-':>6}") for w in walls)
        tot_p = sum(v["pairs"] for v in rows.values())
        tot_s = sum(v["scored"] for v in rows.values())
        L.append(f"{meth:<24}{cells}{tot_s / max(tot_p, 1):>8.2f}")
    L += ["", "cells are the fraction of that wall's valid pairs the method scored.",
          "A method whose zeros cluster on particular walls has a SCOPE CONDITION, not noise:",
          "resample walls (CI (wall) above), not queries, and state the condition in the paper."]
    return "\n".join(L)


# ===========================================================================
# Surface stratification  (finding 20)
# ===========================================================================

def load_surface_map(path: str) -> dict:
    """wall_id -> surface class, from a CSV with a `surface` column.

    The dataset holds two physically distinct populations and nearly every
    anomaly in this benchmark traces back to which one a wall belongs to.
    Textured render: registration near-perfect, label propagation works,
    identities exist. Smooth painted plaster: registration impossible
    (0 of 330 pairs on walls 10/13/14/16/17), propagation mints a fresh
    identity per photo, the wall contributes only distractors.

    That is the mechanism behind the correlation between registration rate
    and matchable-identity count -- not a coincidence between two methods,
    but surface texture driving both. Reporting the table once per regime
    turns the most awkward finding in the review into a stated domain of
    validity.

    Add the column with e.g.
        wall_id,surface
        wall09,textured
        wall13,smooth
    """
    import csv as _csv
    out = {}
    with open(path) as f:
        for row in _csv.DictReader(f):
            if row.get("surface"):
                out[row["wall_id"]] = row["surface"].strip()
    return out


def stratified(out_dir: str, surface_map: dict, split: str = "test") -> dict:
    """Metrics per method per surface class."""
    runs = load_runs(out_dir, split=split)
    table = {}
    for r in runs:
        valid, relevant = masks_for(r)
        wall = r.query["wall_id"]
        rows = {}
        for surf in sorted(set(surface_map.values())):
            keep = np.array([surface_map.get(str(w)) == surf for w in wall])
            if not keep.any():
                continue
            v = valid.copy()
            v[~keep] = False
            rows[surf] = _metrics(r.scores, relevant, v)
        table[r.method] = rows
    return table


def format_stratified(table: dict) -> str:
    classes = sorted({c for rows in table.values() for c in rows})
    L = ["", "=" * 92,
         "BY SURFACE REGIME -- the dataset holds two physically different populations",
         "=" * 92]
    hdr = f"{'method':<24}" + "".join(f"{c:>22}" for c in classes)
    L += [hdr, "-" * len(hdr)]
    for meth, rows in sorted(table.items(),
                             key=lambda kv: -max((m["mAP"] for m in kv[1].values()), default=0)):
        cells = ""
        for c in classes:
            m = rows.get(c)
            cells += (f"{m['rank1']:>10.3f}/{m['n_queries']:<11d}" if m else f"{'-':>22}")
        L.append(f"{meth:<24}{cells}")
    L.append("")
    L.append("cells are R@1 / queries with an answer, within that surface class.")
    return "\n".join(L)


# ===========================================================================
# Reporting
# ===========================================================================

def analyse(out_dir: str, split: str = "test", n_boot: int = 1000,
            seed: int = 0, norm_mode: str = "row") -> dict:
    runs = load_runs(out_dir, split=split)
    if not runs:
        raise SystemExit(f"no saved matrices in {out_dir}/scores/ for split={split!r}. "
                         f"Run benchmark.py first (it now saves them).")

    # Operating thresholds MUST come from a different split than the one
    # being reported. Picking the best-F1 point on test and then reporting
    # the F1 it produces is circular and is the fastest way to get a paper
    # like this rejected. calib_runs supplies that threshold; if the val
    # matrices are missing we say so rather than silently tuning on test.
    calib = {r.method: r for r in load_runs(out_dir, split="val")} if split != "val" else {}

    problems = check_aligned(runs)
    report = {"split": split, "n_methods": len(runs), "alignment_problems": problems,
              "calibrated_on": "val" if calib else "TEST (no val matrices found)",
              "uncalibrated_methods": sorted({r.method for r in runs} - set(calib))
                                      if calib else []}

    rows = []
    for r in runs:
        valid, relevant = masks_for(r)
        thr = _calibrate(calib.get(r.method), norm_mode="none")
        m = _metrics(r.scores, relevant, valid, threshold=thr,
                     q_image=r.query["image_id"], g_image=r.gallery["image_id"])
        m["method"] = r.method
        m["input_scope"] = r.input_scope
        m["total_seconds"] = r.prepare_seconds + r.score_seconds
        m["coverage"] = r.coverage or {}
        # S6: registration's cost is per IMAGE PAIR (one homography serves
        # every crack in that pair) while an embedder's is per instance,
        # so seconds-per-query alone reports the amortisation backwards.
        if (r.coverage or {}).get("image_pairs"):
            m["seconds_per_image_pair"] = m["total_seconds"] / r.coverage["image_pairs"]
        m["seconds_per_query"] = m["total_seconds"] / max(m["n_queries"], 1)
        if n_boot:
            m["ci"] = bootstrap_ci(r.scores, relevant, valid,
                                   n_boot=n_boot, seed=seed)
            m["ci_wall"] = bootstrap_ci(r.scores, relevant, valid, n_boot=n_boot,
                                        seed=seed, cluster=r.query["wall_id"])
        rows.append(m)

    report["full"] = rows

    # ---- calibration: same ranking, comparable-across-queries scores ----
    if norm_mode and norm_mode != "none":
        norm_rows = []
        for r in runs:
            valid, relevant = masks_for(r)
            ns = normalize_scores(r.scores, valid, mode=norm_mode)
            thr = _calibrate(calib.get(r.method), norm_mode=norm_mode)
            m = _metrics(ns, relevant, valid, threshold=thr,
                         q_image=r.query["image_id"], g_image=r.gallery["image_id"])
            m["method"] = r.method
            m["input_scope"] = r.input_scope
            norm_rows.append(m)
        report["normalized"] = norm_rows
        report["norm_mode"] = norm_mode

    # ---- coverage-matched ----
    if len(runs) > 1 and not problems:
        common = common_finite_mask(runs)
        matched = []
        for r in runs:
            valid, relevant = masks_for(r)
            v = valid & common
            m = _metrics(r.scores, relevant, v,
                         q_image=r.query["image_id"], g_image=r.gallery["image_id"])
            m["method"] = r.method
            m["input_scope"] = r.input_scope
            matched.append(m)
        report["matched"] = matched
        base_valid, _ = masks_for(runs[0])
        report["matched_coverage"] = {
            "pairs_kept": int((common & base_valid).sum()),
            "pairs_total": int(base_valid.sum()),
            "fraction": float((common & base_valid).sum() / max(base_valid.sum(), 1)),
        }
    return report


def _fmt_ci(ci: dict, key: str) -> str:
    if not ci or key not in ci:
        return ""
    lo, hi = ci[key]
    return f"[{lo:.3f},{hi:.3f}]"


def format_report(rep: dict) -> str:
    L = []
    L.append(f"SPLIT: {rep['split']}   methods: {rep['n_methods']}")
    L.append(f"assF1 threshold calibrated on: {rep['calibrated_on']}")
    if rep.get("uncalibrated_methods"):
        L.append(f"  !! no val matrix for {rep['uncalibrated_methods']} -- their assF1 is "
                 f"test-tuned and optimistic")
    if rep["alignment_problems"]:
        L.append("\n!! runs are NOT directly comparable:")
        for p in rep["alignment_problems"]:
            L.append(f"   - {p}")
        L.append("   (coverage-matched comparison skipped)")

    L.append("\n" + "=" * 100)
    L.append("FULL COVERAGE  -- each method on every pair it can score (the deployable numbers)")
    L.append("=" * 100)
    hdr = (f"{'method':<22}{'scope':<11}{'nQ':>5}{'R@1':>7}{'R@1 CI (query)':>18}"
           f"{'R@1 CI (wall)':>18}{'mAP':>7}{'DIR@.1':>8}{'assF1':>7}{'scored':>8}")
    L.append(hdr)
    L.append("-" * len(hdr))
    for r in sorted(rep["full"], key=lambda x: -x["mAP"]):
        ci, ciw = r.get("ci", {}), r.get("ci_wall", {})
        L.append(f"{r['method']:<22}{r['input_scope']:<11}{r['n_queries']:>5d}"
                 f"{r['rank1']:>7.3f}{_fmt_ci(ci,'rank1'):>18}"
                 f"{_fmt_ci(ciw,'rank1'):>18}"
                 f"{r['mAP']:>7.3f}"
                 f"{r['dir_at_far10']:>8.3f}{r['assignment_f1']:>7.3f}"
                 f"{r['scoreable_pair_rate']:>8.2f}")
    L.append("\nCI (query) resamples queries; CI (wall) resamples WALLS, which is the")
    L.append("honest unit of replication here. Quote the wall-level interval in the paper --")
    L.append("if two methods' wall-level intervals overlap, the dataset cannot separate them.")

    breakdown = [r for r in rep["full"] if r.get("coverage", {}).get("cells_total")]
    if breakdown:
        L.append("\n" + "=" * 100)
        L.append("WHAT `scored` IS MADE OF -- three events a single number was hiding")
        L.append("=" * 100)
        hdr_c = (f"{'method':<24}{'img pairs':>10}{'registered':>12}"
                 f"{'scored':>9}{'out-of-frame':>14}{'unregistered':>14}")
        L.append(hdr_c)
        L.append("-" * len(hdr_c))
        for r in breakdown:
            c = r["coverage"]
            reg = c["image_pairs"] - c["registration_failures"]
            L.append(f"{r['method']:<24}{c['image_pairs']:>10d}"
                     f"{reg:>7d} ({1 - c['registration_failure_rate']:.0%})"
                     f"{c.get('frac_scored', float('nan')):>9.2f}"
                     f"{c.get('frac_out_of_frame', float('nan')):>14.2f}"
                     f"{c.get('frac_unregistered', float('nan')):>14.2f}")
        L.append("")
        L.append("out-of-frame = the images registered and the query crack projects OUTSIDE the")
        L.append("  gallery photo: a confident, geometrically grounded 'not in this photo' that no")
        L.append("  crop-scope method can produce at all. It is an ANSWER, and counting it as a")
        L.append("  non-answer is what made `scored` read as a weakness.")
        L.append("unregistered = a genuine non-answer. The pair takes the fail distance and ranks")
        L.append("  last; the query is still counted, so this is a penalty, not an easier subset.")

    cost = [r for r in rep["full"] if "seconds_per_image_pair" in r]
    if cost:
        L.append("\nCOST, AMORTISED  (S6: one homography serves every crack in an image pair)")
        for r in sorted(rep["full"], key=lambda x: x.get("seconds_per_query", 0)):
            per_pair = (f"{r['seconds_per_image_pair']:.2f} s/image-pair"
                        if "seconds_per_image_pair" in r else "-- (cost is per instance)")
            L.append(f"   {r['method']:<24}{r.get('seconds_per_query', 0):>8.2f} s/query   {per_pair}")
        L.append("   On a wall with many defects the per-query figure falls while an embedder's")
        L.append("   does not. Report both, and state the crossover.")

    if "matched" in rep:
        cov = rep["matched_coverage"]
        L.append("\n" + "=" * 100)
        L.append(f"COVERAGE-MATCHED -- all methods restricted to the "
                 f"{cov['pairs_kept']}/{cov['pairs_total']} pairs "
                 f"({cov['fraction']:.0%}) EVERY method could score")
        L.append("Isolates answer QUALITY from answer RATE. A partial-coverage method")
        L.append("is penalised in the table above (unscorable pairs rank last, the query")
        L.append("still counts); here it is judged only where it actually answers.")
        L.append("=" * 100)
        hdr2 = (f"{'method':<22}{'scope':<11}{'nQ':>5}{'R@1':>8}{'mAP':>8}"
                f"{'DIR@.1':>8}{'assF1':>8}")
        L.append(hdr2)
        L.append("-" * len(hdr2))
        for r in sorted(rep["matched"], key=lambda x: -x["mAP"]):
            L.append(f"{r['method']:<22}{r['input_scope']:<11}{r['n_queries']:>5d}"
                     f"{r['rank1']:>8.3f}{r['mAP']:>8.3f}"
                     f"{r['dir_at_far10']:>8.3f}{r['assignment_f1']:>8.3f}")

        L.append("\nRANK CHANGE (full -> matched), by mAP:")
        full_order = [r["method"] for r in sorted(rep["full"], key=lambda x: -x["mAP"])]
        match_order = [r["method"] for r in sorted(rep["matched"], key=lambda x: -x["mAP"])]
        for m in full_order:
            a, b = full_order.index(m) + 1, match_order.index(m) + 1
            arrow = "same" if a == b else (f"{a} -> {b}  " + ("WORSE" if b > a else "BETTER"))
            L.append(f"   {m:<22} {arrow}")

    if "normalized" in rep:
        L.append("\n" + "=" * 100)
        L.append(f"CALIBRATED ({rep['norm_mode']}-normalised) -- per-query normalisation "
                 f"before applying the global threshold")
        L.append("R@1/mAP are UNCHANGED by construction (a per-row monotone transform cannot")
        L.append("reorder a row). Only the threshold-based metrics move. A large jump in")
        L.append("DIR@.1 means the method was losing to miscalibration, not to accuracy.")
        L.append("=" * 100)
        raw = {r["method"]: r for r in rep["full"]}
        hdr3 = (f"{'method':<22}{'R@1 raw':>9}{'R@1 norm':>10}{'DIR@.1 raw':>12}"
                f"{'DIR@.1 norm':>13}{'delta':>9}{'assF1 raw':>11}{'assF1 norm':>12}")
        L.append(hdr3)
        L.append("-" * len(hdr3))
        for r in sorted(rep["normalized"], key=lambda x: -x["dir_at_far10"]):
            b = raw[r["method"]]
            d = r["dir_at_far10"] - b["dir_at_far10"]
            L.append(f"{r['method']:<22}{b['rank1']:>9.3f}{r['rank1']:>10.3f}"
                     f"{b['dir_at_far10']:>12.3f}{r['dir_at_far10']:>13.3f}"
                     f"{d:>+9.3f}{b['assignment_f1']:>11.3f}{r['assignment_f1']:>12.3f}")
        drift = max(abs(r["rank1"] - raw[r["method"]]["rank1"]) for r in rep["normalized"])
        L.append(f"\n   max |R@1 raw - R@1 norm| = {drift:.2e}  "
                 f"(must be ~0; non-zero would mean the transform is not per-row monotone)")

    scopes = {r["input_scope"] for r in rep["full"]}
    if len(scopes) > 1:
        L.append(f"\nNOTE: methods span {len(scopes)} input scopes {sorted(scopes)}. "
                 f"A full-image method sees strictly more\n      than a crop method; "
                 f"only compare within a scope, or report the difference explicitly.")
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="benchmark output dir (contains scores/)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="bootstrap resamples for CIs; 0 disables")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frame-gap-sweep", action="store_true",
                    help="near-duplicate control: R@1/mAP as adjacent frames are "
                         "excluded (0..3). Free: works off the saved matrices. NOT the "
                         "viewpoint axis -- frame gap tracks camera roll only.")
    ap.add_argument("--viewpoint-sweep", action="store_true",
                    help="THE viewpoint axis: stratify by measured scale change, using "
                         "the method-independent covariate from viewpoint.py")
    ap.add_argument("--viewpoint-key", default="scale_change",
                    choices=["scale_change", "abs_rotation_deg", "tilt_deg", "projectivity"],
                    help="which viewpoint component to stratify on")
    ap.add_argument("--per-wall-coverage", action="store_true",
                    help="scored-pair rate per wall per method -- shows whether partial "
                         "coverage is noise or a scope condition")
    ap.add_argument("--fuse", nargs=2, metavar=("PRIMARY", "BACKUP"),
                    help="cascade two saved methods by name and report the result")
    ap.add_argument("--surfaces", metavar="WALLS_CSV",
                    help="path to a CSV with wall_id,surface -- reports the table "
                         "once per surface regime (textured render vs smooth paint)")
    ap.add_argument("--norm", default="row", choices=["none", "row", "rank"],
                    help="per-query score normalisation to report alongside raw")
    args = ap.parse_args()

    rep = analyse(args.out_dir, split=args.split, n_boot=args.bootstrap,
                  seed=args.seed, norm_mode=args.norm)
    text = format_report(rep)

    if args.frame_gap_sweep:
        tbl = frame_gap_sweep(args.out_dir, split=args.split)
        text += "\n" + format_frame_gap(tbl)
        rep["frame_gap_sweep"] = tbl

    if args.viewpoint_sweep:
        try:
            vs = viewpoint_sweep(args.out_dir, split=args.split, key=args.viewpoint_key)
            text += "\n" + format_viewpoint(vs)
            rep["viewpoint_sweep"] = vs
        except SystemExit as e:
            text += f"\n\n(viewpoint sweep skipped: {e})"

    if args.per_wall_coverage:
        pwc = per_wall_coverage(args.out_dir, split=args.split)
        text += "\n" + format_per_wall_coverage(pwc)
        rep["per_wall_coverage"] = pwc

    if args.surfaces:
        smap = load_surface_map(args.surfaces)
        if not smap:
            text += f"\n\n(stratification skipped: no `surface` column in {args.surfaces})"
        else:
            st = stratified(args.out_dir, smap, split=args.split)
            text += "\n" + format_stratified(st)
            rep["stratified"] = st

    if args.fuse:
        runs = {r.method: r for r in load_runs(args.out_dir, split=args.split)}
        missing = [m for m in args.fuse if m not in runs]
        if missing:
            text += f"\n\n(fusion skipped: no saved matrix for {missing})"
        else:
            pr, bk = runs[args.fuse[0]], runs[args.fuse[1]]
            problems = check_aligned([pr, bk])
            if problems:
                text += f"\n\n(fusion skipped: {problems})"
            else:
                valid, relevant = masks_for(pr)
                m = _metrics(fuse(pr, bk, valid), relevant, valid)
                text += (f"\n\nFUSION  {args.fuse[0]} -> {args.fuse[1]}\n"
                         f"  R@1={m['rank1']:.3f}  mAP={m['mAP']:.3f}  "
                         f"DIR@.1={m['dir_at_far10']:.3f}  scored={m['scoreable_pair_rate']:.2f}\n"
                         f"  (vs {args.fuse[0]} alone: "
                         f"R@1={_metrics(pr.scores, relevant, valid)['rank1']:.3f})")
                rep["fusion"] = {"primary": args.fuse[0], "backup": args.fuse[1], **m}

    print(text)
    with open(os.path.join(args.out_dir, f"analysis_{args.split}.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out_dir, f"analysis_{args.split}.json"), "w") as f:
        json.dump(rep, f, indent=2, default=float)
    print(f"\nWrote analysis_{args.split}.txt / .json to {args.out_dir}/")
