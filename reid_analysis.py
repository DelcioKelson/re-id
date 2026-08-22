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
        )
        if split and r.split != split:
            continue
        runs[r.key] = r          # later mtime overwrites an earlier re-run
    return list(runs.values())


# ===========================================================================
# Protocol masks, rebuilt from the saved arrays
# ===========================================================================

def masks_for(run: ScoreRun, same_wall_only: bool = True,
              exclude_same_session: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(valid, relevant), matching reid_eval.build_validity_mask exactly but
    vectorised, since the analysis rebuilds these for every bootstrap."""
    q, g = run.query, run.gallery
    same_image = q["image_id"][:, None] == g["image_id"][None, :]
    valid = ~same_image
    if exclude_same_session:
        valid &= q["session"][:, None] != g["session"][None, :]
    if same_wall_only:
        valid &= q["wall_id"][:, None] == g["wall_id"][None, :]

    q_id, g_id = q["identity"], g["identity"]
    relevant = (q_id[:, None] == g_id[None, :]) & (q_id[:, None] != "")
    return valid, relevant


def _metrics(scores, relevant, valid, threshold=None) -> dict:
    closed = closed_set_metrics(scores, relevant, valid)
    openset = open_set_curve(scores, relevant, valid)
    pr = pair_pr_curve(scores, relevant, valid)
    if threshold is None:
        idx = int(np.argmax(pr["f1"])) if pr["f1"] else 0
        threshold = pr["thresholds"][idx] if pr["thresholds"] else 0.0
    assign = assignment_accuracy(scores, relevant, valid, threshold)
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
        m = _metrics(r.scores, relevant, valid, threshold=thr)
        m["method"] = r.method
        m["input_scope"] = r.input_scope
        m["total_seconds"] = r.prepare_seconds + r.score_seconds
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
            m = _metrics(ns, relevant, valid, threshold=thr)
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
            m = _metrics(r.scores, relevant, v)
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
    ap.add_argument("--norm", default="row", choices=["none", "row", "rank"],
                    help="per-query score normalisation to report alongside raw")
    args = ap.parse_args()

    rep = analyse(args.out_dir, split=args.split, n_boot=args.bootstrap,
                  seed=args.seed, norm_mode=args.norm)
    text = format_report(rep)
    print(text)
    with open(os.path.join(args.out_dir, f"analysis_{args.split}.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out_dir, f"analysis_{args.split}.json"), "w") as f:
        json.dump(rep, f, indent=2, default=float)
    print(f"\nWrote analysis_{args.split}.txt / .json to {args.out_dir}/")
