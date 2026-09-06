"""What every metric scores when the method does nothing.

Run:  python chance_baseline.py dataset --out banchmark_out

Writes `<out>/chance.json`, which `paper/verify_claims.py` reads and the
paper quotes. Cheap -- it needs no score matrices, only the dataset and
(for the coverage analysis) `pair_outcomes.json`.

--------------------------------------------------------------------------
WHY THIS EXISTS
--------------------------------------------------------------------------
Three of the numbers in the benchmark's first published table were at or
below chance, and nobody noticed, because no chance level was ever
computed. The specific trap is pairwise F1.

`pair_pr_curve` reports best-F1 over a threshold grid. That maximisation
includes the degenerate threshold at which every pair is predicted
positive, which scores 2p/(1+p) for positive prevalence p REGARDLESS of
the scores. On the CrackID test split p = 0.180, so pairwise F1 cannot go
below 0.305 -- and the ten benchmarked crop-scope methods span
[0.305, 0.359]. The "0.054-wide band across five method families" is a
band pressed against a floor, and three methods (ORB, DeiT, CrackShape)
sit exactly on it.

Read as excess over chance the same table says something sharper and
better supported: crop appearance buys between 0.000 and 0.054 on the
decision a maintenance system actually makes, while geometry buys 0.35.

--------------------------------------------------------------------------
THE SECOND TRAP: PREVALENCE IS NOT CONSTANT ACROSS METHODS
--------------------------------------------------------------------------
The floor is a property of the POOL, so it moves when the pool moves. The
geometric method scores only the pairs it can register, and registration
succeeds precisely when the two photographs overlap -- which is also when
the answer is present. Measured here: the scorable subset keeps 100% of
positive pairs and 37% of negatives, so prevalence rises 0.180 -> 0.373
and the floor rises 0.305 -> 0.544.

That is why `reid_eval.pair_pr_curve` now CHARGES unscorable pairs instead
of dropping them, and why `benchmark.py --ablations` ships a
`coverage-only` baseline: a method that answers YES to every registered
pair and never looks at a crack scores 0.544 here.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reid_eval import (  # noqa: E402
    assignment_accuracy, build_relevance, build_validity_mask, chance_pair_f1,
    closed_set_metrics, dir_at_far, frame_index, open_set_curve, pair_pr_curve,
)

PROTOCOL = dict(same_wall_only=True, exclude_same_session=True)


def _refs(root: str, split: str):
    """Rebuild the split's InstanceRefs without re-extracting instances."""
    from benchmark import Dataset
    splits = json.load(open(os.path.join(root, "splits.json")))
    data = Dataset(root)
    return data.query_gallery(set(splits[split]))


def chance_levels(relevant, valid, q_image, g_image, n_repeats=200, seed=0):
    """Every metric, scored on pure noise, `n_repeats` times.

    Reported as mean +/- sd. pairwise F1's sd is ~1e-4 because it is a
    floor rather than an average -- random scores do not beat it, they
    meet it.
    """
    rng = np.random.default_rng(seed)
    acc = collections.defaultdict(list)
    for _ in range(n_repeats):
        s = rng.standard_normal(valid.shape)
        cs = closed_set_metrics(s, relevant, valid)
        acc["rank1"].append(cs.rank1)
        acc["rank5"].append(cs.rank5)
        acc["mAP"].append(cs.mAP)
        acc["dir_at_far10"].append(dir_at_far(open_set_curve(s, relevant, valid), 0.1))
        pr = pair_pr_curve(s, relevant, valid)
        acc["pair_f1"].append(pr["best_f1"])
        thr = pr["thresholds"][int(np.argmax(pr["f1"]))]
        acc["assign_f1"].append(
            assignment_accuracy(s, relevant, valid, thr,
                                q_image=q_image, g_image=g_image)["f1"])
    return {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                "n": n_repeats} for k, v in acc.items()}


def frame_gap_structure(queries, gallery, relevant):
    """The pool at each min_frame_gap, and how far the nearest answer is.

    At min_frame_gap=0 -- the setting the paper's Table I reports -- the
    nearest correct answer is the IMMEDIATELY ADJACENT frame for 100% of
    answerable test queries, a median of 3 s of wall-clock apart. Every
    ranking number is therefore a near-duplicate retrieval result unless
    the sweep below is reported beside it.
    """
    out = {"per_gap": {}, "nearest_answer_frame_distance": {}}
    rng = np.random.default_rng(0)
    for gap in (0, 1, 2, 3):
        v = build_validity_mask(queries, gallery, min_frame_gap=gap, **PROTOCOL)
        pos = int((relevant & v).sum())
        tot = int(v.sum())
        p = pos / tot if tot else 0.0
        # Chance Rank-1 moves with the gap too, and in the opposite direction
        # to intuition: excluding near frames shrinks the POSITIVE side of
        # each row faster than the gallery, so chance falls. A method whose
        # absolute Rank-1 merely holds across the sweep is therefore gaining
        # on chance, not treading water -- which cannot be seen without this.
        r1 = [closed_set_metrics(rng.standard_normal(v.shape), relevant, v).rank1
              for _ in range(25)]
        out["per_gap"][str(gap)] = {
            "answerable_queries": int((relevant & v).any(1).sum()),
            "valid_pairs": tot, "positive_pairs": pos,
            "prevalence": p, "chance_pair_f1": (2 * p / (1 + p)) if p else 0.0,
            "chance_rank1": float(np.mean(r1)),
            "chance_rank1_sd": float(np.std(r1, ddof=1)),
        }
    v0 = build_validity_mask(queries, gallery, min_frame_gap=0, **PROTOCOL)
    dists = []
    for i in range(len(queries)):
        fi = frame_index(queries[i].image_id)
        if fi is None:
            continue
        d = [abs(frame_index(gallery[j].image_id) - fi)
             for j in np.nonzero(relevant[i] & v0[i])[0]
             if frame_index(gallery[j].image_id) is not None]
        if d:
            dists.append(min(d))
    c = collections.Counter(dists)
    out["nearest_answer_frame_distance"] = {
        "n": len(dists),
        "histogram": {str(k): c[k] for k in sorted(c)},
        "median": float(np.median(dists)) if dists else None,
        "frac_adjacent": float(np.mean([d == 1 for d in dists])) if dists else None,
    }
    return out


def coverage_confound(queries, gallery, relevant, valid, outcomes_path):
    """How registration coverage correlates with the label.

    Registration is not missing-at-random with respect to the answer: two
    photographs register when they overlap, and they overlap when the
    crack is in both. So restricting pairwise F1 to the scorable subset --
    which the old implementation did by dropping non-finite cells -- is
    selection on the label.
    """
    if not os.path.exists(outcomes_path):
        return None
    raw = json.load(open(outcomes_path))["registered"]
    reg = {}
    for k, ok in raw.items():
        a, b = k.split("|")
        reg[(a, b)] = reg[(b, a)] = bool(ok)
    fin = np.zeros(valid.shape, bool)
    for i, q in enumerate(queries):
        for j in np.nonzero(valid[i])[0]:
            fin[i, j] = reg.get((q.image_id, gallery[j].image_id), False)

    pos_all = int((relevant & valid).sum())
    neg_all = int((valid & ~relevant).sum())
    pos_fin = int((relevant & fin).sum())
    neg_fin = int((fin & ~relevant).sum())
    p_fin = pos_fin / (pos_fin + neg_fin) if (pos_fin + neg_fin) else 0.0

    known = (relevant & valid).any(1)
    rows = valid.any(1)
    dead = rows & ~fin.any(1)
    return {
        "valid_pairs": pos_all + neg_all,
        "scorable_pairs": pos_fin + neg_fin,
        "positives_retained": pos_fin / pos_all if pos_all else 0.0,
        "negatives_retained": neg_fin / neg_all if neg_all else 0.0,
        "prevalence_full_pool": pos_all / (pos_all + neg_all),
        "prevalence_scorable_pool": p_fin,
        "chance_pair_f1_full_pool": 2 * (pos_all / (pos_all + neg_all))
                                    / (1 + pos_all / (pos_all + neg_all)),
        "chance_pair_f1_scorable_pool": (2 * p_fin / (1 + p_fin)) if p_fin else 0.0,
        # The exact score of "answer YES to every registered pair", which
        # never looks at a crack. TP = every positive, FP = every scorable
        # negative, FN = 0.
        "coverage_only_pair_f1": (2 * pos_fin / (2 * pos_fin + neg_fin)
                                  if pos_fin else 0.0),
        # Queries the OLD open_set_curve dropped, by answerability.
        "queries_total": int(rows.sum()),
        "queries_known": int(known.sum()),
        "queries_unknown": int((rows & ~known).sum()),
        "dropped_known": int((dead & known).sum()),
        "dropped_unknown": int((dead & ~known).sum()),
        "dropped_by_wall": dict(collections.Counter(
            queries[i].wall_id for i in np.nonzero(dead)[0])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--out", default="banchmark_out")
    ap.add_argument("--split", default="test")
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    queries, gallery = _refs(args.root, args.split)
    valid = build_validity_mask(queries, gallery, min_frame_gap=0, **PROTOCOL)
    relevant = build_relevance(queries, gallery)
    q_image = [r.image_id for r in queries]
    g_image = [r.image_id for r in gallery]

    floor = chance_pair_f1(relevant, valid)
    print(f"\npool: {len(queries)} queries, {int(valid.sum())} valid pairs, "
          f"{int((relevant & valid).sum())} positive "
          f"(prevalence {(relevant & valid).sum() / valid.sum():.4f})")
    print(f"analytic pairwise-F1 floor 2p/(1+p) = {floor:.4f}\n")

    print(f"scoring {args.repeats} random matrices ...")
    levels = chance_levels(relevant, valid, q_image, g_image,
                           n_repeats=args.repeats, seed=args.seed)
    for k, v in levels.items():
        print(f"  {k:14s} {v['mean']:.4f} +/- {v['sd']:.4f}")

    gaps = frame_gap_structure(queries, gallery, relevant)
    na = gaps["nearest_answer_frame_distance"]
    print(f"\nnearest correct answer is the ADJACENT frame for "
          f"{na['frac_adjacent']:.1%} of {na['n']} answerable queries")
    for g, d in gaps["per_gap"].items():
        print(f"  min_frame_gap={g}: {d['answerable_queries']:4d} answerable, "
              f"prevalence {d['prevalence']:.4f}, floor {d['chance_pair_f1']:.4f}")

    conf = coverage_confound(queries, gallery, relevant, valid,
                             os.path.join(args.out, "pair_outcomes.json"))
    if conf:
        print(f"\nregistration coverage vs the label:")
        print(f"  positives retained {conf['positives_retained']:.1%}, "
              f"negatives retained {conf['negatives_retained']:.1%}")
        print(f"  prevalence {conf['prevalence_full_pool']:.4f} -> "
              f"{conf['prevalence_scorable_pool']:.4f}   "
              f"floor {conf['chance_pair_f1_full_pool']:.4f} -> "
              f"{conf['chance_pair_f1_scorable_pool']:.4f}")
        print(f"  'YES to every registered pair' scores pairwise F1 "
              f"{conf['coverage_only_pair_f1']:.4f}")
        print(f"  old open_set_curve dropped {conf['dropped_unknown']} of "
              f"{conf['queries_unknown']} unknown queries and "
              f"{conf['dropped_known']} of {conf['queries_known']} known")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "chance.json")
    with open(path, "w") as f:
        json.dump({
            "split": args.split, "seed": args.seed, "repeats": args.repeats,
            "n_queries": len(queries), "n_gallery": len(gallery),
            "valid_pairs": int(valid.sum()),
            "positive_pairs": int((relevant & valid).sum()),
            "prevalence": float((relevant & valid).sum() / valid.sum()),
            "analytic_pair_f1_floor": floor,
            "chance": levels,
            "frame_gap": gaps,
            "coverage_confound": conf,
        }, f, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
