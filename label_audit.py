"""
Ground truth built by an automatic rule needs a validation number, the
same way a model does.

WHAT THE RULE IS
----------------
Identities in dataset/labels/ were not drawn by hand one photo at a time.
prefill_labels.py propagates a click point from photo to photo through an
ORB homography and a Hungarian assignment inside `assign_px`, and a
later merge step absorbed prefill identities into one another where their
components looked like fragments of one physical crack. labels/
_changelog.json records the result: on wall01, identity wall01_crack01
alone absorbed thirty prefill identities across 59 components and 8
photos, with a maximum merge gap of 312 px.

Merging is the right idea -- one physical crack really does split into
several connected components, and mAP is built to treat them all as
relevant. But a merge that joins two GENUINELY DISTINCT cracks converts a
hard negative into a free positive and inflates every method's mAP at
once, proposed and baseline alike. The paper therefore has to state the
rule, its gap threshold, and an audited error rate -- and there is
currently one annotator, no second pass, and no agreement figure.

WHAT THIS MODULE PROVIDES
-------------------------
    --report      the merge statistics the paper must state, per split
    --sample N    a reproducible random sample of identities to check,
                  with contact sheets rendered so the check actually
                  happens, and a JSON file to record verdicts in
    --score FILE  the audited error rate with a binomial interval
    --agree A B   inter-annotator agreement between two label directories

AGREEMENT IS A CLUSTERING MEASURE HERE, NOT A CLASSIFICATION ONE
----------------------------------------------------------------
Two annotators do not choose from a fixed label set; they PARTITION the
components of a wall into identities, and the identity strings they pick
are arbitrary. Cohen's kappa cannot be computed on that. The right
statistic is agreement over PAIRS of components -- did both annotators
put this pair in the same identity, or both in different ones -- summed
into the Adjusted Rand Index, which is chance-corrected. Report the ARI
per wall and the pairwise agreement rate beside it.

    python label_audit.py dataset --report
    python label_audit.py dataset --sample 30 --seed 0
    python label_audit.py dataset --score dataset/labels/_audit_0.json
    python label_audit.py dataset --agree dataset/labels dataset/labels_b

--agree ALSO MEASURES THE MERGE ITSELF
--------------------------------------
Point it at the pre-merge prefill output and it reports how much the merge
pass restructured the partition:

    python label_audit.py dataset --agree dataset/labels dataset/labels_old
    -> ARI 0.080 over 1051 matched points, 140 photos

0.080 is near-zero agreement: the merged labelling is almost unrelated to
the one it was built from, which is the same fact as "45% of identities
absorbed more than one prefill identity" seen from the other side. That is
not an argument against merging -- it is the reason the merge needs an
audited error rate before any mAP computed on it can be quoted.
"""

from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict

import numpy as np


# ===========================================================================
# Loading
# ===========================================================================

def _manifest(root: str) -> list[dict]:
    with open(os.path.join(root, "walls.csv")) as f:
        return list(csv.DictReader(f))


def _splits(root: str) -> dict:
    p = os.path.join(root, "splits.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def _split_of(root: str) -> dict:
    out = {}
    for name, walls in _splits(root).items():
        for w in walls:
            out[w] = name
    return out


def load_labels(label_dir: str) -> dict[str, list[dict]]:
    """{image_id: [point, ...]} for every label file in a directory."""
    out = {}
    for fn in sorted(os.listdir(label_dir)):
        if not fn.endswith(".json") or fn.startswith("_"):
            continue
        with open(os.path.join(label_dir, fn)) as f:
            d = json.load(f)
        out[d.get("image_id", fn[:-5])] = d.get("points", [])
    return out


def load_changelog(root: str) -> dict:
    p = os.path.join(root, "labels", "_changelog.json")
    return json.load(open(p)) if os.path.exists(p) else {}


# ===========================================================================
# 1. The merge rule, stated
# ===========================================================================

def merge_report(root: str) -> dict:
    """Per-split merge statistics -- the paragraph the paper owes S8."""
    log = load_changelog(root)
    split_of = _split_of(root)
    out = {}
    for wall, rec in sorted(log.items()):
        ids = rec.get("identities", {})
        for ident, meta in ids.items():
            absorbed = len(meta.get("absorbed_prefill_ids", []))
            out.setdefault(split_of.get(wall, "unassigned"), []).append({
                "wall": wall, "identity": ident,
                "n_components": meta.get("n_components", 0),
                "n_photos": meta.get("n_photos", 0),
                "max_merge_gap": meta.get("max_merge_gap", 0.0),
                "absorbed": absorbed,
                "merged": absorbed > 1,
            })
    return out


def format_merge_report(rep: dict, triage: list | None = None) -> str:
    L = ["", "=" * 84,
         "HOW THE GROUND TRUTH WAS BUILT -- state this in the paper, with a number",
         "=" * 84,
         f"{'split':<12}{'identities':>11}{'merged':>9}{'%':>6}"
         f"{'median gap':>12}{'p90 gap':>9}{'max gap':>9}"]
    L.append("-" * len(L[-1]))
    for split, rows in sorted(rep.items()):
        merged = [r for r in rows if r["merged"]]
        gaps = np.array([r["max_merge_gap"] for r in merged]) if merged else np.array([0.0])
        L.append(f"{split:<12}{len(rows):>11}{len(merged):>9}"
                 f"{100 * len(merged) / max(len(rows), 1):>6.0f}"
                 f"{np.median(gaps):>12.1f}{np.percentile(gaps, 90):>9.1f}{gaps.max():>9.1f}")
    L += ["",
          "'merged' = an identity that absorbed more than one prefill identity, i.e. one",
          "the automatic rule decided were fragments of a single physical crack.",
          "'gap' = the largest pixel distance bridged inside one identity (max_merge_gap).",
          "",
          "A merge that joins two DISTINCT cracks turns a hard negative into a free positive",
          "and inflates mAP for every method at once. So the paper must quote: the rule, the",
          "gap threshold, and an audited error rate over a random sample (--sample/--score)."]
    if triage:
        worst = sorted(triage, key=lambda t: -t.get("max_merge_gap", 0))[:8]
        L += ["", f"ALREADY FLAGGED BY TRIAGE: {len(triage)} identities. Widest gaps:",
              f"  {'identity':<22}{'photos':>7}{'comps':>7}{'gap px':>9}  reason"]
        for t in worst:
            L.append(f"  {t['identity']:<22}{t.get('n_photos', 0):>7}"
                     f"{t.get('n_components', 0):>7}{t.get('max_merge_gap', 0):>9.1f}"
                     f"  {t.get('reason', '')}")
        L.append("  These are where the audit sample should be weighted, and where a second")
        L.append("  annotator's disagreement will concentrate.")
    return "\n".join(L)


# ===========================================================================
# 2. A sample a human can actually check
# ===========================================================================

def sample_identities(root: str, n: int = 30, seed: int = 0,
                      split: str | None = None) -> list[dict]:
    """Reproducible stratified sample: half merged identities, half not.

    Stratified because the merged ones are where the error is, and an
    unstratified sample of 30 out of 271 would draw too few of them to
    estimate their error rate -- but the unmerged half is kept so the
    audit can also catch the opposite failure, one crack split in two.
    """
    rep = merge_report(root)
    rows = [r for split_name, rs in rep.items() for r in rs
            if split is None or split_name == split]
    merged = [r for r in rows if r["merged"]]
    plain = [r for r in rows if not r["merged"]]
    rng = random.Random(seed)
    take_m = min(len(merged), n // 2)
    take_p = min(len(plain), n - take_m)
    picked = rng.sample(merged, take_m) + rng.sample(plain, take_p)
    rng.shuffle(picked)
    for p in picked:
        p["verdict"] = ""          # one of: correct | over-merged | split | wrong-point
        p["note"] = ""
    return picked


def render_sample(root: str, picked: list[dict], out_dir: str,
                  max_dim: int = 1400) -> None:
    """One contact sheet per sampled identity: every photo it appears in,
    with its click points marked. An audit nobody can perform does not get
    performed, and reviewing 30 identities across 140 12-MP photos by hand
    in a viewer is exactly the task that gets skipped."""
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    rows = _manifest(root)
    by_wall = defaultdict(list)
    for r in rows:
        by_wall[r["wall_id"]].append(r)
    labels = load_labels(os.path.join(root, "labels"))

    for item in picked:
        ident, wall = item["identity"], item["wall"]
        tiles = []
        for r in sorted(by_wall[wall], key=lambda x: x["image_id"]):
            pts = [p for p in labels.get(r["image_id"], []) if p["identity"] == ident]
            if not pts:
                continue
            img = cv2.imread(os.path.join(root, r["path"]), cv2.IMREAD_COLOR)
            if img is None:
                continue
            s = min(1.0, max_dim / max(img.shape[:2]))
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            for p in pts:
                x, y = int(p["xy"][0] * s), int(p["xy"][1] * s)
                cv2.circle(img, (x, y), 14, (0, 0, 255), 3)
                cv2.drawMarker(img, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
            cv2.putText(img, r["image_id"], (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)
            tiles.append(img)
        if not tiles:
            continue
        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 6,
                                    cv2.BORDER_CONSTANT, value=(20, 20, 20)) for t in tiles]
        sheet = np.hstack(tiles)
        cv2.imwrite(os.path.join(out_dir, f"{ident}.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"  contact sheets -> {out_dir}/  "
          f"(open each, then fill `verdict` in the audit JSON)")


# ===========================================================================
# 3. The number the paper quotes
# ===========================================================================

VERDICTS = ("correct", "over-merged", "split", "wrong-point")


def score_audit(path: str) -> str:
    """Error rate over the filled-in audit file, with a Wilson interval."""
    with open(path) as f:
        items = json.load(f)["items"]
    done = [i for i in items if i.get("verdict")]
    bad_kinds = defaultdict(int)
    for i in done:
        if i["verdict"] != "correct":
            bad_kinds[i["verdict"]] += 1
    n, k = len(done), sum(bad_kinds.values())

    L = ["", "LABEL AUDIT",
         f"  reviewed        {n}/{len(items)} sampled identities"]
    if not n:
        L.append("  nothing scored yet: fill the `verdict` field "
                 f"({'/'.join(VERDICTS)}) in {path}")
        return "\n".join(L)
    p = k / n
    z = 1.96
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    L += [f"  errors          {k}  ({p:.1%})",
          f"  95% interval    [{max(0, centre - half):.1%}, {min(1, centre + half):.1%}]  (Wilson)"]
    for kind, c in sorted(bad_kinds.items(), key=lambda kv: -kv[1]):
        L.append(f"    {kind:<14}{c}")
    merged = [i for i in done if i.get("merged")]
    if merged:
        mk = sum(1 for i in merged if i["verdict"] != "correct")
        L.append(f"  of the {len(merged)} MERGED identities reviewed, {mk} were wrong "
                 f"({mk / len(merged):.0%})")
    L += ["", "  Quote the interval, not the point estimate, and say how many identities were",
          "  reviewed. An audited 5% label error rate is a stronger paper than an unaudited",
          "  claim of correctness, because a reviewer can price it into every number."]
    return "\n".join(L)


# ===========================================================================
# 4. Inter-annotator agreement
# ===========================================================================

def _partition(labels: dict[str, list[dict]], walls: set[str] | None = None
               ) -> dict[str, dict[str, str]]:
    """{wall: {point_key: identity}} keyed by a location that is annotator
    independent -- the image plus the rounded click coordinate. Two
    annotators do not click the same pixel, so points are matched to each
    other by nearest neighbour in `agreement`; this is the raw form."""
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for image_id, pts in labels.items():
        wall = image_id.split("_")[0]
        if walls and wall not in walls:
            continue
        for i, p in enumerate(pts):
            out[wall][f"{image_id}#{i}"] = (p["identity"], tuple(p["xy"]))
    return out


def _adjusted_rand(a: list[str], b: list[str]) -> float:
    """ARI between two labellings of the same items."""
    ua = {v: i for i, v in enumerate(sorted(set(a)))}
    ub = {v: i for i, v in enumerate(sorted(set(b)))}
    m = np.zeros((len(ua), len(ub)), dtype=np.int64)
    for x, y in zip(a, b):
        m[ua[x], ub[y]] += 1
    n = m.sum()
    if n < 2:
        return float("nan")

    def comb2(x):
        return (x * (x - 1) // 2).sum() if hasattr(x, "sum") else x * (x - 1) // 2

    sum_ij = comb2(m)
    sum_i = comb2(m.sum(1))
    sum_j = comb2(m.sum(0))
    total = n * (n - 1) // 2
    exp = sum_i * sum_j / total
    mx = (sum_i + sum_j) / 2
    return float((sum_ij - exp) / (mx - exp)) if mx != exp else float("nan")


def agreement(root: str, dir_a: str, dir_b: str, tol: float = 60.0) -> str:
    """Agreement between two annotators over the walls both labelled.

    Points are matched between annotators by nearest neighbour within
    `tol` px inside the same photo; unmatched points on either side are
    reported separately, because "annotator B saw a crack A did not" is a
    different disagreement from "they named the same crack differently"
    and only the second is what ARI measures.
    """
    A, B = load_labels(dir_a), load_labels(dir_b)
    shared_images = sorted(set(A) & set(B))
    walls = sorted({i.split("_")[0] for i in shared_images})
    if not shared_images:
        return f"\nno image labelled in both {dir_a} and {dir_b}"

    L = ["", "=" * 84,
         f"INTER-ANNOTATOR AGREEMENT   {dir_a}  vs  {dir_b}",
         "=" * 84,
         f"{'wall':<10}{'photos':>7}{'matched':>9}{'only A':>8}{'only B':>8}{'ARI':>8}"]
    L.append("-" * len(L[-1]))
    all_a, all_b = [], []
    for w in walls:
        la, lb, only_a, only_b = [], [], 0, 0
        for image_id in shared_images:
            if not image_id.startswith(w):
                continue
            pa, pb = A[image_id], B[image_id]
            used = set()
            for p in pa:
                best, bd = None, tol
                for j, q in enumerate(pb):
                    if j in used:
                        continue
                    d = float(np.hypot(p["xy"][0] - q["xy"][0], p["xy"][1] - q["xy"][1]))
                    if d < bd:
                        best, bd = j, d
                if best is None:
                    only_a += 1
                    continue
                used.add(best)
                la.append(p["identity"])
                lb.append(pb[best]["identity"])
            only_b += len(pb) - len(used)
        ari = _adjusted_rand(la, lb) if len(la) >= 2 else float("nan")
        all_a += la
        all_b += lb
        n_ph = sum(1 for i in shared_images if i.startswith(w))
        L.append(f"{w:<10}{n_ph:>7}{len(la):>9}{only_a:>8}{only_b:>8}{ari:>8.3f}")
    if len(all_a) >= 2:
        L.append("-" * 50)
        L.append(f"{'ALL':<10}{len(shared_images):>7}{len(all_a):>9}"
                 f"{'':>8}{'':>8}{_adjusted_rand(all_a, all_b):>8.3f}")
    L += ["", "ARI is chance-corrected agreement on the PARTITION of components into",
          "identities -- the right statistic here, because the identity strings themselves",
          "are arbitrary and the annotators are clustering, not classifying.",
          "'only A'/'only B' count cracks one annotator marked and the other did not; those",
          "are a detection disagreement, not a naming one, and belong in the paper separately."]
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--report", action="store_true", help="merge statistics per split")
    ap.add_argument("--sample", type=int, default=0,
                    help="draw N identities to audit and render contact sheets")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default=None, help="restrict the sample to one split")
    ap.add_argument("--score", metavar="AUDIT_JSON",
                    help="report the error rate from a filled-in audit file")
    ap.add_argument("--agree", nargs=2, metavar=("DIR_A", "DIR_B"),
                    help="inter-annotator agreement between two label directories")
    args = ap.parse_args()

    did = False
    if args.report:
        triage_path = os.path.join(args.root, "labels", "_triage.json")
        triage = json.load(open(triage_path)) if os.path.exists(triage_path) else None
        print(format_merge_report(merge_report(args.root), triage))
        did = True

    if args.sample:
        picked = sample_identities(args.root, n=args.sample, seed=args.seed,
                                   split=args.split)
        out = os.path.join(args.root, "labels", f"_audit_{args.seed}.json")
        with open(out, "w") as f:
            json.dump({"seed": args.seed, "split": args.split,
                       "verdicts": list(VERDICTS), "items": picked}, f, indent=1)
        print(f"\nsampled {len(picked)} identities "
              f"({sum(1 for p in picked if p['merged'])} merged) -> {out}")
        render_sample(args.root, picked,
                      os.path.join(args.root, "labels", f"_audit_{args.seed}_sheets"))
        did = True

    if args.score:
        print(score_audit(args.score))
        did = True

    if args.agree:
        print(agreement(args.root, *args.agree))
        did = True

    if not did:
        ap.print_help()
