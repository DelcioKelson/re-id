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
    python label_audit.py dataset --unresolved

--unresolved NEEDS NO ANNOTATOR, AND IS THE BIGGER NUMBER
--------------------------------------------------------
--sample estimates a label error rate from 30 identities and needs a human
to fill in verdicts. --unresolved needs nobody: it runs the benchmark's own
resolution rule and reports what the ground truth actually reaches.

Two figures come out, and they must not be confused.

The click count is large (over half the points) and mostly harmless. The
label files place several points on one crack per photo, and
Dataset._resolve_identity consumes one click per connected component, so a
photo with 44 clicks on 3 cracks is not 41 errors. Some are literal
duplicates of another point in the same file. Only this one is ground truth
leaving the experiment:

    COMPONENTS DROPPED FROM EVALUATION  67/1051 (6.4%)

A component whose nearest click is further than point_tolerance gets
identity=None, and benchmark.py drops identity=None refs from the query set.
Those cracks are then in no denominator: no method is penalised for missing
them, and they cannot be found either.

The contested count is the one that threatens the RESULTS rather than the
completeness:

    contested         437/1051 (41.6% of all components)
    labels erased     445

_resolve_identity returns on the FIRST point inside a component, so when a
component carries clicks from two identities the second label is discarded
with no trace in any output file. Either the merge joined two distinct
cracks -- a hard negative turned into a free positive, inflating mAP for
every method at once -- or the segmentation fused two cracks into one
component, in which case the labels are right and the mask is wrong. The
two need opposite fixes and the benchmark cannot separate them, so
--unresolved renders one sheet per contested component with the kept label
drawn as a cross and the erased one as a hollow circle, and writes both
lists to dataset/labels/_unresolved.json for verdicts.

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
        doc = json.load(f)
    items = doc["items"]
    # the file names its own verdict vocabulary, so --score works for the
    # identity sample and the unresolved/contested files alike
    allowed = doc.get("verdicts", list(VERDICTS))
    ok_verdict = "correct" if "correct" in allowed else allowed[0]
    noun = doc.get("noun", "identities")
    done = [i for i in items if i.get("verdict")]
    bad_kinds = defaultdict(int)
    for i in done:
        if i["verdict"] != ok_verdict:
            bad_kinds[i["verdict"]] += 1
    n, k = len(done), sum(bad_kinds.values())

    L = ["", "LABEL AUDIT",
         f"  reviewed        {n}/{len(items)} sampled {noun}"]
    if not n:
        L.append("  nothing scored yet: fill the `verdict` field "
                 f"({'/'.join(allowed)}) in {path}")
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
        mk = sum(1 for i in merged if i["verdict"] != ok_verdict)
        L.append(f"  of the {len(merged)} MERGED identities reviewed, {mk} were wrong "
                 f"({mk / len(merged):.0%})")
    L += ["", "  Quote the interval, not the point estimate, and say how many identities were",
          "  reviewed. An audited 5% label error rate is a stronger paper than an unaudited",
          "  claim of correctness, because a reviewer can price it into every number."]
    return "\n".join(L)


# ===========================================================================
# 3b. Points that resolve to no component
# ===========================================================================

UNRESOLVED_VERDICTS = ("mask-missed-crack", "point-misplaced", "not-a-crack")


def _resolve_components(data, image_id: str,
                        points: list[dict]) -> list[dict]:
    """Per component: which point won, and which points fall inside it.

    A copy of Dataset._resolve_identity that keeps the bookkeeping the
    original discards. Two questions need it and neither can be answered from
    the identity string alone: which point the component took, and which
    points it covers. A second point of the SAME identity inside the same
    component is redundant, not wrong -- counting it as a defect inflated the
    unresolved total to 68%, which is a bug in the audit, not in the labels.
    """
    out = []
    for inst in data.instances(image_id, apply_mask=False):
        x0, y0, w, h = inst.bbox
        crop = inst.mask_crop
        inside, best_i, best_d = [], None, data.point_tolerance + 1
        for i, p in enumerate(points):
            px, py = p["xy"]
            lx, ly = px - x0, py - y0
            if 0 <= lx < crop.shape[1] and 0 <= ly < crop.shape[0] \
                    and crop[ly, lx] > 0:
                inside.append(i)
            cx, cy = x0 + w / 2, y0 + h / 2
            d = float(np.hypot(px - cx, py - cy))
            if d < best_d:
                best_i, best_d = i, d
        if inside:
            win = inside[0]                     # exact path: first point inside
        elif best_i is not None and best_d <= data.point_tolerance:
            win = best_i                         # near-miss: nearest to bbox centre
        else:
            win = None
        out.append({
            "instance_id": f"{image_id}_c{len(out):02d}",
            "bbox": [int(x0), int(y0), int(w), int(h)],
            "winner": win,
            "inside": inside,
            "identity": points[win]["identity"] if win is not None else None,
            "area": int((crop > 0).sum()),
        })
    return out


def _mask_distance(mask: np.ndarray, x: int, y: int, radius: int) -> float:
    """Distance in px from (x, y) to the nearest mask pixel, capped at radius."""
    h, w = mask.shape[:2]
    if not (0 <= y < h and 0 <= x < w):
        return float(radius + 1)
    if mask[y, x] > 0:
        return 0.0
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    sub = (mask[y0:y1, x0:x1] > 0)
    if not sub.any():
        return float(radius + 1)
    ys, xs = np.nonzero(sub)
    return float(np.min(np.hypot(ys - (y - y0), xs - (x - x0))))


def unresolved_points(root: str, min_area: int = 200, close_px: int = 5,
                      point_tolerance: int = 25) -> tuple[list[dict], dict]:
    """Every ground-truth point that no component claims, with the reason.

    benchmark.py drops a component whose nearest point is further than
    point_tolerance, and drops the resulting identity=None ref from the query
    set. So an unresolvable point is not a cosmetic defect: the crack it
    names is absent from the evaluation, and a method is never penalised for
    missing it. The number below is the size of that hole in the benchmark.
    """
    from benchmark import Dataset

    data = Dataset(root, min_area=min_area, close_px=close_px,
                   point_tolerance=point_tolerance)
    rows = _manifest(root)
    by_id = {r["image_id"]: r for r in rows}
    items: list[dict] = []
    n_pts = n_orphan = 0
    n_comp = n_unlabelled = 0
    n_dup = 0
    orphan_photos: set[str] = set()

    for image_id, photo in sorted(data.photos.items()):
        path = os.path.join(root, "labels", f"{image_id}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            points = json.load(f).get("points", [])
        if not points:
            continue
        n_pts += len(points)
        seen = defaultdict(int)
        for p in points:
            seen[(p["identity"], tuple(p["xy"]))] += 1
        n_dup += sum(c - 1 for c in seen.values())
        comps = _resolve_components(data, image_id, points)
        mask = data.mask(image_id)

        effective: set[int] = set()
        n_comp += len(comps)
        n_unlabelled += sum(1 for c in comps if c["winner"] is None)
        for c in comps:
            if c["winner"] is None:
                continue
            effective.add(c["winner"])
            for i in c["inside"]:              # same identity, same crack: fine
                if points[i]["identity"] == c["identity"]:
                    effective.add(i)
        n_orphan += len(points) - len(effective)
        if len(effective) < len(points):
            orphan_photos.add(image_id)

        for i, p in enumerate(points):
            if i in effective:
                continue
            x, y = p["xy"]
            d = _mask_distance(mask, x, y, radius=point_tolerance * 3)
            if d == 0.0:
                reason = "on-mask-but-unclaimed"
            elif d > point_tolerance:
                reason = "no-mask-within-tolerance"
            else:
                reason = "near-mask-but-unclaimed"
            items.append({
                "image_id": image_id,
                "wall": by_id.get(image_id, {}).get("wall_id", image_id[:6]),
                "identity": p["identity"],
                "xy": [x, y],
                "provisional": bool(p.get("provisional")),
                "dist_to_mask_px": round(d, 1),
                "mask_px": int((mask > 0).sum()),
                "n_components": len(comps),
                "reason": reason,
                "verdict": "",
                "note": "",
            })

    stats = {
        "points_total": n_pts,
        "points_unresolved": n_orphan,
        "photos_total": len(data.photos),
        "photos_affected": len(orphan_photos),
        "components_total": n_comp,
        "components_unlabelled": n_unlabelled,
        "duplicate_points": n_dup,
    }
    return items, stats


def render_unresolved(root: str, items: list[dict], out_dir: str,
                      tile: int = 360, cols: int = 5) -> None:
    """One tile per unresolvable point: a zoom on the click, mask overlaid.

    The mask is drawn in red so the audit question is answerable at a glance:
    is there a crack under the crosshair that the segmentation missed (fix the
    mask), or is the crosshair on bare wall (fix the label)?
    """
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    by_photo: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_photo[it["image_id"]].append(it)

    manifest = {r["image_id"]: os.path.join(root, r["path"]) for r in _manifest(root)}
    n_sheets = 0
    for image_id in sorted(by_photo):
        img = cv2.imread(manifest[image_id], cv2.IMREAD_COLOR)
        if img is None:
            continue
        mask = cv2.imread(os.path.join(root, "masks", f"{image_id}.png"),
                          cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape[:2] != img.shape[:2]:
            mask = np.zeros(img.shape[:2], np.uint8)
        overlay = img.copy()
        red = np.array([0, 0, 255], np.float32)
        m3 = (mask > 127)
        overlay[m3] = (0.55 * img[m3] + 0.45 * red).astype(np.uint8)

        tiles = []
        for it in sorted(by_photo[image_id], key=lambda i: i["dist_to_mask_px"]):
            x, y = it["xy"]
            half = tile // 2
            x0, x1 = max(0, x - half), min(img.shape[1], x + half)
            y0, y1 = max(0, y - half), min(img.shape[0], y + half)
            t = overlay[y0:y1, x0:x1].copy()
            if t.shape[0] < tile or t.shape[1] < tile:
                t = cv2.copyMakeBorder(t, 0, max(0, tile - t.shape[0]),
                                       0, max(0, tile - t.shape[1]),
                                       cv2.BORDER_CONSTANT, value=(20, 20, 20))
            cx, cy = x - x0, y - y0
            cv2.line(t, (cx - 26, cy), (cx + 26, cy), (0, 255, 255), 2)
            cv2.line(t, (cx, cy - 26), (cx, cy + 26), (0, 255, 255), 2)
            cv2.circle(t, (cx, cy), 13, (0, 0, 255), 2)
            cv2.putText(t, f"{it['dist_to_mask_px']:.0f}px", (6, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(t, it["identity"][-8:], (6, tile - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            tiles.append(t)

        for start in range(0, len(tiles), cols):
            row = tiles[start:start + cols]
            while len(row) < cols:
                row.append(np.full((tile, tile, 3), 20, np.uint8))
            sheet = np.hstack(row)
            banner = np.full((34, sheet.shape[1], 3), 32, np.uint8)
            cv2.putText(banner, f"{image_id}  {len(by_photo[image_id])} unresolved"
                        f"  (sheet {start // cols + 1})", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2)
            out = np.vstack([banner, sheet])
            cv2.imwrite(os.path.join(out_dir, f"{image_id}_{start // cols:02d}.jpg"),
                        out, [cv2.IMWRITE_JPEG_QUALITY, 90])
            n_sheets += 1
    print(f"  {n_sheets} sheets -> {out_dir}/")


def contested_components(root: str, min_area: int = 200, close_px: int = 5,
                          point_tolerance: int = 25) -> tuple[list[dict], dict]:
    """Components carrying clicks from more than one identity.

    This is the over-merge signature, and unlike the sampled identity audit
    it is exact and needs no annotator. Dataset._resolve_identity takes the
    FIRST point that lands inside a component and returns, discarding the
    rest. So on a component holding clicks from two identities, one label
    wins and the other is erased with no trace in any output file. Two very
    different faults produce that signature and the benchmark cannot tell
    them apart:

      a) the merge joined two distinct cracks, so one component carries both
         their labels -- a hard negative has become a free positive, which
         inflates mAP for every method at once;
      b) the segmentation fused two cracks into one component, so two
         correctly distinct labels collide on a mask artefact -- here the
         LABELS are right and the mask is wrong.

    (a) and (b) need opposite fixes, so the number is a routing measurement,
    not a verdict. It says where to look, and how much of the benchmark the
    answer rests on.
    """
    from benchmark import Dataset

    data = Dataset(root, min_area=min_area, close_px=close_px,
                   point_tolerance=point_tolerance)
    items: list[dict] = []
    n_comp = 0
    by_wall: dict[str, int] = defaultdict(int)
    for image_id in sorted(data.photos):
        path = os.path.join(root, "labels", f"{image_id}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            points = json.load(f).get("points", [])
        for c in _resolve_components(data, image_id, points):
            n_comp += 1
            ids = sorted({points[i]["identity"] for i in c["inside"]})
            if len(ids) < 2:
                continue
            by_wall[image_id[:6]] += 1
            items.append({
                "image_id": image_id,
                "wall": image_id[:6],
                "instance_id": c["instance_id"],
                "bbox": c["bbox"],
                "n_identities": len(ids),
                "identities": ids,
                "winner": c["identity"],
                "erased": [i for i in ids if i != c["identity"]],
                "area_px": c["area"],
                "verdict": "",
                "note": "",
            })
    return items, {"components": n_comp, "contested": len(items),
                   "erased_labels": sum(i["n_identities"] - 1 for i in items),
                   "by_wall": dict(sorted(by_wall.items(), key=lambda kv: -kv[1]))}


CONTESTED_VERDICTS = ("over-merged", "mask-fused", "correct")


def format_unresolved(stats: dict, items: list[dict],
                      contested: dict | None = None,
                      contested_items: list[dict] | None = None) -> str:
    n, tot = stats["points_unresolved"], stats["points_total"]
    nc, nu = stats.get("components_total", 0), stats.get("components_unlabelled", 0)
    L = ["", "=" * 84,
         "POINTS THAT RESOLVE TO NO COMPONENT",
         "=" * 84,
         f"  unclaimed clicks  {n}/{tot} points ({n / max(1, tot):.1%})",
         f"  photos affected   {stats['photos_affected']}/{stats['photos_total']}",
         f"  COMPONENTS DROPPED FROM EVALUATION  {nu}/{nc} ({nu / max(1, nc):.1%})"]
    by_reason = defaultdict(int)
    for it in items:
        by_reason[it["reason"]] += 1
    for reason, c in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        L.append(f"    {reason:<28}{c}")
    thin = sum(1 for i in items if i["mask_px"] < 200)
    dup = stats.get("duplicate_points", 0)
    L += ["",
          "  READ THE TWO NUMBERS SEPARATELY. The click count is mostly redundancy: the label",
          "  files place several points on one crack per photo, and one click per component is",
          f"  all the benchmark consumes. {dup} of the unclaimed clicks are exact duplicates of",
          "  another point in the same file, which is pure noise in the label JSON. Only the",
          "  COMPONENT line is ground truth that leaves the experiment: those components get",
          "  identity=None, are dropped from the query set (benchmark.py:311), and are in no",
          "  denominator -- so no method is penalised for missing them.",
          "",
          "  no-mask-within-tolerance is the segmentation's fault: no crack is drawn anywhere",
          f"  near the click. {thin} of these clicks sit in photos whose whole mask is under",
          "  200 px. Verdicts: " + " / ".join(UNRESOLVED_VERDICTS) + "."]

    if contested is not None:
        c, ci = contested["contested"], contested_items or []
        L += ["", "=" * 84,
              "COMPONENTS CARRYING MORE THAN ONE IDENTITY -- the over-merge signature",
              "=" * 84,
              f"  contested         {c}/{contested['components']} "
              f"({c / max(1, contested['components']):.1%} of all components)",
              f"  labels erased     {contested['erased_labels']}  "
              "(the loser's click leaves no trace in any output file)"]
        by_n = defaultdict(int)
        for it in ci:
            by_n[it["n_identities"]] += 1
        L.append("    identities per contested component: "
                 + ", ".join(f"{k}->{v}" for k, v in sorted(by_n.items())))
        L.append("    worst walls: "
                 + ", ".join(f"{w} {n}" for w, n in
                             list(contested["by_wall"].items())[:8]))
        L += ["",
              "  _resolve_identity returns on the FIRST point inside a component, so a second",
              "  identity on the same component is discarded silently. Two faults look identical",
              "  here and need opposite fixes:",
              "    over-merged  the merge joined two DISTINCT cracks -- a hard negative became a",
              "                free positive, inflating mAP for every method at once;",
              "    mask-fused   the segmentation fused two cracks into one component -- the",
              "                labels are right and the mask is wrong.",
              "  This is exact and needs no annotator, so unlike --sample it is not an estimate.",
              "  Verdict each one: " + " / ".join(CONTESTED_VERDICTS) + "."]
    return "\n".join(L)


def render_contested(root: str, items: list[dict], out_dir: str,
                     tile: int = 420, cols: int = 4) -> None:
    """One tile per contested component: every identity's click, in its own colour.

    This is the sheet that answers over-merged vs mask-fused. Two clicks of
    different colours on one connected blob means the question is real; the
    colour tells you which label the benchmark kept, and the loser is drawn
    hollow so the erased label is the one you judge.
    """
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    palette = [(0, 255, 255), (255, 160, 0), (0, 255, 0), (200, 0, 255)]
    by_wall: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_wall[it["wall"]].append(it)

    paths = {r["image_id"]: os.path.join(root, r["path"]) for r in _manifest(root)}
    n_sheets = 0
    for wall in sorted(by_wall):
        img_cache: dict[str, np.ndarray] = {}
        mask_cache: dict[str, np.ndarray] = {}
        tiles = []
        for it in by_wall[wall]:
            iid = it["image_id"]
            if iid not in img_cache:
                img = cv2.imread(paths[iid], cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img_cache[iid] = img
                m = cv2.imread(os.path.join(root, "masks", f"{iid}.png"),
                               cv2.IMREAD_GRAYSCALE)
                mask_cache[iid] = (m if m is not None
                                   and m.shape[:2] == img.shape[:2]
                                   else np.zeros(img.shape[:2], np.uint8))
            img, mask = img_cache[iid], mask_cache[iid]
            with open(os.path.join(root, "labels", f"{iid}.json")) as f:
                points = json.load(f)["points"]

            bx, by, bw, bh = it["bbox"]
            half = max(tile // 2, max(bw, bh) // 2 + 40)
            cx, cy = bx + bw // 2, by + bh // 2
            x0 = max(0, min(cx - half, img.shape[1] - tile))
            y0 = max(0, min(cy - half, img.shape[0] - tile))
            x1, y1 = min(img.shape[1], x0 + tile), min(img.shape[0], y0 + tile)
            x0, y0 = max(0, x1 - tile), max(0, y1 - tile)
            t = img[y0:y1, x0:x1].copy()
            comp = (mask[y0:y1, x0:x1] > 0) & (
                (np.arange(y0, y1)[:, None] >= by)
                & (np.arange(y0, y1)[:, None] < by + bh)
                & (np.arange(x0, x1)[None, :] >= bx)
                & (np.arange(x0, x1)[None, :] < bx + bw))
            contours, _ = cv2.findContours(comp.astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(t, contours, -1, (0, 0, 255), 2)

            for k, ident in enumerate(it["identities"]):
                col = palette[k % len(palette)]
                for p in points:
                    if p["identity"] != ident:
                        continue
                    px, py = p["xy"][0] - x0, p["xy"][1] - y0
                    if not (0 <= px < t.shape[1] and 0 <= py < t.shape[0]):
                        continue
                    if ident == it["winner"]:       # kept: filled
                        cv2.drawMarker(t, (px, py), col, cv2.MARKER_CROSS, 26, 3)
                    else:                          # erased by the benchmark: hollow
                        cv2.circle(t, (px, py), 11, col, 2)
                cv2.putText(t, str(k + 1), (t.shape[1] - 26, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            cv2.putText(t, it["instance_id"][-8:], (6, t.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            tiles.append(t)

        for start in range(0, len(tiles), cols):
            row = tiles[start:start + cols]
            while len(row) < cols:
                row.append(np.full((tile, tile, 3), 20, np.uint8))
            sheet = np.hstack(row)
            banner = np.full((56, sheet.shape[1], 3), 32, np.uint8)
            cv2.putText(banner, f"{wall}  contested components {len(by_wall[wall])}"
                        f"  sheet {start // cols + 1}", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2)
            cv2.putText(banner, "cross = label the benchmark kept   "
                        "circle = label it discarded   red = component",
                        (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (170, 200, 235), 1)
            cv2.imwrite(os.path.join(out_dir, f"{wall}_{start // cols:02d}.jpg"),
                        np.vstack([banner, sheet]), [cv2.IMWRITE_JPEG_QUALITY, 90])
            n_sheets += 1
    print(f"  {n_sheets} contested sheets -> {out_dir}/")


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


# ===========================================================================
# 4. Inter-annotator agreement
# ===========================================================================

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
    ap.add_argument("--unresolved", action="store_true",
                    help="audit points that no component claims, and render zoom sheets")
    ap.add_argument("--min-area", type=int, default=200)
    ap.add_argument("--close-px", type=int, default=5)
    ap.add_argument("--point-tolerance", type=int, default=25)
    args = ap.parse_args()

    did = False
    if args.report:
        triage_path = os.path.join(args.root, "labels", "_triage.json")
        triage = json.load(open(triage_path)) if os.path.exists(triage_path) else None
        print(format_merge_report(merge_report(args.root), triage))
        did = True

    if args.unresolved:
        items, stats = unresolved_points(args.root, min_area=args.min_area,
                                        close_px=args.close_px,
                                        point_tolerance=args.point_tolerance)
        cont, cont_stats = contested_components(
            args.root, min_area=args.min_area, close_px=args.close_px,
            point_tolerance=args.point_tolerance)
        print(format_unresolved(stats, items, cont_stats, cont))
        out = os.path.join(args.root, "labels", "_unresolved.json")
        with open(out, "w") as f:
            json.dump({"point_tolerance": args.point_tolerance,
                       "stats": stats, "contested_stats": cont_stats,
                       "verdicts": list(UNRESOLVED_VERDICTS), "noun": "clicks",
                       "contested_verdicts": list(CONTESTED_VERDICTS),
                       "contested_noun": "components",
                       "items": items, "contested": cont}, f, indent=1)
        print(f"\n  -> {out}")
        render_unresolved(args.root, items,
                          os.path.join(args.root, "labels", "_unresolved_sheets"))
        render_contested(args.root, cont,
                         os.path.join(args.root, "labels", "_contested_sheets"))
        did = True

    if args.sample:
        picked = sample_identities(args.root, n=args.sample, seed=args.seed,
                                   split=args.split)
        out = os.path.join(args.root, "labels", f"_audit_{args.seed}.json")
        with open(out, "w") as f:
            json.dump({"seed": args.seed, "split": args.split,
                       "verdicts": list(VERDICTS), "noun": "identities",
                       "items": picked}, f, indent=1)
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
