"""
Pre-populate dataset/labels/*.json with provisional click points so
labelling a wall is review-and-fix instead of click-from-scratch.

Every point this writes carries "provisional": true and label_points.py
renders it dashed/orange with a "?" -- it is a GUESS, not a label, until a
human opens the wall in the labeller and either accepts it (`a`) or fixes
it (right-click to delete, left-click the right spot under the right
crack number). Disclose the use of this prefill step wherever the labels
get used, since it changes how the ground truth was produced.

--------------------------------------------------------------------------
HOW IT PROPAGATES IDENTITIES
--------------------------------------------------------------------------
Within one wall, photos are walked in image_id order. The first photo's
crack anchors (see component_anchors() in label_points.py -- same anchors
the interactive tool offers) each get a fresh identity. For every photo
after that:

  1. ORB features are matched between the previous photo and this one,
     and a homography is fit with RANSAC.
  2. The previous photo's points are projected through that homography
     into this photo's frame.
  3. Each projected point claims the crack COMPONENT it lands in, or the
     nearest component within --assign-px, and that component's anchor
     inherits the identity. Components nothing claims get a brand new
     identity (a crack newly in view, or a homography that didn't hold --
     a human sorts out which during review).

Step 3 deliberately does NOT match projected point to nearest anchor.
An anchor is a component's deepest pixel by distance transform, which on
an elongated crack slides along its length whenever segmentation shifts
slightly -- it is not a stable physical landmark. Measured over 456
projected points on high-overlap pairs: 80% land within 10px of a crack
PIXEL (median 0px), but only 55% land within 60px of an ANCHOR. Matching
on the component, not the anchor, is what makes propagation work.

This is a per-step (not chained-to-a-fixed-reference) propagation, so a
photo with a missing mask or a failed homography just breaks the chain
there -- the next photo's unmatched anchors get fresh identities instead
of silently inheriting wrong ones. That's a degradation the reviewer will
notice and fix, not a correctness risk.

Photos that already have points (hand-labelled, or an earlier prefill
run) are left untouched and used as-is to seed propagation into later
photos -- pass --force to overwrite them instead.

    python prefill_labels.py dataset --wall wall01
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import cv2
import numpy as np

from label_points import load_rows


def parse_crack_no(identity: str) -> int:
    m = re.search(r"crack(\d+)$", identity)
    return int(m.group(1)) if m else 0


def load_points(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return json.load(f).get("points", [])
    except (json.JSONDecodeError, OSError):
        return []


def estimate_homography(gray1: np.ndarray, gray2: np.ndarray,
                         min_inliers: int) -> np.ndarray | None:
    orb = cv2.ORB_create(4000)
    k1, d1 = orb.detectAndCompute(gray1, None)
    k2, d2 = orb.detectAndCompute(gray2, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in pairs if n is not None and m.distance < 0.75 * n.distance]
    if len(good) < min_inliers:
        return None
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    if H is None or int(inlier_mask.sum()) < min_inliers:
        return None
    return H


def project(H: np.ndarray, xys: list[list[int]]) -> np.ndarray:
    pts = np.float32(xys).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def labelled_components(mask: np.ndarray, min_area: int, close_px: int):
    """(label map, [(lid, anchor_xy)]) for components above min_area.

    Same closing / min_area / distance-transform anchor as
    component_anchors(), but the label map is kept so a projected point
    can be resolved to the component it lands in."""
    binary = (mask > 0).astype(np.uint8)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    dt = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    comps = []
    for lid in range(1, n):
        if stats[lid, cv2.CC_STAT_AREA] < min_area:
            continue
        sel = labels == lid
        d = np.where(sel, dt, -1.0)
        y, x = np.unravel_index(int(np.argmax(d)), d.shape)
        comps.append((lid, (int(x), int(y))))
    return labels, comps


def locate(labels: np.ndarray, lid_index: dict[int, int], px: float, py: float,
           assign_px: float):
    """Which component index a projected point claims, and at what distance.

    Inside a component -> distance 0. Otherwise the nearest crack pixel
    within assign_px wins, searched in a local window (cheap, and the
    window is exactly the acceptance radius so nothing outside it matters).
    Returns (component_index, distance) or (None, None)."""
    h, w = labels.shape[:2]
    xi, yi = int(round(px)), int(round(py))
    if 0 <= xi < w and 0 <= yi < h:
        lid = int(labels[yi, xi])
        if lid in lid_index:
            return lid_index[lid], 0.0

    r = int(assign_px)
    x0, x1 = max(xi - r, 0), min(xi + r + 1, w)
    y0, y1 = max(yi - r, 0), min(yi + r + 1, h)
    if x0 >= x1 or y0 >= y1:
        return None, None
    patch = labels[y0:y1, x0:x1]
    ys, xs = np.nonzero(patch)
    if len(ys) == 0:
        return None, None
    d = np.hypot((xs + x0) - px, (ys + y0) - py)
    order = np.argsort(d)
    for oi in order:
        if d[oi] > assign_px:
            break
        lid = int(patch[ys[oi], xs[oi]])
        if lid in lid_index:
            return lid_index[lid], float(d[oi])
    return None, None


def prefill_wall(root: str, rows: list[dict], min_area: int, close_px: int,
                  assign_px: float, min_inliers: int, force: bool) -> dict:
    stats = {"prefilled": 0, "skipped_existing": 0, "no_homography": 0,
             "points": 0, "propagated": 0}
    next_id = 1
    prev_gray: np.ndarray | None = None
    prev_points: list[dict] = []

    for r in rows:
        label_path = os.path.join(root, "labels", r["image_id"] + ".json")
        existing = load_points(label_path)
        img_path = os.path.join(root, r["path"])
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        if existing and not force:
            stats["skipped_existing"] += 1
            for pt in existing:
                next_id = max(next_id, parse_crack_no(pt["identity"]) + 1)
            prev_gray, prev_points = gray, existing
            continue

        mask_path = os.path.join(root, "masks", r["image_id"] + ".png")
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            labels, comps = labelled_components(m, min_area, close_px)
        else:
            labels, comps = None, []
        lid_index = {lid: i for i, (lid, _) in enumerate(comps)}

        assigned: list[dict] = []
        H = None
        if prev_points and comps:
            H = estimate_homography(prev_gray, gray, min_inliers)
        if H is None and prev_points:
            stats["no_homography"] += 1

        # component index -> (distance, identity); nearest projected point wins,
        # so two cracks merging into one component here can't both claim it
        claims: dict[int, tuple[float, str]] = {}
        if H is not None and prev_points:
            proj = project(H, [p["xy"] for p in prev_points])
            for prev_pt, (px, py) in zip(prev_points, proj):
                ci, d = locate(labels, lid_index, px, py, assign_px)
                if ci is None:
                    continue
                if ci not in claims or d < claims[ci][0]:
                    claims[ci] = (d, prev_pt["identity"])
                    stats["propagated"] += 1

        for idx, (lid, (ax, ay)) in enumerate(comps):
            if idx in claims:
                identity = claims[idx][1]
            else:
                identity = f"{r['wall_id']}_crack{next_id:02d}"
                next_id += 1
            assigned.append({"identity": identity, "xy": [ax, ay],
                              "provisional": True})

        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        with open(label_path, "w") as f:
            json.dump({"image_id": r["image_id"], "points": assigned}, f, indent=2)

        stats["prefilled"] += 1
        stats["points"] += len(assigned)
        prev_gray, prev_points = gray, assigned

    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="dataset root (contains walls.csv)")
    ap.add_argument("--wall", nargs="*", help="only prefill these wall_ids")
    ap.add_argument("--min-area", type=int, default=200,
                    help="must match label_points.py / benchmark.py's Dataset(min_area=...)")
    ap.add_argument("--close-px", type=int, default=5,
                    help="must match label_points.py / benchmark.py's Dataset(close_px=...)")
    ap.add_argument("--assign-px", type=float, default=40.0,
                    help="how far (original-image px) a projected point may sit from a "
                         "crack component and still claim it; 0 means it must land "
                         "inside one")
    ap.add_argument("--min-inliers", type=int, default=12,
                    help="min RANSAC inlier matches to trust a homography")
    ap.add_argument("--force", action="store_true",
                    help="overwrite photos that already have points, "
                         "INCLUDING hand-verified ones -- use with care")
    a = ap.parse_args()

    rows = load_rows(a.root)
    if a.wall:
        rows = [r for r in rows if r["wall_id"] in set(a.wall)]
    if not rows:
        raise SystemExit("no photos match")

    by_wall = defaultdict(list)
    for r in rows:
        by_wall[r["wall_id"]].append(r)

    totals = defaultdict(int)
    for wall_id in sorted(by_wall):
        wall_rows = sorted(by_wall[wall_id], key=lambda r: r["image_id"])
        stats = prefill_wall(a.root, wall_rows, a.min_area, a.close_px,
                              a.assign_px, a.min_inliers, a.force)
        for k, v in stats.items():
            totals[k] += v
        print(f"{wall_id}: {stats['prefilled']} photo(s) prefilled "
              f"({stats['points']} provisional points, "
              f"{stats['propagated']} propagated), "
              f"{stats['skipped_existing']} already labelled, "
              f"{stats['no_homography']} homography failure(s)")

    print(f"\nTOTAL: {totals['prefilled']} photos, {totals['points']} provisional "
          f"points, {totals['propagated']} carried across photos.")
    print("Next: python label_points.py <root> --wall <id>   (press 'a' to accept "
          "a photo's provisional points once you've checked them, or fix wrong ones "
          "the usual way)")


if __name__ == "__main__":
    main()
