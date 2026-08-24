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

from scipy.optimize import linear_sum_assignment

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


MAX_DIM = 1600            # detect on a downscale; 12 MP buys nothing here


def estimate_homography(gray1: np.ndarray, gray2: np.ndarray,
                         min_inliers: int) -> np.ndarray | None:
    """Homography from photo 1 to photo 2, or None.

    Three fixes over the original:

      * CLAHE + downscale before detection. These are handheld frames with
        autoexposure drift between shots, and ORB was running on the full
        4080x3072 image at roughly 30 s/pair for no accuracy gain.

      * knnMatch results are length-checked. BFMatcher returns fewer than
        k neighbours when the train set is small, and `for m, n in pairs`
        raises on those; the sibling matchers in this repo all guard it.

      * The homography is SANITY-CHECKED. crack_registration_reid ships
        _homography_is_sane and this never called it, so a folded or
        wildly rescaled matrix with >= min_inliers passed straight
        through and projected points into nonsense -- minting confidently
        WRONG identities, which is worse for ground truth than minting
        none. Measured: 3% of accepted homographies are degenerate,
        including the only one accepted anywhere on wall14.
    """
    from crack_registration_reid import _homography_is_sane

    s1 = min(1.0, MAX_DIM / max(gray1.shape[:2]))
    s2 = min(1.0, MAX_DIM / max(gray2.shape[:2]))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    small1 = clahe.apply(cv2.resize(gray1, None, fx=s1, fy=s1, interpolation=cv2.INTER_AREA))
    small2 = clahe.apply(cv2.resize(gray2, None, fx=s2, fy=s2, interpolation=cv2.INTER_AREA))

    orb = cv2.ORB_create(4000)
    k1, d1 = orb.detectAndCompute(small1, None)
    k2, d2 = orb.detectAndCompute(small2, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = [pr for pr in bf.knnMatch(d1, d2, k=2) if len(pr) == 2]
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
    if len(good) < min_inliers:
        return None
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H_small, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    if H_small is None or inlier_mask is None or int(inlier_mask.sum()) < min_inliers:
        return None

    # back to full resolution: H = S2^-1 @ H_small @ S1
    H = np.diag([1.0 / s2, 1.0 / s2, 1.0]) @ H_small @ np.diag([s1, s1, 1.0])
    H /= H[2, 2]
    sane, _ = _homography_is_sane(H, gray1.shape[:2])
    return H if sane else None


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

    r = int(np.ceil(assign_px))   # ceil: the window is square, acceptance circular
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
                  assign_px: float, min_inliers: int, force: bool,
                  force_all: bool = False) -> dict:
    stats = {"prefilled": 0, "skipped_existing": 0, "no_homography": 0,
             "points": 0, "propagated": 0, "empty_photos": 0, "no_mask": 0,
             "skipped_accepted": 0}
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

        # A photo is protected when a human has ACCEPTED it: label_points.py
        # pops "provisional" on accept, so an accepted point is exactly one
        # without that flag.
        #
        # This used to be all-or-nothing on --force, which was a trap: every
        # point already exists, so a re-run does nothing WITHOUT --force, and
        # WITH it overwrites human-verified work too. Improving this script
        # and re-running would silently destroy a review pass. Now --force
        # regenerates only still-provisional photos, and accepted ones stay
        # put while still seeding propagation into later frames.
        accepted = bool(existing) and not any(pt.get("provisional") for pt in existing)
        if existing and (not force or (accepted and not force_all)):
            stats["skipped_existing"] += 1
            if accepted:
                stats["skipped_accepted"] += 1
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
            stats["no_mask"] += 1
        lid_index = {lid: i for i, (lid, _) in enumerate(comps)}

        assigned: list[dict] = []
        H = None
        if prev_points and comps:
            H = estimate_homography(prev_gray, gray, min_inliers)
        if H is None and prev_points and comps:
            stats["no_homography"] += 1     # only counted when one was ATTEMPTED

        # component index -> (distance, identity), resolved as a GLOBAL
        # one-to-one assignment rather than greedily. Nearest-wins dropped
        # the loser silently: its identity died and was reborn as a fresh id
        # in the next photo, manufacturing a false negative. The Hungarian
        # solver is already a dependency of this repo.
        claims: dict[int, tuple[float, str]] = {}
        if H is not None and prev_points:
            proj = project(H, [p["xy"] for p in prev_points])
            cand: list[tuple[int, int, float]] = []      # (prev idx, comp idx, dist)
            for pi, (px, py) in enumerate(proj):
                ci, d = locate(labels, lid_index, px, py, assign_px)
                if ci is not None:
                    cand.append((pi, ci, d))
            if cand:
                rows = sorted({c[0] for c in cand})
                cols = sorted({c[1] for c in cand})
                ri = {v: i for i, v in enumerate(rows)}
                cj = {v: j for j, v in enumerate(cols)}
                BIG = float(assign_px) * 10.0
                cost = np.full((len(rows), len(cols)), BIG, dtype=float)
                for pi, ci, d in cand:
                    cost[ri[pi], cj[ci]] = min(cost[ri[pi], cj[ci]], d)
                for a_i, b_j in zip(*linear_sum_assignment(cost)):
                    if cost[a_i, b_j] >= BIG:
                        continue
                    claims[cols[b_j]] = (float(cost[a_i, b_j]),
                                         prev_points[rows[a_i]]["identity"])
            # count once per surviving claim, not once per improvement
            stats["propagated"] += len(claims)

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
        # Only advance the anchor when this photo actually has something to
        # propagate. Twelve photos in the dataset yield no component above
        # min_area, and setting prev_points = [] on those split their wall
        # into disjoint identity namespaces: the next photo found no anchor,
        # attempted no homography, and minted every crack fresh. wall05 lost
        # 4 of 10 photos that way, wall14 3 of 9, wall16 2 of 14.
        if assigned:
            prev_gray, prev_points = gray, assigned
        else:
            stats["empty_photos"] += 1

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
                    help="regenerate photos whose points are still provisional. "
                         "Photos a human has accepted in label_points.py are kept "
                         "and still seed propagation, so re-running after improving "
                         "this script cannot destroy review work.")
    ap.add_argument("--force-all", action="store_true",
                    help="overwrite EVERY photo, including human-accepted ones. "
                         "This throws away review work -- use only to rebuild a "
                         "wall from scratch.")
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
                              a.assign_px, a.min_inliers, a.force or a.force_all,
                              a.force_all)
        for k, v in stats.items():
            totals[k] += v
        print(f"{wall_id}: {stats['prefilled']} photo(s) prefilled "
              f"({stats['points']} provisional points, "
              f"{stats['propagated']} propagated), "
              f"{stats['skipped_existing']} already labelled "
              f"({stats['skipped_accepted']} human-accepted, kept), "
              f"{stats['no_homography']} homography failure(s), "
              f"{stats['empty_photos']} photo(s) with no crack above min_area, "
              f"{stats['no_mask']} missing mask(s)")

    print(f"\nTOTAL: {totals['prefilled']} photos, {totals['points']} provisional "
          f"points, {totals['propagated']} carried across photos.")
    print("Next: python label_points.py <root> --wall <id>   (press 'a' to accept "
          "a photo's provisional points once you've checked them, or fix wrong ones "
          "the usual way)")


if __name__ == "__main__":
    main()
