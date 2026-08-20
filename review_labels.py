"""
Audit the click points in dataset/labels/ the way benchmark.py will read them.

The labeller writes points; benchmark.py never uses them directly. It walks
the connected components of each mask and asks _resolve_identity() which
point (if any) claims each component. That indirection is where labels go
quietly wrong -- a point can be written to JSON and still name nothing, or
two points can land in one component and one of them is silently dropped.
This reports what survives that resolution step, so the numbers you read
here are the numbers the benchmark gets.

Resolution is replicated from benchmark.py Dataset._resolve_identity:
a point inside the component's own label mask claims it outright (first
such point in file order wins); otherwise the nearest point by distance
from the component's padded bbox CENTRE claims it, if within
--point-tolerance. Components no point claims become distractors.

    python review_labels.py dataset
    python review_labels.py dataset --wall wall02 --verbose
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np

from label_points import load_rows


def components_from_mask(mask: np.ndarray, min_area: int, close_px: int,
                         pad: int = 10) -> list[dict]:
    """Replicates extract_crack_instances() geometry without needing the photo.

    Only bbox and the per-label mask crop matter for identity resolution,
    so the BGR crop that extract_crack_instances() builds is skipped -- it
    would mean loading 140 full-resolution photos for nothing."""
    binary = (mask > 0).astype(np.uint8)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h_img, w_img = binary.shape[:2]
    out = []
    for lid in range(1, n_labels):
        area = int(stats[lid, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y, w, h = stats[lid, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT + 4]
        x0, y0 = max(int(x) - pad, 0), max(int(y) - pad, 0)
        x1, y1 = min(int(x) + int(w) + pad, w_img), min(int(y) + int(h) + pad, h_img)
        out.append({"bbox": (x0, y0, x1 - x0, y1 - y0), "area": area,
                    "mask_crop": (labels[y0:y1, x0:x1] == lid)})
    return out


def resolve(comp: dict, points: list[dict], tolerance: float):
    """(identity, mode, point_index) for one component.

    mode is 'inside' | 'near' | None -- mirrors _resolve_identity's two
    acceptance paths so the audit can tell a solid label from one that only
    scraped in on the distance fallback."""
    x0, y0, w, h = comp["bbox"]
    mc = comp["mask_crop"]
    best = (None, None, None)
    best_d = tolerance + 1
    for idx, p in enumerate(points):
        px, py = p["xy"]
        lx, ly = px - x0, py - y0
        if 0 <= lx < mc.shape[1] and 0 <= ly < mc.shape[0] and mc[ly, lx]:
            return p["identity"], "inside", idx
        cx, cy = x0 + w / 2, y0 + h / 2
        d = float(np.hypot(px - cx, py - cy))
        if d < best_d:
            best, best_d = (p["identity"], "near", idx), d
    return best if best[0] is not None else (None, None, None)


def audit(root: str, rows: list[dict], min_area: int, close_px: int,
          tolerance: float, verbose: bool):
    per_photo = []
    # identity -> wall -> set(image_id) and list of component areas
    id_photos: dict[str, set] = defaultdict(set)
    id_areas: dict[str, list] = defaultdict(list)
    id_wall: dict[str, set] = defaultdict(set)

    n_points = n_prov = 0
    tot_comps = tot_claimed = tot_inside = tot_near = 0
    orphan_points = []          # points that claim no component at all
    dup_in_photo = []           # same identity on 2+ components of one photo
    contested = []              # 2+ points inside one component (all but one lost)

    for r in rows:
        image_id = r["image_id"]
        lpath = os.path.join(root, "labels", image_id + ".json")
        mpath = os.path.join(root, "masks", image_id + ".png")
        try:
            with open(lpath) as f:
                points = json.load(f).get("points", [])
        except (json.JSONDecodeError, OSError, FileNotFoundError):
            points = []
        mask = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            per_photo.append({"image_id": image_id, "wall": r["wall_id"],
                              "comps": 0, "points": len(points), "claimed": 0,
                              "inside": 0, "near": 0, "no_mask": True})
            continue

        comps = components_from_mask(mask, min_area, close_px)
        n_points += len(points)
        n_prov += sum(1 for p in points if p.get("provisional"))

        claimed_by = defaultdict(list)          # point index -> [comp index]
        ident_hits = defaultdict(list)          # identity -> [comp index]
        inside = near = 0
        for ci, c in enumerate(comps):
            ident, mode, pidx = resolve(c, points, tolerance)
            c["identity"] = ident
            if ident is None:
                continue
            claimed_by[pidx].append(ci)
            ident_hits[ident].append(ci)
            if mode == "inside":
                inside += 1
            else:
                near += 1
            id_photos[ident].add(image_id)
            id_areas[ident].append(c["area"])
            id_wall[ident].add(r["wall_id"])

        # points nothing resolved to: written to JSON, invisible to benchmark
        for pi, p in enumerate(points):
            if pi not in claimed_by:
                orphan_points.append((image_id, p["identity"], tuple(p["xy"])))
        for ident, cis in ident_hits.items():
            if len(cis) > 1:
                dup_in_photo.append((image_id, ident, len(cis)))
        # more points inside one component than the one that won it
        for ci, c in enumerate(comps):
            x0, y0, w, h = c["bbox"]
            mc = c["mask_crop"]
            hits = []
            for pi, p in enumerate(points):
                lx, ly = p["xy"][0] - x0, p["xy"][1] - y0
                if 0 <= lx < mc.shape[1] and 0 <= ly < mc.shape[0] and mc[ly, lx]:
                    hits.append(p["identity"])
            if len(set(hits)) > 1:
                contested.append((image_id, ci, hits))

        claimed = inside + near
        tot_comps += len(comps)
        tot_claimed += claimed
        tot_inside += inside
        tot_near += near
        per_photo.append({"image_id": image_id, "wall": r["wall_id"],
                          "comps": len(comps), "points": len(points),
                          "claimed": claimed, "inside": inside, "near": near,
                          "no_mask": False})

    # ---- report ----
    print("=" * 74)
    print("KEYPOINT REVIEW  (as benchmark.py resolves them)")
    print("=" * 74)
    print(f"photos                {len(rows)}")
    print(f"click points in JSON  {n_points}"
          f"{f'   ({n_prov} still provisional/unreviewed)' if n_prov else ''}")
    print(f"mask components       {tot_comps}")
    pct = (100.0 * tot_claimed / tot_comps) if tot_comps else 0.0
    print(f"components labelled   {tot_claimed}  ({pct:.1f}%)   "
          f"-> {tot_comps - tot_claimed} distractors")
    print(f"  resolved 'inside'   {tot_inside}"
          f"   ({100.0 * tot_inside / tot_claimed:.1f}% of labelled)"
          if tot_claimed else "")
    print(f"  resolved 'near'     {tot_near}   (fallback within {tolerance:g}px "
          f"of bbox centre -- fragile)")

    print("\n" + "-" * 74)
    print("IDENTITY CHAINS  (an identity must appear in >=2 photos to be scorable)")
    print("-" * 74)
    lens = defaultdict(int)
    for ident, photos in id_photos.items():
        lens[len(photos)] += 1
    singles = lens.get(1, 0)
    multi = sum(v for k, v in lens.items() if k >= 2)
    print(f"identities resolved   {len(id_photos)}")
    print(f"  in 1 photo only     {singles}   (unusable: no answer exists)")
    print(f"  in >=2 photos       {multi}   (scorable)")
    if lens:
        dist = "  ".join(f"{k}:{v}" for k, v in sorted(lens.items()))
        print(f"  chain length dist   {dist}")

    print("\n" + "-" * 74)
    print("RED FLAGS")
    print("-" * 74)
    print(f"points claiming no component      {len(orphan_points)}")
    print(f"identity on 2+ comps in one photo {len(dup_in_photo)}")
    print(f"components with 2+ rival points   {len(contested)}")

    # a chain whose component area swings wildly is a likely mis-propagation
    wild = []
    for ident, areas in id_areas.items():
        if len(areas) >= 2:
            lo, hi = min(areas), max(areas)
            if lo > 0 and hi / lo >= 8.0:
                wild.append((ident, len(areas), lo, hi, hi / lo))
    wild.sort(key=lambda t: -t[4])
    print(f"chains with >=8x area swing       {len(wild)}   "
          f"(same crack should not change size that much)")

    cross = [(i, sorted(w)) for i, w in id_wall.items() if len(w) > 1]
    print(f"identities spanning 2+ walls      {len(cross)}   (should be 0)")

    if verbose:
        for title, items in (("points claiming no component", orphan_points[:25]),
                             ("identity on 2+ comps in one photo", dup_in_photo[:25]),
                             ("components with rival points", contested[:25]),
                             ("wild area swings", wild[:25]),
                             ("cross-wall identities", cross[:25])):
            if items:
                print(f"\n  {title}:")
                for it in items:
                    print(f"    {it}")

    print("\n" + "-" * 74)
    print("PER WALL")
    print("-" * 74)
    by_wall = defaultdict(list)
    for p in per_photo:
        by_wall[p["wall"]].append(p)
    wall_ids = defaultdict(set)
    for ident, photos in id_photos.items():
        for w in id_wall[ident]:
            wall_ids[w].add(ident)
    print(f"{'wall':8} {'photos':>6} {'comps':>6} {'lab':>5} {'%':>6} "
          f"{'ids':>5} {'>=2':>5}")
    for w in sorted(by_wall):
        ps = by_wall[w]
        c = sum(p["comps"] for p in ps)
        l = sum(p["claimed"] for p in ps)
        ids = wall_ids.get(w, set())
        m = sum(1 for i in ids if len(id_photos[i]) >= 2)
        print(f"{w:8} {len(ps):>6} {c:>6} {l:>5} "
              f"{(100.0 * l / c if c else 0):>5.1f}% {len(ids):>5} {m:>5}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="dataset root (contains walls.csv)")
    ap.add_argument("--wall", nargs="*", help="only audit these wall_ids")
    ap.add_argument("--min-area", type=int, default=200,
                    help="must match benchmark.py's Dataset(min_area=...)")
    ap.add_argument("--close-px", type=int, default=5,
                    help="must match benchmark.py's Dataset(close_px=...)")
    ap.add_argument("--point-tolerance", type=float, default=25.0,
                    help="must match benchmark.py's Dataset(point_tolerance=...)")
    ap.add_argument("--verbose", action="store_true", help="list the flagged items")
    a = ap.parse_args()

    rows = load_rows(a.root)
    if a.wall:
        rows = [r for r in rows if r["wall_id"] in set(a.wall)]
    if not rows:
        raise SystemExit("no photos match")
    rows.sort(key=lambda r: r["image_id"])
    audit(a.root, rows, a.min_area, a.close_px, a.point_tolerance, a.verbose)


if __name__ == "__main__":
    main()
