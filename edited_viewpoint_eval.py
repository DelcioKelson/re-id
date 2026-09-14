"""
Combined structural-edit + genuine-viewpoint discrimination evaluation.

For each source photograph, this creates a synthetic QUERY by:
  1. Shortening the crack mask by a chosen fraction (structural edit)
  2. Inpainting the removed crack region in the image (so embedding
     methods see plausible wall texture, not a black hole)
  3. Warping both the edited image and mask by a known affine
     viewpoint transform (genuine geometric change)

The query is then scored against the REAL gallery (other photographs of
the same wall) using the standard reid_eval.evaluate() pipeline.  Both
methods — skeleton-loftr and osnet@ctx1 — face identical inputs, so
Rank-1, mAP, DIR@FAR, and pairwise F1 are directly comparable in the
same units.

Unlike illustrative_comparison.py (which compares a source against its
own near-clone), this is a discrimination task: the gallery contains
hard negatives (other cracks on the same wall) and both methods can be
wrong.  Unlike synthetic_viewpoint.py alone, the structural edit adds a
controlled crack-identity change on top of the viewpoint change, so the
experiment measures how each method degrades when the crack itself
differs — the scenario Fig.~\\ref{fig:evolution} hypothesises about.

Usage:
    python edited_viewpoint_eval.py dataset --out edit_viewpoint_out \\
        --edit-fracs 0.0,0.25,0.50,0.75 \\
        --scales 1.0,1.5 --rotations 0,15 --tilts 0,20 \\
        --methods skeleton-loftr osnet@ctx1 \\
        --min-sharpness 10 --max-sources-per-wall 4

Requires the same deps as benchmark.py for whichever --methods you pass.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from collections import defaultdict

import cv2
import numpy as np

from benchmark import Dataset, Photo, PROTOCOL, build_scorers as build_benchmark_scorers
from crack_reid_baselines import skeletonize_mask, extract_crack_instances, CrackInstance
from reid_eval import evaluate, InstanceRef
from synthetic_viewpoint import make_transform, _selftest


# ===========================================================================
# 1. Structural edit: shorten a crack mask by removing pixels from one end
# ===========================================================================

def shorten_crack_mask(mask: np.ndarray, frac_removed: float,
                       close_px: int = 5) -> np.ndarray:
    """Remove the bottom ``frac_removed`` fraction of a crack mask.

    The crack's principal axis is found by PCA on the skeleton.  Skeleton
    pixels whose projection onto that axis falls in the lowest
    ``frac_removed`` of the total projection range are deleted.  The
    remaining skeleton is dilated back to a mask of roughly the original
    width.

    Parameters
    ----------
    mask : uint8
        Binary crack mask (0 or 255).
    frac_removed : float in [0, 1)
        Fraction of the crack length to remove from one end.  0.0 returns
        the mask unchanged; 0.75 removes three-quarters.
    close_px : int
        Morphological closing width, matching ``extract_crack_instances``.
    """
    if frac_removed <= 0.0:
        return mask.copy()

    binary = (mask > 0).astype(np.uint8)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n_labels <= 1:
        return mask.copy()

    # Largest component.
    lid = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    # Skeleton of the largest component.
    comp = (labels == lid).astype(np.uint8)
    skel = skeletonize_mask(comp * 255)
    sy, sx = np.nonzero(skel)
    if len(sx) < 4:
        return mask.copy()

    pts = np.column_stack([sx, sy]).astype(np.float64)
    centre = pts.mean(axis=0)
    cov = np.cov((pts - centre).T) + 1e-9 * np.eye(2)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, 1]  # principal axis (largest eigenvalue)

    proj = pts @ axis
    proj_min, proj_max = float(proj.min()), float(proj.max())
    proj_range = proj_max - proj_min
    if proj_range < 1e-6:
        return mask.copy()

    cutoff = proj_min + frac_removed * proj_range
    keep = proj >= cutoff

    # Rebuild a mask from the remaining skeleton, dilated to roughly the
    # original component width.
    remaining = np.zeros_like(comp)
    remaining[sy[keep], sx[keep]] = 1

    # Estimate average half-width from the full component.
    dist_full = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
    avg_radius = float(dist_full[comp > 0].mean()) if comp.any() else 3.0
    dilate_r = max(1, int(round(avg_radius)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (2 * dilate_r + 1, 2 * dilate_r + 1))
    edited = cv2.dilate(remaining, k, iterations=1)

    # Clip to the original mask extent so the edit doesn't spread into
    # areas the original crack didn't cover.
    edited = edited & comp
    return (edited * 255).astype(np.uint8)


def inpaint_shortened(image: np.ndarray, original_mask: np.ndarray,
                      edited_mask: np.ndarray,
                      dilate_px: int = 5) -> np.ndarray:
    """Inpaint the region removed by the structural edit.

    The inpaint region is the dilated difference between the original and
    edited masks: pixels the edit erased, plus a small border to avoid
    hard edges.  cv2.inpaint fills the gap with plausible wall texture so
    embedding methods see a natural shorter crack rather than a black
    hole.
    """
    orig_bin = (original_mask > 0).astype(np.uint8)
    edit_bin = (edited_mask > 0).astype(np.uint8)
    diff = cv2.dilate(
        orig_bin - edit_bin,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px)),
    )
    if diff.sum() == 0:
        return image.copy()
    return cv2.inpaint(image, diff, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


# ===========================================================================
# 2. Dataset: real photos + synthetic edited-and-warped queries
# ===========================================================================

class EditedViewpointDataset(Dataset):
    """Adds synthetic queries that combine a structural edit with a
    known viewpoint transform, on top of the real dataset.

    ``image()`` / ``mask()`` are overridden to serve cached edited+warped
    versions on first access — everything downstream (instances, crop,
    the registration/embedding scorers) goes through those two accessors
    polymorphically and needs no changes.
    """

    def __init__(self, *a, **kw):
        self._synth_recipe: dict[str, dict] = {}
        self._synth_points: dict[str, list[dict]] = {}
        self.synth_transform: dict[str, dict] = {}
        super().__init__(*a, **kw)

    def image(self, image_id: str) -> np.ndarray:
        if image_id in self._synth_recipe and image_id not in self._img_cache:
            r = self._synth_recipe[image_id]
            self._img_cache[image_id] = cv2.warpAffine(
                r["edited_image"], r["H"][:2, :],
                (r["edited_image"].shape[1], r["edited_image"].shape[0]),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return super().image(image_id)

    def mask(self, image_id: str) -> np.ndarray:
        if image_id in self._synth_recipe and image_id not in self._mask_cache:
            r = self._synth_recipe[image_id]
            self._mask_cache[image_id] = cv2.warpAffine(
                r["edited_mask"], r["H"][:2, :],
                (r["edited_mask"].shape[1], r["edited_mask"].shape[0]),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                borderValue=0)
        return super().mask(image_id)

    def _load_points(self, image_id: str) -> list[dict]:
        if image_id in self._synth_points:
            return self._synth_points[image_id]
        return super()._load_points(image_id)

    def add_edited_synthetic(self, source_image_id: str, frac_removed: float,
                              scale: float, rotation_deg: float,
                              tilt_deg: float) -> str | None:
        """Register one synthetic query = structural edit + viewpoint warp.

        Returns the new image_id, or None if the source has no labelled
        points or the edit leaves no mask.
        """
        src_points = self._load_points(source_image_id)
        if not src_points:
            return None

        src_photo = self.photos[source_image_id]
        img = self.image(source_image_id)
        msk = self.mask(source_image_id)
        h, w = img.shape[:2]

        # --- Structural edit ---
        edited_mask = shorten_crack_mask(msk, frac_removed)
        if not edited_mask.any():
            return None
        edited_image = inpaint_shortened(img, msk, edited_mask)

        # --- Viewpoint warp ---
        H = make_transform(scale, rotation_deg, tilt_deg, w, h)

        tag = (f"e{frac_removed:g}_s{scale:g}"
               f"_r{rotation_deg:g}_t{tilt_deg:g}")
        synth_id = f"{source_image_id}__SYN__{tag}"
        if synth_id in self.photos:
            return synth_id

        self.photos[synth_id] = Photo(
            image_id=synth_id, wall_id=src_photo.wall_id,
            session=f"{src_photo.session}__synth_{tag}",
            img_path="<synthetic>", mask_path="<synthetic>",
            label_path="<synthetic>")

        self._synth_recipe[synth_id] = dict(
            source_image_id=source_image_id,
            frac_removed=frac_removed, H=H,
            edited_image=edited_image, edited_mask=edited_mask)

        self.synth_transform[synth_id] = dict(
            source_image_id=source_image_id,
            frac_removed=frac_removed, scale=scale,
            rotation_deg=rotation_deg, tilt_deg=tilt_deg)

        # Warp the label points so identity resolution works.
        pts = np.float32([p["xy"] for p in src_points]).reshape(-1, 1, 2)
        warped = cv2.transform(pts, H[:2, :]).reshape(-1, 2)
        self._synth_points[synth_id] = [
            {"identity": p["identity"],
             "xy": [int(round(x)), int(round(y))]}
            for p, (x, y) in zip(src_points, warped)]

        # Build instance refs.
        insts = self.instances(synth_id, apply_mask=False)
        points = self._synth_points[synth_id]
        for c_idx, inst in enumerate(insts):
            identity = self._resolve_identity(inst, points)
            ref = InstanceRef(
                instance_id=f"{synth_id}_c{c_idx:02d}",
                image_id=synth_id,
                wall_id=src_photo.wall_id,
                session=self.photos[synth_id].session,
                identity=identity)
            self.refs.append(ref)
            self._ref_to_instance[ref.instance_id] = inst
        return synth_id


# ===========================================================================
# 3. Sweep + evaluate
# ===========================================================================

def build_scorers(names: list[str], data: Dataset) -> dict:
    """Share method parsing with the real benchmark (including @ctx flags)."""
    return {scorer.name: scorer
            for scorer in build_benchmark_scorers(names, data, prune=True)}


def run_edit_sweep(root: str,
                   edit_fracs: list[float],
                   scales: list[float],
                   rotations: list[float],
                   tilts: list[float],
                   methods: list[str],
                   min_sharpness: float | None = 10,
                   max_sources_per_wall: int | None = 4,
                   seed: int = 0) -> list[dict]:
    """Create edited+viewpoint queries and evaluate every method.

    For each (source, edit_frac, scale, rot, tilt) combination the query
    is scored against real gallery photographs of the same wall.  Both
    closed-set (Rank-1, mAP) and open-set (DIR@FAR) metrics are computed
    by reid_eval, so the numbers are in identical units across methods.
    """
    _selftest()

    data = EditedViewpointDataset(root, min_sharpness=min_sharpness)
    scorers = build_scorers(methods, data)

    rng = np.random.default_rng(seed)
    by_wall: dict[str, list[str]] = defaultdict(list)
    for image_id, photo in data.photos.items():
        if data._load_points(image_id):
            by_wall[photo.wall_id].append(image_id)

    sources: list[str] = []
    for wall_id, ids in by_wall.items():
        ids = sorted(ids)
        if max_sources_per_wall and len(ids) > max_sources_per_wall:
            ids = list(rng.choice(ids, size=max_sources_per_wall, replace=False))
        sources.extend(ids)

    bins = list(itertools.product(edit_fracs, scales, rotations, tilts))
    n_evals = len(sources) * len(bins) * len(scorers)
    print(f"{len(sources)} source photos × {len(bins)} edit×viewpoint bins × "
          f"{len(scorers)} methods = {n_evals} evaluations")

    rows: list[dict] = []
    t_total = time.time()

    for si, source_id in enumerate(sources):
        wall_id = data.photos[source_id].wall_id
        real_gallery = [r for r in data.refs
                        if r.wall_id == wall_id
                        and r.image_id not in data._synth_recipe
                        and r.image_id != source_id]
        if not real_gallery:
            continue

        for frac, scale, rot, tilt in bins:
            synth_id = data.add_edited_synthetic(
                source_id, frac, scale, rot, tilt)
            if synth_id is None:
                continue
            query_refs = [r for r in data.refs
                          if r.image_id == synth_id
                          and r.identity is not None]
            if not query_refs:
                continue

            for method, scorer in scorers.items():
                res = evaluate(scorer, query_refs, real_gallery, data,
                               **PROTOCOL)
                rows.append({
                    "method": method,
                    "source_image_id": source_id,
                    "wall_id": wall_id,
                    "frac_removed": frac,
                    "scale": scale,
                    "rotation_deg": rot,
                    "tilt_deg": tilt,
                    "n_queries": res["closed_set"]["n_queries"],
                    "rank1": res["closed_set"]["rank1"],
                    "mAP": res["closed_set"]["mAP"],
                    "dir_at_far10": res["open_set_dir_at_far10"],
                    "pair_f1": res.get("pair_f1_at_threshold", 0.0),
                    "scoreable_pair_rate": res["scoreable_pair_rate"],
                })

        elapsed = time.time() - t_total
        eta = elapsed * (len(sources) - si - 1) / max(si + 1, 1)
        print(f"\r  {si + 1}/{len(sources)} source photos done  "
              f"({elapsed/60:.1f}m elapsed, {eta/60:.1f}m left)   ",
              end="", flush=True)

    print()
    return rows


# ===========================================================================
# 4. Summarise
# ===========================================================================

def summarize(rows: list[dict]) -> str:
    """One row per (method, edit_frac, scale, rot, tilt), averaged over
    source photos."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["method"], r["frac_removed"], r["scale"],
               r["rotation_deg"], r["tilt_deg"])
        groups[key].append(r)

    hdr = (f"{'method':<18}{'edit%':>6}{'scale':>7}{'rot':>7}{'tilt':>7}"
           f"{'nQ':>5}{'R@1':>7}{'mAP':>7}{'DIR.1':>7}{'F1@v':>7}"
           f"{'scored':>8}")
    lines = [hdr, "-" * len(hdr)]
    for key in sorted(groups):
        method, frac, scale, rot, tilt = key
        g = groups[key]
        n = sum(r["n_queries"] for r in g)
        w = np.array([r["n_queries"] for r in g], dtype=float)
        w = w / w.sum() if w.sum() else w
        rank1 = float(np.sum(w * [r["rank1"] for r in g]))
        mAP = float(np.sum(w * [r["mAP"] for r in g]))
        dirf = float(np.mean([r["dir_at_far10"] for r in g]))
        f1v = float(np.mean([r["pair_f1"] for r in g]))
        scored = float(np.mean([r["scoreable_pair_rate"] for r in g]))
        lines.append(
            f"{method:<18}{frac*100:>5.0f}%{scale:>7.2f}{rot:>7.1f}"
            f"{tilt:>7.1f}{n:>5d}{rank1:>7.3f}{mAP:>7.3f}"
            f"{dirf:>7.3f}{f1v:>7.3f}{scored:>8.2f}")
    return "\n".join(lines)


def degradation_table(rows: list[dict]) -> str:
    """Per-method degradation over edit fraction at two named operating
    points: the no-viewpoint bin (pure structural change) and the
    maximum-viewpoint bin (structural change plus the strongest geometric
    deformation in the sweep).  Both must exist in the run or the row is
    skipped.
    """
    no_view = {"scale": "1.00", "rotation_deg": "0.00", "tilt_deg": "0.00"}
    all_view = {k: f"{max(r[k] for r in rows):.2f}" for k in
                ("scale", "rotation_deg", "tilt_deg")}

    def block(rows, label, sel):
        if not rows:
            return [f"  ({label}: no rows)"]
        keep = [r for r in rows if all(
            f"{r[k]:.2f}" == v for k, v in sel.items())]
        if not keep:
            return [f"  ({label}: viewpoint bin not in run)"]
        groups: dict[float, list[dict]] = defaultdict(list)
        for r in keep:
            groups[r["frac_removed"]].append(r)
        lines = [f"  {label}: {sel['scale']}x, {sel['rotation_deg']}°, "
                 f"{sel['tilt_deg']}° tilt",
                 f"  {'edit%':>6}{'nQ':>6}{'R@1':>7}{'mAP':>7}{'DIR@FAR':>9}"
                 f"{'pair F1':>9}",
                 "  " + "-" * 44]
        for frac in sorted(groups):
            g = groups[frac]
            n = sum(r["n_queries"] for r in g)
            w = np.array([r["n_queries"] for r in g], dtype=float)
            w = w / w.sum() if w.sum() else w
            lines.append(
                f"  {frac*100:>5.0f}%{n:>6d}"
                f"{np.sum(w*[r['rank1'] for r in g]):>7.3f}"
                f"{np.sum(w*[r['mAP'] for r in g]):>7.3f}"
                f"{np.mean([r['dir_at_far10'] for r in g]):>9.3f}"
                f"{np.mean([r['pair_f1'] for r in g]):>9.3f}")
        return lines

    lines = []
    for method in sorted({r["method"] for r in rows}):
        mrows = [r for r in rows if r["method"] == method]
        lines.append(f"\n  {method}")
        lines.extend(block(mrows, "no viewpoint (structural edit only)",
                           no_view))
        lines.extend(block(mrows, "max viewpoint (edit + deformation)",
                           all_view))
    return "\n".join(lines)


# ===========================================================================
# 5. CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Structural-edit + genuine-viewpoint discrimination "
                    "evaluation for skeleton-loftr vs osnet@ctx1.")
    ap.add_argument("root", help="Path to dataset/ directory")
    ap.add_argument("--out", default="edit_viewpoint_out")
    ap.add_argument("--edit-fracs", default="0.0,0.25,0.50,0.75",
                    help="Comma-separated fractions of crack to remove")
    ap.add_argument("--scales", default="1.0,1.5",
                    help="Comma-separated isotropic scale factors")
    ap.add_argument("--rotations", default="0,15",
                    help="Comma-separated in-plane rotation degrees")
    ap.add_argument("--tilts", default="0,20",
                    help="Comma-separated out-of-plane tilt degrees")
    ap.add_argument("--methods", nargs="+",
                    default=["skeleton-loftr", "osnet@ctx1"])
    ap.add_argument("--min-sharpness", type=float, default=10)
    ap.add_argument("--max-sources-per-wall", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    edit_fracs = [float(x) for x in args.edit_fracs.split(",")]
    scales = [float(x) for x in args.scales.split(",")]
    rotations = [float(x) for x in args.rotations.split(",")]
    tilts = [float(x) for x in args.tilts.split(",")]

    rows = run_edit_sweep(
        args.root, edit_fracs, scales, rotations, tilts, args.methods,
        min_sharpness=args.min_sharpness,
        max_sources_per_wall=args.max_sources_per_wall,
        seed=args.seed)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "edit_viewpoint_rows.json"), "w") as f:
        json.dump(rows, f, indent=2)

    table = summarize(rows)
    print(table)
    with open(os.path.join(args.out, "edit_viewpoint_table.txt"), "w") as f:
        f.write(table + "\n")

    deg = degradation_table(rows)
    print(deg)
    with open(os.path.join(args.out, "degradation_table.txt"), "w") as f:
        f.write(deg + "\n")

    print(f"\nWrote results to {args.out}/")


if __name__ == "__main__":
    main()
