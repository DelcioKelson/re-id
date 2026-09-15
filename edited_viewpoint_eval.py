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

The GIMP illustrative near-clones (illustrative_synthetic/) join the same
experiment as real queries: each is scored once, without any viewpoint
change, against its wall's real gallery with its disclosed edit fraction,
so the two experiment families share one table and one protocol.

The CROSS-PHOTO family (--cross-photo) answers the natural objection to
the GIMP clones: they are pixel-identical to the source photograph, so an
appearance embedder could score by texture cloning rather than by crack
identity.  For each illustrative wall a DIFFERENT real photograph of the
same wall is chosen (its own mask, its own label points, its own
coordinate frame -- unlike the clones there is no cross-frame alignment
to get wrong) and the parametric shorten+inpaint+warp pipeline runs on
that photograph, scored against the wall's real gallery.  Queries are
genuine cross-viewpoint revisits with the crack shortened by a controlled
fraction.

Usage:
    python edited_viewpoint_eval.py dataset --out edit_viewpoint_out \\
        --edit-fracs 0.0,0.25,0.50,0.75 \\
        --scales 1.0,1.5 --rotations 0,15 --tilts 0,20 \\
        --methods skeleton-loftr osnet@ctx1 \\
        --min-sharpness 10 --max-sources-per-wall 4 \\
        --cross-photo

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

HERE = os.path.dirname(os.path.abspath(__file__))
ILLUSTRATIVE_DIR = os.path.join(HERE, "illustrative_synthetic")


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
        # super().__init__() -> _build_refs() reads EVERY real photo to
        # extract instances, leaving all of them decoded in _img_cache /
        # _mask_cache -- 140 photos at 3072x4080 is >5 GB before the sweep
        # begins, and that is what pushes Colab past ~12 GB. The refs and
        # per-instance crops we keep are small; the full-res pixels are
        # reloaded lazily by image()/mask() when a wall is actually scored,
        # and forget_wall() then drops them again.
        self._img_cache.clear()
        self._mask_cache.clear()

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

    def add_illustrative_synthetic(self, source_image_id: str,
                                   illustrative_path: str,
                                   cutoff_y: int,
                                   frac_removed: float,
                                   scale: float = 1.0,
                                   rotation_deg: float = 0.0,
                                   tilt_deg: float = 0.0) -> str | None:
        """Register one GIMP illustrative near-clone as a query.

        The illustrative images (illustrative_synthetic/) are near-clones
        of a single real source photo: the crack's lower portion was cloned
        over with wall texture through a feathered mask, so the crack is
        intact above the taper end and gone below it.  The mask is
        reconstructed deterministically from the disclosed GIMP recipe
        (source mask, rows at/below the taper end zeroed) and the source
        photo's label points that still lie in the kept region supply the
        identity -- so this is a genuine same-identity query, exactly like
        the parametric ones, but with a realistic hand-made edit.

        By default no viewpoint change is applied (scale=1, rotation_deg=0,
        tilt_deg=0), which reproduces the original near-clone behaviour and
        is suitable for the qualitative baseline.

        Pass non-identity warp parameters to apply an affine transform to
        the GIMP image and mask BEFORE scoring.  This destroys the
        pixel-identical kept region that would otherwise allow appearance
        embedders (e.g. OSNet) to trivially match by texture cloning rather
        than by crack identity.  The warp tag is encoded in the synth_id so
        multiple warp bins of the same GIMP image can coexist in the dataset.

        Label points above the cutoff are warped by the same H, so identity
        resolution remains correct after the geometric change.
        """
        if source_image_id not in self.photos:
            return None
        kept_points = [p for p in self._load_points(source_image_id)
                       if p["xy"][1] < cutoff_y]
        if not kept_points:
            return None

        src_photo = self.photos[source_image_id]
        edited_image = cv2.imread(illustrative_path, cv2.IMREAD_COLOR)
        if edited_image is None:
            return None
        edited_mask = self.mask(source_image_id).copy()
        edited_mask[cutoff_y:, :] = 0
        if not edited_mask.any():
            return None

        # Affine warp -- identity when scale=1, rotation=0, tilt=0, which
        # preserves the original near-clone behaviour.  Non-identity breaks
        # the pixel-level equality between query and gallery so appearance
        # embedders must match across a real geometric change.
        h_img, w_img = edited_image.shape[:2]
        H = make_transform(scale, rotation_deg, tilt_deg, w_img, h_img)

        warp_tag = f"s{scale:g}_r{rotation_deg:g}_t{tilt_deg:g}"
        synth_id = f"{src_photo.wall_id}_sql_{warp_tag}"
        if synth_id in self.photos:
            return synth_id

        self.photos[synth_id] = Photo(
            image_id=synth_id, wall_id=src_photo.wall_id,
            session=f"{src_photo.session}__illustrative_{warp_tag}",
            img_path="<synthetic>", mask_path="<synthetic>",
            label_path="<synthetic>")
        self._synth_recipe[synth_id] = dict(
            source_image_id=source_image_id, frac_removed=frac_removed,
            H=H, edited_image=edited_image, edited_mask=edited_mask)
        self.synth_transform[synth_id] = dict(
            source_image_id=source_image_id, frac_removed=frac_removed,
            scale=scale, rotation_deg=rotation_deg, tilt_deg=tilt_deg)

        # Warp label points by the same H so identity resolution is correct
        # after the geometric change.
        pts = np.float32([p["xy"] for p in kept_points]).reshape(-1, 1, 2)
        warped_pts = cv2.transform(pts, H[:2, :]).reshape(-1, 2)
        self._synth_points[synth_id] = [
            {"identity": p["identity"],
             "xy": [int(round(x)), int(round(y))]}
            for p, (x, y) in zip(kept_points, warped_pts)]

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

    def forget_synthetic(self, synth_id: str) -> None:
        """Drop every cached artefact of one synthetic query.

        The full-resolution edited and warped images would otherwise pile
        up in ``_img_cache`` / ``_synth_recipe`` across thousands of
        source×bin combinations and exhaust Colab's ~12 GB.  Called right
        after the query's evaluations are complete; nothing that happens
        afterwards re-reads the synthetic image.
        """
        self.photos.pop(synth_id, None)
        self._img_cache.pop(synth_id, None)
        self._mask_cache.pop(synth_id, None)
        self._synth_recipe.pop(synth_id, None)   # frees edited_image/edited_mask
        self._synth_points.pop(synth_id, None)
        self.synth_transform.pop(synth_id, None)
        prefix = f"{synth_id}_"
        self._inst_cache = {k: v for k, v in self._inst_cache.items()
                            if k[0] != synth_id}
        self._crop_cache = {k: v for k, v in self._crop_cache.items()
                            if not k.startswith(prefix)}
        self._ref_to_instance = {k: v for k, v in self._ref_to_instance.items()
                                 if not k.startswith(prefix)}
        self.refs = [r for r in self.refs if r.image_id != synth_id]

    def forget_wall(self, wall_id: str) -> None:
        """Free the real images and masks of a wall once it is done.

        ``run_edit_sweep`` visits sources wall-by-wall, so after the last
        source of a wall the whole wall can be evicted from the image
        cache instead of accumulating 140 full-res photos in RAM.
        Column-scope crops used by the scorers live in ``_crop_cache`` /
        ``_ref_to_instance``; they are dropped too so big-crack crops do
        not pile up across walls.
        """
        for pid, photo in self.photos.items():
            if (photo.wall_id == wall_id
                    and pid not in self._synth_recipe):
                self._img_cache.pop(pid, None)
                self._mask_cache.pop(pid, None)
        own = [(k, v) for k, v in self._inst_cache.items()
               if k[0] not in self._synth_recipe and self._photo_wall(k[0]) == wall_id]
        for k, _ in own:
            del self._inst_cache[k]
        prefix = f"{wall_id}_"
        self._crop_cache = {k: v for k, v in self._crop_cache.items()
                            if not k.startswith(prefix)}
        self._ref_to_instance = {k: v for k, v in self._ref_to_instance.items()
                                 if not k.startswith(prefix)}
        self.refs = [r for r in self.refs if r.wall_id != wall_id]

    def _photo_wall(self, image_id: str) -> str | None:
        photo = self.photos.get(image_id)
        return photo.wall_id if photo is not None else None


# ===========================================================================
# 3. Sweep + evaluate
# ===========================================================================

def _evict_scorer_caches(scorer, image_id: str) -> None:
    """Drop scorer-side per-instance caches for one image (real or synth).

    The scorer wrappers cache one prepared object per instance_id
    (LoFTR resized tensors, OSNet embeddings) and never release them. With
    thousands of synthetic bins that cache grows linearly in instance
    count, so it must be pruned together with the dataset caches. The
    wrapper internals are duck-typed here: any dict-valued attribute whose
    keys look like instance ids ('<image>_cNN') of this image is pruned.
    """
    img_prefix = f"{image_id}_"
    for attr in vars(scorer).values():
        if not isinstance(attr, dict):
            continue
        for k in [k for k in attr if isinstance(k, str) and k.startswith(img_prefix)]:
            del attr[k]


def _evict_all_scorer_caches(scorers: dict, image_id: str) -> None:
    for scorer in scorers.values():
        _evict_scorer_caches(scorer, image_id)


def _evict_wall_scorer_caches(scorers: dict, wall_id: str) -> None:
    """Drop scorer cache entries for every (real or synthetic) image whose
    image_id starts with the wall id."""
    wall_prefix = f"{wall_id}_"
    for scorer in scorers.values():
        for attr in vars(scorer).values():
            if not isinstance(attr, dict):
                continue
            for k in [k for k in attr if isinstance(k, str) and k.startswith(wall_prefix)]:
                del attr[k]

def build_scorers(names: list[str], data: Dataset) -> dict:
    """Share method parsing with the real benchmark (including @ctx flags)."""
    return {scorer.name: scorer
            for scorer in build_benchmark_scorers(names, data, prune=True)}


def _pick_secondary_photo(data: Dataset, wall_id: str, exclude_id: str,
                          sources: list[str]) -> str | None:
    """Deterministically pick a different, labelled photo of the wall to
    host the cross-photo supportive query.

    The candidate must be neither the GIMP source (that photo already
    serves the near-clone family) nor one of the sampled sweep sources
    (its shorten+inpaint+warp rows would collide with the main sweep's,
    which share the same synth ids).  Preference: the photo with the most
    click points, so identity resolution has options to land on; ties fall
    back to a deterministic sort.
    """
    cands = [i for i, p in data.photos.items()
             if p.wall_id == wall_id and i != exclude_id
             and i not in sources and data._load_points(i)]
    cands.sort(key=lambda i: (-len(data._load_points(i)), i))
    return cands[0] if cands else None


def _row_key(method: str, source_image_id: str, frac: float, scale: float,
             rot: float, tilt: float, cross_photo: bool,
             illustrative: bool) -> tuple:
    """Identity of one evaluation cell, for checkpoint/resume bookkeeping."""
    return (method, source_image_id, float(frac), float(scale), float(rot),
            float(tilt), bool(cross_photo), bool(illustrative))


def _completed_keys(path: str) -> set[tuple]:
    """Re-read a checkpoint JSONL into the set of finished eval keys."""
    done: set[tuple] = set()
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                done.add(_row_key(r["method"], r["source_image_id"],
                                  r["frac_removed"], r["scale"],
                                  r["rotation_deg"], r["tilt_deg"],
                                  r.get("cross_photo", False),
                                  r.get("illustrative", False)))
    return done


def _bin_done(completed: set[tuple], scorer_names: list[str],
              source_image_id: str, frac: float, scale: float, rot: float,
              tilt: float, cross_photo: bool, illustrative: bool) -> bool:
    """True when every method's eval for one (source, bin, family) is done."""
    return all(_row_key(m, source_image_id, frac, scale, rot, tilt,
                        cross_photo, illustrative) in completed
               for m in scorer_names)


def _capped_gallery_refs(refs: list, limit: int | None) -> list:
    """Deterministic, answer-preserving cap on a wall's gallery.

    Evaluate() scores each synthetic query against the whole wall gallery,
    and the big walls carry hundreds of instances (wall02 has 629) -- so
    skeleton-loftr's pairwise LoFTR grid dominates the runtime.  When the
    gallery exceeds `limit`, keep at least one labelled instance per
    identity (so no query loses its answer and turns artificially
    open-set), then fill the rest deterministically: remaining instances of
    each identity in sorted order, then unlabelled distractors.  Both
    methods see the SAME reduced gallery, so method-vs-method comparisons
    stay valid; only the absolute pool differs from the full gallery.
    """
    if limit is None or limit <= 0 or len(refs) <= limit:
        return sorted(refs, key=lambda r: r.instance_id)
    refs = sorted(refs, key=lambda r: r.instance_id)
    labelled = [r for r in refs if r.identity is not None]
    unlabelled = [r for r in refs if r.identity is None]
    by_identity: dict[str, list] = {}
    for r in labelled:
        by_identity.setdefault(r.identity, []).append(r)
    floor = len(by_identity)
    budget = max(limit, floor)          # never fewer than one answer per identity
    picked = [by_identity[ident][0] for ident in sorted(by_identity)]
    if budget > floor:
        extras = [r for ident in sorted(by_identity)
                  for r in by_identity[ident][1:]]
        picked.extend((extras + unlabelled)[: budget - floor])
    return picked


def _wall_gallery(data: Dataset, wall_id: str, exclude_ids: set[str],
                  limit: int | None) -> list:
    """Real refs of one wall, minus excluded photos, optionally capped."""
    refs = [r for r in data.refs
            if r.wall_id == wall_id
            and r.image_id not in data._synth_recipe
            and r.image_id not in exclude_ids]
    return _capped_gallery_refs(refs, limit)


def run_edit_sweep(root: str,
                   edit_fracs: list[float],
                   scales: list[float],
                   rotations: list[float],
                   tilts: list[float],
                   methods: list[str],
                   min_sharpness: float | None = 10,
                   max_sources_per_wall: int | None = 4,
                   seed: int = 0,
                   illus_scales: list[float] | None = None,
                   illus_rotations: list[float] | None = None,
                   illus_tilts: list[float] | None = None,
                   enable_cross_photo: bool = False,
                   out_dir: str | None = None,
                   resume: bool = False,
                   max_gallery_per_wall: int | None = None,
                   degradation_only: bool = False) -> list[dict]:
    """Create edited+viewpoint queries and evaluate every method.

    For each (source, edit_frac, scale, rot, tilt) combination the query
    is scored against real gallery photographs of the same wall.  Both
    closed-set (Rank-1, mAP) and open-set (DIR@FAR) metrics are computed
    by reid_eval, so the numbers are in identical units across methods.

    illus_scales / illus_rotations / illus_tilts control the affine warp
    applied to each GIMP illustrative image before it is scored.  Defaults
    to [(1.0, 0.0, 0.0)] (identity) so existing callers are unaffected.
    Pass non-identity values to break the pixel-identical kept region that
    would otherwise allow appearance embedders to trivially match by texture
    cloning rather than by crack identity.  Each (scale, rotation, tilt)
    combination produces a separate row tagged with warp_scale,
    warp_rotation_deg, and warp_tilt_deg.

    enable_cross_photo adds the cross-photo supportive family: for each
    illustrative wall, a different real photograph of the same wall (not
    the GIMP source, not one of the sampled sweep sources) hosts the same
    shorten+inpaint+warp pipeline, scored against the wall's real gallery.
    Rows are tagged cross_photo=True and are kept out of the main-sweep
    averages so the near-clone, sweep, and cross-photo families never mix.

    Cost controls: the pairwise matchers (skeleton-loftr) run one LoFTR
    forward per (query instance, gallery instance) cell, and the wall
    galleries are large (wall02 alone has 629 instances), so a full default
    sweep is several hours of GPU.  Three levers:
      * degradation_only   -- run only the two viewpoint bins degradation_table
                              reports (no-viewpoint and max-viewpoint) instead
                              of every point in the product grid (~4x fewer).
                              Other bins still appear in summarize(); the two
                              operating-point tables become fully populated.
      * max_gallery_per_wall -- deterministic, answer-preserving cap on each
                              wall's gallery (see _capped_gallery_refs).  Both
                              methods see the identical reduced pool.
      * out_dir + resume  -- rows are appended to
                              <out_dir>/edit_viewpoint_rows.jsonl as the sweep
                              runs; --resume reloads it and skips finished
                              cells, so a killed run loses at most the current
                              cell.  Resume requires the same bins, gallery
                              cap and families as the interrupted run.
    """
    _selftest()

    # Default: one identity warp bin (original near-clone behaviour).
    if illus_scales is None:
        illus_scales = [1.0]
    if illus_rotations is None:
        illus_rotations = [0.0]
    if illus_tilts is None:
        illus_tilts = [0.0]
    illus_warp_bins = list(itertools.product(illus_scales, illus_rotations, illus_tilts))

    data = EditedViewpointDataset(root, min_sharpness=min_sharpness)
    scorers = build_scorers(methods, data)
    scorer_names = sorted(scorers)

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
    if degradation_only:
        no_view = (1.0, 0.0, 0.0)
        max_view = (max(scales), max(rotations), max(tilts))
        bins = ([(f, *no_view) for f in edit_fracs]
                + [(f, *max_view) for f in edit_fracs])
        print(f"degradation-only: restricting {len(list(itertools.product(edit_fracs, scales, rotations, tilts)))} "
              f"bins to the 2 operating points degradation_table reports "
              f"({no_view} and {max_view}) -> {len(bins)} bins")

    n_evals = len(sources) * len(bins) * len(scorers)
    print(f"{len(sources)} source photos × {len(bins)} edit×viewpoint bins × "
          f"{len(scorers)} methods = {n_evals} evaluations")
    print(f"  + {17} GIMP illustrative images × {len(illus_warp_bins)} warp bins "
          f"× {len(scorers)} methods")

    # Checkpoint JSONL: every emitted row is appended and flushed, so a
    # killed run keeps everything before the current cell.  On --resume the
    # same file is replayed into `rows` and the completed-key set.
    ckpt_path = os.path.join(out_dir, "edit_viewpoint_rows.jsonl") if out_dir else None
    if ckpt_path:
        os.makedirs(out_dir, exist_ok=True)
    completed: set[tuple] = set()
    rows: list[dict] = []
    if resume and ckpt_path and os.path.isfile(ckpt_path):
        with open(ckpt_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        completed = _completed_keys(ckpt_path)
        print(f"resume: reloaded {len(rows)} checkpointed rows, "
              f"{len(completed)} completed eval cells to skip")
    ckpt = open(ckpt_path, "a") if ckpt_path else None

    def emit(r: dict) -> None:
        rows.append(r)
        if ckpt is not None:
            ckpt.write(json.dumps(r) + "\n")
            ckpt.flush()

    t_total = time.time()
    current_wall = None

    # GIMP illustrative near-clones (one per wall).
    # Each is scored once per (warp_scale, warp_rotation, warp_tilt) bin.
    # At the identity warp this is the original near-clone behaviour.
    # At non-identity warps the GIMP image is affine-transformed before
    # scoring, breaking the pixel-identical texture that OSNet exploits.
    illustrative: dict[str, dict] = {}
    manif = os.path.join(ILLUSTRATIVE_DIR, "manifest.json")
    if os.path.isfile(manif):
        with open(manif) as f:
            for name, recipe in json.load(f).items():
                src_path = os.path.join(HERE, recipe["source"])
                src_id = os.path.splitext(os.path.basename(src_path))[0]
                if src_id in data.photos:
                    illustrative[data.photos[src_id].wall_id] = (name, recipe)

    for si, source_id in enumerate(sources):
        wall_id = data.photos[source_id].wall_id
        if wall_id != current_wall:
            # Sources are processed wall-by-wall (see by_wall construction),
            # so the previous wall's real photos will not be needed again.
            if current_wall is not None:
                data.forget_wall(current_wall)
                _evict_wall_scorer_caches(scorers, current_wall)
            current_wall = wall_id

        if wall_id in illustrative:
            name, recipe = illustrative.pop(wall_id)
            src_path = os.path.join(HERE, recipe["source"])
            src_id = os.path.splitext(os.path.basename(src_path))[0]
            frac = 1.0 - float(recipe["split_frac_kept"])
            cutoff = int(recipe["taper_zone_y"][1])
            illus_gallery = _wall_gallery(data, wall_id, {src_id},
                                          max_gallery_per_wall)

            for w_scale, w_rot, w_tilt in illus_warp_bins:
                if _bin_done(completed, scorer_names, src_id, frac,
                             w_scale, w_rot, w_tilt, False, True):
                    continue
                query = data.add_illustrative_synthetic(
                    src_id,
                    os.path.join(ILLUSTRATIVE_DIR, name),
                    cutoff, frac,
                    scale=w_scale, rotation_deg=w_rot, tilt_deg=w_tilt)
                query_refs = [r for r in data.refs
                              if query is not None and r.image_id == query
                              and r.identity is not None]
                if query is not None and query_refs and illus_gallery:
                    for method, scorer in scorers.items():
                        if _row_key(method, src_id, frac, w_scale, w_rot,
                                    w_tilt, False, True) in completed:
                            continue
                        res = evaluate(scorer, query_refs, illus_gallery,
                                       data, **PROTOCOL)
                        emit({
                            "method": method,
                            "source_image_id": src_id,
                            "wall_id": wall_id,
                            "frac_removed": frac,
                            "scale": w_scale,
                            "rotation_deg": w_rot,
                            "tilt_deg": w_tilt,
                            "illustrative": True,
                            "cross_photo": False,
                            "warp_scale": w_scale,
                            "warp_rotation_deg": w_rot,
                            "warp_tilt_deg": w_tilt,
                            "n_queries": res["closed_set"]["n_queries"],
                            "rank1": res["closed_set"]["rank1"],
                            "mAP": res["closed_set"]["mAP"],
                            "dir_at_far10": res["open_set_dir_at_far10"],
                            "pair_f1": res.get("pair_f1_at_threshold", 0.0),
                            "scoreable_pair_rate": res["scoreable_pair_rate"],
                        })
                if query is not None:
                    data.forget_synthetic(query)
                    _evict_all_scorer_caches(scorers, query)

            # Cross-photo supportive family: the shorten+inpaint+warp
            # pipeline rerun on a DIFFERENT real photograph of the same
            # wall.  Unlike the GIMP clone this query is a genuine
            # cross-viewpoint revisit -- no pixel-identical kept region, no
            # cross-frame coordinate coupling (the secondary photo's own
            # mask / labels / frame are used throughout).
            if enable_cross_photo:
                secondary = _pick_secondary_photo(data, wall_id, src_id,
                                                  sources)
                if secondary is not None:
                    x_gallery = _wall_gallery(data, wall_id, {secondary},
                                              max_gallery_per_wall)
                    if x_gallery:
                        for frac, scale, rot, tilt in bins:
                            if _bin_done(completed, scorer_names, secondary,
                                         frac, scale, rot, tilt, True, False):
                                continue
                            qid = data.add_edited_synthetic(
                                secondary, frac, scale, rot, tilt)
                            if qid is None:
                                continue
                            qrefs = [r for r in data.refs
                                     if r.image_id == qid
                                     and r.identity is not None]
                            if not qrefs:
                                continue
                            for method, scorer in scorers.items():
                                if _row_key(method, secondary, frac, scale,
                                            rot, tilt, True, False) in completed:
                                    continue
                                res = evaluate(scorer, qrefs, x_gallery, data,
                                               **PROTOCOL)
                                emit({
                                    "method": method,
                                    "source_image_id": secondary,
                                    "wall_id": wall_id,
                                    "frac_removed": frac,
                                    "scale": scale,
                                    "rotation_deg": rot,
                                    "tilt_deg": tilt,
                                    "illustrative": False,
                                    "cross_photo": True,
                                    "n_queries": res["closed_set"]["n_queries"],
                                    "rank1": res["closed_set"]["rank1"],
                                    "mAP": res["closed_set"]["mAP"],
                                    "dir_at_far10": res["open_set_dir_at_far10"],
                                    "pair_f1": res.get("pair_f1_at_threshold", 0.0),
                                    "scoreable_pair_rate": res["scoreable_pair_rate"],
                                })
                            data.forget_synthetic(qid)
                            _evict_all_scorer_caches(scorers, qid)


        real_gallery = _wall_gallery(data, wall_id, {source_id},
                                     max_gallery_per_wall)
        if not real_gallery:
            continue

        for frac, scale, rot, tilt in bins:
            if _bin_done(completed, scorer_names, source_id,
                         frac, scale, rot, tilt, False, False):
                continue
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
                if _row_key(method, source_id, frac, scale, rot, tilt,
                            False, False) in completed:
                    continue
                res = evaluate(scorer, query_refs, real_gallery, data,
                               **PROTOCOL)
                emit({
                    "method": method,
                    "source_image_id": source_id,
                    "wall_id": wall_id,
                    "frac_removed": frac,
                    "scale": scale,
                    "rotation_deg": rot,
                    "tilt_deg": tilt,
                    "illustrative": False,
                    "cross_photo": False,
                    "n_queries": res["closed_set"]["n_queries"],
                    "rank1": res["closed_set"]["rank1"],
                    "mAP": res["closed_set"]["mAP"],
                    "dir_at_far10": res["open_set_dir_at_far10"],
                    "pair_f1": res.get("pair_f1_at_threshold", 0.0),
                    "scoreable_pair_rate": res["scoreable_pair_rate"],
                })

            # The synthetic query's edited+warped full-res artefacts have
            # now been consumed (metrics are summary numbers); free them so
            # thousands of bins do not accumulate in ~12 GB of Colab RAM.
            data.forget_synthetic(synth_id)
            _evict_all_scorer_caches(scorers, synth_id)

        elapsed = time.time() - t_total
        eta = elapsed * (len(sources) - si - 1) / max(si + 1, 1)
        print(f"\r  {si + 1}/{len(sources)} source photos done  "
              f"({elapsed/60:.1f}m elapsed, {eta/60:.1f}m left)   ",
              end="", flush=True)

    if current_wall is not None:
        data.forget_wall(current_wall)
        _evict_wall_scorer_caches(scorers, current_wall)

    print()
    if ckpt is not None:
        ckpt.close()
    return rows


# ===========================================================================
# 4. Summarise
# ===========================================================================

def summarize(rows: list[dict]) -> str:
    """One row per (method, family, edit_frac, scale, rot, tilt), averaged
    over source photos.  The cross_photo family (a different real photo of
    the same wall hosting the edit) is kept in its own rows, marked 'x', so
    it never dilutes the main-sweep averages."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["method"], r.get("cross_photo", False),
               r["frac_removed"], r["scale"],
               r["rotation_deg"], r["tilt_deg"])
        groups[key].append(r)

    hdr = (f"{'method':<20}{'edit%':>6}{'scale':>7}{'rot':>7}{'tilt':>7}"
           f"{'nQ':>5}{'R@1':>7}{'mAP':>7}{'DIR.1':>7}{'F1@v':>7}"
           f"{'scored':>8}")
    lines = [hdr, "-" * len(hdr)]
    for key in sorted(groups):
        method, cross, frac, scale, rot, tilt = key
        g = groups[key]
        n = sum(r["n_queries"] for r in g)
        w = np.array([r["n_queries"] for r in g], dtype=float)
        w = w / w.sum() if w.sum() else w
        rank1 = float(np.sum(w * [r["rank1"] for r in g]))
        mAP = float(np.sum(w * [r["mAP"] for r in g]))
        dirf = float(np.mean([r["dir_at_far10"] for r in g]))
        f1v = float(np.mean([r["pair_f1"] for r in g]))
        scored = float(np.mean([r["scoreable_pair_rate"] for r in g]))
        label = f"{method}{' *x' if cross else ''}"
        lines.append(
            f"{label:<20}{frac*100:>5.0f}%{scale:>7.2f}{rot:>7.1f}"
            f"{tilt:>7.1f}{n:>5d}{rank1:>7.3f}{mAP:>7.3f}"
            f"{dirf:>7.3f}{f1v:>7.3f}{scored:>8.2f}")
    return "\n".join(lines)


def degradation_table(rows: list[dict]) -> str:
    """Per-method degradation over edit fraction at two named operating
    points: the no-viewpoint bin (pure structural change) and the
    maximum-viewpoint bin (structural change plus the strongest geometric
    deformation in the sweep).  Both must exist in the run or the row is
    skipped.  The cross_photo family is tabulated separately (marked 'x')."""
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
    for method, cross in sorted({(r["method"], r.get("cross_photo", False))
                                 for r in rows}):
        mrows = [r for r in rows
                 if r["method"] == method and r.get("cross_photo", False) == cross]
        heading = f"{method}{'  [cross-photo]' if cross else ''}"
        lines.append(f"\n  {heading}")
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
    ap.add_argument("--illus-scales", default="1.0",
                    help="Comma-separated scale factors for GIMP illustrative warps "
                         "(default 1.0 = no warp). Use e.g. '1.0,1.1,1.2' to add "
                         "geometric perturbation that breaks pixel identity.")
    ap.add_argument("--illus-rotations", default="0",
                    help="Comma-separated in-plane rotations (degrees) for GIMP "
                         "illustrative warps (default 0). Use e.g. '0,5,10'.")
    ap.add_argument("--illus-tilts", default="0",
                    help="Comma-separated out-of-plane tilts (degrees) for GIMP "
                         "illustrative warps (default 0). Use e.g. '0,10'.")
    ap.add_argument("--methods", nargs="+",
                    default=["skeleton-loftr", "osnet@ctx1"])
    ap.add_argument("--min-sharpness", type=float, default=10)
    ap.add_argument("--max-sources-per-wall", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cross-photo", action="store_true",
                    help="For each illustrative wall, also run the "
                         "shorten+inpaint+warp pipeline on a DIFFERENT real "
                         "photo of the same wall (its own mask/labels) and "
                         "score it against the wall's real gallery. Rows are "
                         "tagged cross_photo=True.")
    ap.add_argument("--degradation-only", action="store_true",
                    help="Run only the 2 viewpoint bins degradation_table "
                         "reports (no-viewpoint + max-viewpoint) instead of "
                         "the whole product grid (~4x fewer evals).")
    ap.add_argument("--max-gallery-per-wall", type=int, default=0,
                    help="Cap each wall's gallery to this many instances "
                         "(deterministic, keeps >=1 answer per identity). "
                         "Tames the big walls: wall02 alone has 629 "
                         "instances. 0 = no cap.")
    ap.add_argument("--resume", action="store_true",
                    help="Re-read <out>/edit_viewpoint_rows.jsonl and skip "
                         "already-finished (source x bin x method) cells. "
                         "Requires the same bins, gallery cap and families "
                         "as the interrupted run. The JSONL is written "
                         "incrementally, so killed runs keep completed rows.")
    args = ap.parse_args()

    edit_fracs = [float(x) for x in args.edit_fracs.split(",")]
    scales = [float(x) for x in args.scales.split(",")]
    rotations = [float(x) for x in args.rotations.split(",")]
    tilts = [float(x) for x in args.tilts.split(",")]
    illus_scales = [float(x) for x in args.illus_scales.split(",")]
    illus_rotations = [float(x) for x in args.illus_rotations.split(",")]
    illus_tilts = [float(x) for x in args.illus_tilts.split(",")]

    rows = run_edit_sweep(
        args.root, edit_fracs, scales, rotations, tilts, args.methods,
        min_sharpness=args.min_sharpness,
        max_sources_per_wall=args.max_sources_per_wall,
        seed=args.seed,
        illus_scales=illus_scales,
        illus_rotations=illus_rotations,
        illus_tilts=illus_tilts,
        enable_cross_photo=args.cross_photo,
        out_dir=args.out,
        resume=args.resume,
        max_gallery_per_wall=(args.max_gallery_per_wall
                              if args.max_gallery_per_wall > 0 else None),
        degradation_only=args.degradation_only)

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
