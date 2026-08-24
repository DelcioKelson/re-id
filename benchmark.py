"""
Turn a folder of real inspection photos into a runnable re-ID benchmark.

This is the bridge between three pieces that already exist:
  * crack_reid_baselines.py  - the seven baseline matchers + embedders
  * crack_registration_reid.py - the proposed registration method
  * reid_eval.py             - the shared CMC / mAP / open-set protocol

It supplies the two things those files assume but don't provide for real
data: a concrete on-disk DATASET LAYOUT, and a LABELLING scheme that
survives re-segmentation, so you can start labelling the day you shoot
and not have the labels break when you retune the segmenter.

--------------------------------------------------------------------------
DATASET LAYOUT
--------------------------------------------------------------------------
    dataset/
      walls.csv                       # image_id, wall_id, session, path
      images/
        wall03_s1_0007.jpg
        wall03_s2_0011.jpg
        ...
      masks/                          # produced by your UNet++ (same stem)
        wall03_s1_0007.png
        ...
      labels/
        wall03_s1_0007.json           # click points, see below
        ...
      splits.json                     # {"val": ["wall01","wall02"], "test": [...]}

walls.csv (one row per photo):
    image_id,wall_id,session,path
    wall03_s1_0007,wall03,2026-08-10-am,images/wall03_s1_0007.jpg
    wall03_s2_0011,wall03,2026-09-14-am,images/wall03_s2_0011.jpg

--------------------------------------------------------------------------
LABELLING (click points, not per-instance IDs)
--------------------------------------------------------------------------
For each photo you drop one point on each distinct physical crack and
give it an identity string that is CONSISTENT ACROSS SESSIONS for the
same wall. A crack photographed in session 1 and again in session 2 gets
the SAME identity string in both label files.

    labels/wall03_s1_0007.json
    {
      "image_id": "wall03_s1_0007",
      "points": [
        {"identity": "wall03_crack01", "xy": [412, 690]},
        {"identity": "wall03_crack02", "xy": [980, 240]}
      ]
    }

Why points and not instance indices: connected-component indices depend
on min_area / close_px / the segmentation threshold. If you label by
index and later retune any of those, every label silently rots. A click
inside the crack is resolved to whatever component currently contains it,
so labels are stable under re-segmentation. Multiple components of one
physical crack all resolve to the same identity because the click lands
in one of them and the rest inherit via mask connectivity at label time
-- or, if they're genuinely separate components, you drop one point per
component with the same identity string. mAP treats them all as relevant.

    pip install opencv-contrib-python numpy scipy
    (torch / kornia / torchreid / transformers / ultralytics as needed)
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from crack_reid_baselines import extract_crack_instances, CrackInstance
from reid_eval import (
    InstanceRef, EmbeddingScorer, RegistrationScorer, DistanceScorer,
    evaluate, calibrate_threshold, results_table,
)


# ===========================================================================
# 1. Load the dataset manifest
# ===========================================================================

@dataclass
class Photo:
    image_id: str
    wall_id: str
    session: str
    img_path: str
    mask_path: str
    label_path: str


def load_manifest(root: str) -> list[Photo]:
    photos = []
    with open(os.path.join(root, "walls.csv")) as f:
        for row in csv.DictReader(f):
            image_id = row["image_id"]
            photos.append(Photo(
                image_id=image_id,
                wall_id=row["wall_id"],
                session=row["session"],
                img_path=os.path.join(root, row["path"]),
                mask_path=os.path.join(root, "masks", f"{image_id}.png"),
                label_path=os.path.join(root, "labels", f"{image_id}.json"),
            ))
    return photos


# ===========================================================================
# 2. Resolve instances + attach identities from click points
# ===========================================================================

class Dataset:
    """Holds every extracted instance and lazily serves crops / images.

    The `data` object passed to reid_eval scorers: it must expose
    .crop(ref), .image(image_id), .mask(image_id) and the instance lists.
    Instances are extracted ONCE per (image, apply_mask) setting and
    cached, because keypoint and embedding families disagree on masking.
    """

    def __init__(self, root: str, min_area: int = 200, close_px: int = 5,
                 point_tolerance: int = 25):
        self.root = root
        self.min_area = min_area
        self.close_px = close_px
        self.point_tolerance = point_tolerance
        self.photos = {p.image_id: p for p in load_manifest(root)}

        self._img_cache: dict[str, np.ndarray] = {}
        self._mask_cache: dict[str, np.ndarray] = {}
        self._inst_cache: dict[tuple[str, bool], list[CrackInstance]] = {}
        self._crop_cache: dict[str, np.ndarray] = {}

        self.refs: list[InstanceRef] = []
        self._ref_to_instance: dict[str, CrackInstance] = {}
        self._build_refs()

    # ---- raw IO ----
    def image(self, image_id: str) -> np.ndarray:
        if image_id not in self._img_cache:
            img = cv2.imread(self.photos[image_id].img_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(self.photos[image_id].img_path)
            self._img_cache[image_id] = img
        return self._img_cache[image_id]

    def mask(self, image_id: str) -> np.ndarray:
        if image_id not in self._mask_cache:
            m = cv2.imread(self.photos[image_id].mask_path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise FileNotFoundError(self.photos[image_id].mask_path)
            img = self.image(image_id)
            if m.shape[:2] != img.shape[:2]:
                m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            self._mask_cache[image_id] = m
        return self._mask_cache[image_id]

    def instances(self, image_id: str, apply_mask: bool) -> list[CrackInstance]:
        key = (image_id, apply_mask)
        if key not in self._inst_cache:
            self._inst_cache[key] = extract_crack_instances(
                self.image(image_id), self.mask(image_id),
                min_area=self.min_area, close_px=self.close_px, apply_mask=apply_mask,
            )
        return self._inst_cache[key]

    # ---- identity resolution ----
    def _load_points(self, image_id: str) -> list[dict]:
        path = self.photos[image_id].label_path
        if not os.path.exists(path):
            return []
        with open(path) as f:
            return json.load(f).get("points", [])

    def _resolve_identity(self, inst: CrackInstance, points: list[dict]) -> str | None:
        """Return the identity of the click point lying in (or nearest to)
        this instance's mask, within tolerance. None if unlabelled."""
        x0, y0, w, h = inst.bbox
        best_id, best_d = None, self.point_tolerance + 1
        for p in points:
            px, py = p["xy"]
            lx, ly = px - x0, py - y0
            if 0 <= lx < inst.mask_crop.shape[1] and 0 <= ly < inst.mask_crop.shape[0] \
                    and inst.mask_crop[ly, lx] > 0:
                return p["identity"]              # point falls inside the crack: exact
            # else fall back to distance from bbox centre, for near-misses
            cx, cy = x0 + w / 2, y0 + h / 2
            d = float(np.hypot(px - cx, py - cy))
            if d < best_d:
                best_id, best_d = p["identity"], d
        return best_id if best_d <= self.point_tolerance else None

    def _build_refs(self):
        """Create one InstanceRef per connected component, with identity
        resolved from click points. Uses apply_mask=False geometry as the
        canonical instance set; the two mask settings share bboxes, so a
        ref maps to the right component under either."""
        for image_id, photo in self.photos.items():
            points = self._load_points(image_id)
            insts = self.instances(image_id, apply_mask=False)
            for c_idx, inst in enumerate(insts):
                identity = self._resolve_identity(inst, points)
                ref = InstanceRef(
                    instance_id=f"{image_id}_c{c_idx:02d}",
                    image_id=image_id,
                    wall_id=photo.wall_id,
                    session=photo.session,
                    identity=identity,
                )
                self.refs.append(ref)
                self._ref_to_instance[ref.instance_id] = inst

    # ---- crop service for scorers ----
    def crop(self, ref: InstanceRef, apply_mask: bool = False,
             context: float = 0.0, invert: bool = False) -> np.ndarray:
        """The pixels a crop-scope method gets to see.

        context: expand the instance bbox by this fraction of its own width
            and height on every side before cropping. 0.0 is the tight crop
            (reid_eval's "crop" scope); >0 is the "crop+ctx" scope, which
            reid_eval declares but nothing previously implemented. Without
            it every crop method competes on an isolated patch against a
            registration method that sees the entire photo, so the
            comparison measures field of view as much as it measures method.
        """
        key = f"{ref.instance_id}_{apply_mask}_{context:g}_{invert}"
        if key not in self._crop_cache:
            crop, mask, _ = self._window(ref, context)
            if invert:
                crop = _erase_crack(crop, mask)
            elif apply_mask:
                crop = cv2.bitwise_and(crop, crop, mask=mask)
            self._crop_cache[key] = crop
        return self._crop_cache[key]

    def _window(self, ref: InstanceRef, context: float):
        """(crop, mask, bbox) for this instance at the requested context."""
        inst = self._ref_to_instance[ref.instance_id]
        if context <= 0:
            return inst.crop, inst.mask_crop, inst.bbox
        img = self.image(ref.image_id)
        h_img, w_img = img.shape[:2]
        x, y, w, h = inst.bbox
        mx, my = int(round(w * context)), int(round(h * context))
        x0, y0 = max(x - mx, 0), max(y - my, 0)
        x1, y1 = min(x + w + mx, w_img), min(y + h + my, h_img)
        crop = img[y0:y1, x0:x1].copy()
        # Re-place the component mask inside the widened window, so
        # apply_mask still isolates the same physical crack.
        full = np.zeros((h_img, w_img), np.uint8)
        full[y:y + inst.mask_crop.shape[0], x:x + inst.mask_crop.shape[1]] = inst.mask_crop
        return crop, full[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)

    def instance_with_context(self, ref: InstanceRef, context: float = 0.0,
                              invert: bool = False) -> CrackInstance:
        """CrackInstance at the requested context.

        The keypoint and learned-pairwise matchers consume CrackInstance
        objects rather than bare crops (they do their own masking from
        mask_crop), so they need this rather than crop().
        """
        if context <= 0 and not invert:
            return self._ref_to_instance[ref.instance_id]
        crop, mask, bbox = self._window(ref, context)
        if invert:
            crop = _erase_crack(crop, mask)
        return CrackInstance(crop=crop, mask_crop=mask, bbox=bbox)

    def instance_of(self, ref: InstanceRef) -> CrackInstance:
        return self._ref_to_instance[ref.instance_id]

    # ---- query / gallery construction ----
    def query_gallery(self, wall_ids: set[str] | None = None):
        """Every labelled instance is a query; gallery is every instance
        from a different session. reid_eval's validity mask enforces the
        same-wall / different-session protocol, so here we just return the
        pools and let it restrict."""
        refs = [r for r in self.refs if wall_ids is None or r.wall_id in wall_ids]
        queries = [r for r in refs if r.identity is not None]
        gallery = refs                                    # includes distractors
        return queries, gallery

    def labelled_summary(self) -> str:
        by_wall = defaultdict(lambda: defaultdict(set))
        n_labelled = n_total = 0
        for r in self.refs:
            n_total += 1
            if r.identity is not None:
                n_labelled += 1
                by_wall[r.wall_id][r.identity].add(r.session)
        lines = [f"{n_total} instances, {n_labelled} labelled, "
                 f"{n_total - n_labelled} distractors/unlabelled"]
        matchable = 0
        for wall, ids in sorted(by_wall.items()):
            multi = sum(1 for sessions in ids.values() if len(sessions) >= 2)
            matchable += multi
            lines.append(f"  {wall}: {len(ids)} identities, "
                         f"{multi} seen in >=2 sessions (matchable)")
        lines.append(f"TOTAL matchable identities (usable as queries with an answer): {matchable}")
        return "\n".join(lines)


# The evaluation protocol, defined once. run() passes it to reid_eval, and
# the expensive scorers below use it to skip cells the metrics discard.
PROTOCOL = dict(same_wall_only=True, exclude_same_session=True, min_frame_gap=0)


def valid_mask(queries, gallery) -> np.ndarray:
    """Which (query, gallery) cells the metrics will actually read.

    reid_eval masks every invalid cell out of CMC, mAP, the open-set curve
    and the pair PR curve, so scoring them is pure waste -- and for the
    genuinely pairwise matchers (SuperGlue/LoFTR) that waste is a network
    forward pass each. Delegating to reid_eval's own function means the
    two can never disagree about what "valid" means."""
    from reid_eval import build_validity_mask
    return build_validity_mask(queries, gallery, **PROTOCOL)


def _missing(e: ImportError) -> str:
    """Describe a failed import.

    e.name is None whenever the ImportError was raised by hand rather than
    by the import machinery -- which is exactly the case for the SuperGlue
    checkout, whose message carries the clone command. Printing e.name
    unconditionally turned that into "needs 'None'" and threw the useful
    part away."""
    return f"needs '{e.name}'" if e.name else str(e)


def _progress(name: str, done: int, todo: int, t0: float):
    """One rewritten line. A full pairwise run is tens of thousands of
    forward passes; without this there is no way to tell a slow method
    from a hung one."""
    if done == 0:
        return
    el = time.time() - t0
    eta = el * (todo - done) / done
    print(f"\r    {name}: {done}/{todo} pairs  {el/60:.1f}m elapsed, "
          f"{eta/60:.1f}m left   ", end="", flush=True)
    if done >= todo:
        print()


# ===========================================================================
# 3. Adapters that connect the baseline matchers to reid_eval's interface
# ===========================================================================
# reid_eval already ships EmbeddingScorer and RegistrationScorer. The
# keypoint / learned-pairwise baselines need a thin scorer that produces a
# (n_query, n_gallery) score matrix by caching descriptors once per crop.

class PairwiseMatcherScorer(DistanceScorer):
    """Wraps a crack_reid_baselines matcher (SIFT/ORB/SuperGlue/LoFTR).

    Descriptors are cached per instance in prepare(); score_matrix reuses
    them, so the whole (Q x G) grid costs one detect per crop plus Q*G
    cheap comparisons (or Q*G forward passes for the genuinely-pairwise
    SuperGlue/LoFTR, which is their real, reportable cost)."""

    def __init__(self, matcher, prune: bool = True, context: float = 0.0,
                 apply_mask: bool | None = None, name: str | None = None,
                 invert: bool = False):
        self.matcher = matcher
        self.context = context
        self.invert = invert
        # None = keep the matcher's own default. Overriding matters because
        # every keypoint matcher defaults to apply_mask=True, which blacks
        # out everything but the crack -- and a thin crack silhouette has
        # almost no keypoints to find. That default was asserted in a
        # comment, never measured.
        self.apply_mask = self.matcher.apply_mask if apply_mask is None else apply_mask
        self.name = name or matcher.name
        self.input_scope = "crop+ctx" if context > 0 else "crop"
        self.prune = prune
        self._prep: dict[str, object] = {}

    def prepare(self, refs, data: Dataset):
        todo = [r for r in refs if r.instance_id not in self._prep]
        if not todo:
            return
        insts = [data.instance_with_context(r, self.context, self.invert) for r in todo]
        # matcher.prepare works on CrackInstance lists and honours apply_mask
        preps = self.matcher.prepare(
            [_maybe_mask(inst, self.apply_mask) for inst in insts]
        )
        for r, p in zip(todo, preps):
            self._prep[r.instance_id] = p

    def distance_matrix(self, queries, gallery, data: Dataset):
        # DistanceScorer negates, so return a distance: use -score.
        q = [self._prep[r.instance_id] for r in queries]
        g = [self._prep[r.instance_id] for r in gallery]
        valid = valid_mask(queries, gallery) if self.prune else None
        D = np.full((len(q), len(g)), np.inf, dtype=float)
        todo = int(valid.sum()) if valid is not None else len(q) * len(g)
        done = 0
        t0 = time.time()
        for i, pa in enumerate(q):
            cols = np.nonzero(valid[i])[0] if valid is not None else range(len(g))
            for j in cols:
                D[i, j] = -self.matcher.score_pair(pa, g[j])
                done += 1
            if todo and (i % 25 == 0 or i == len(q) - 1):
                _progress(self.name, done, todo, t0)
        return D


def _erase_crack(crop: np.ndarray, mask: np.ndarray, dilate_px: int = 5) -> np.ndarray:
    """Remove the crack from a crop, leaving plausible wall behind.

    The control this serves asks "is the method matching the crack, or the
    wall patch it sits in?". Zeroing the crack pixels does NOT answer that:
    it leaves a crack-SHAPED black region, so the morphology survives
    perfectly as negative space and the control measures nothing. Inpainting
    fills the region from surrounding wall texture, so the silhouette is
    genuinely gone. The mask is dilated first so the crack's darker edge
    pixels go too.
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    m = cv2.dilate((mask > 0).astype(np.uint8), k)
    if m.sum() == 0:
        return crop
    return cv2.inpaint(crop, m, 3, cv2.INPAINT_TELEA)


def _maybe_mask(inst: CrackInstance, apply: bool) -> CrackInstance:
    if not apply:
        return inst
    masked = cv2.bitwise_and(inst.crop, inst.crop, mask=inst.mask_crop)
    return CrackInstance(crop=masked, mask_crop=inst.mask_crop, bbox=inst.bbox)


class ShapeEmbeddingScorer(EmbeddingScorer):
    """Embeds the crack MASK, not the photo pixels.

    crack_shape_descriptor works on morphology, so it needs mask_crop
    rather than the BGR crop every other embedder consumes. Everything
    downstream (cosine similarity, caching, the metrics) is unchanged.
    """

    def __init__(self, name: str, embed_fn, context: float = 0.0):
        super().__init__(name=name, embed_fn=embed_fn, input_scope="crop")
        self.context = context

    def prepare(self, refs, data):
        todo = [r for r in refs if r.instance_id not in self._cache]
        if not todo:
            return
        masks = [data.instance_with_context(r, self.context).mask_crop for r in todo]
        for r, e in zip(todo, self.embed_fn(masks)):
            self._cache[r.instance_id] = e


class ContextEmbeddingScorer(EmbeddingScorer):
    """EmbeddingScorer that feeds the backbone a context-expanded crop.

    Identical to its parent except for the field of view, which is the
    whole point: holding the backbone fixed and varying only `context`
    isolates how much of a method's score comes from seeing surrounding
    wall rather than from the representation itself.
    """

    def __init__(self, name: str, embed_fn, context: float, input_scope: str,
                 invert: bool = False):
        super().__init__(name=name, embed_fn=embed_fn, input_scope=input_scope)
        self.context = context
        self.invert = invert

    def prepare(self, refs, data):
        todo = [r for r in refs if r.instance_id not in self._cache]
        if not todo:
            return
        embs = self.embed_fn([data.crop(r, context=self.context, invert=self.invert)
                              for r in todo])
        for r, e in zip(todo, embs):
            self._cache[r.instance_id] = e


def parse_method_key(key: str) -> tuple[str, float, bool | None, bool]:
    """Parse 'name[@flag]...' into (name, context, apply_mask, invert).

        clip                    -> ('clip', 0.0, None,  False)
        clip@ctx1.5             -> ('clip', 1.5, None,  False)
        sift@nomask             -> ('sift', 0.0, False, False)
        sift@ctx1@nomask        -> ('sift', 1.0, False, False)
        clip@ctx2@nocrack       -> ('clip', 2.0, None,  True)

    @ctxN    expand the bbox by N times its size on each side (crop+ctx
             scope). '@ctx' alone means 1.0.
    @nomask  let the method see the wall behind the crack.
    @mask    force background to black.
    @nocrack ERASE the crack and keep only its surroundings. This is a
             control, not a method: if it scores well, the benchmark is
             measuring wall-patch matching rather than crack re-ID.
    """
    parts = key.split("@")
    base, context, apply_mask, invert = parts[0], 0.0, None, False
    for flag in parts[1:]:
        if flag.startswith("ctx"):
            context = float(flag[3:]) if flag[3:] else 1.0
        elif flag == "nomask":
            apply_mask = False
        elif flag == "mask":
            apply_mask = True
        elif flag == "nocrack":
            invert = True
        else:
            raise ValueError(f"unknown flag '@{flag}' in method key {key!r}")
    return base, context, apply_mask, invert


def build_scorers(methods: list[str], data: Dataset, prune: bool = True):
    """Instantiate reid_eval scorers for the requested method keys,
    skipping any whose heavy dependency is not installed."""
    from crack_reid_baselines import REGISTRY

    def _suffix(context, apply_mask, default_mask, invert=False):
        bits = []
        if context:
            bits.append(f"+ctx{context:g}")
        if apply_mask is not None and apply_mask != default_mask:
            bits.append("+nomask" if not apply_mask else "+mask")
        if invert:
            bits.append("+NOCRACK")
        return "".join(bits)

    scorers = []
    for raw in methods:
        key, context, apply_mask, invert = parse_method_key(raw)
        try:
            if key in ("sift", "orb", "superglue", "loftr"):
                m = REGISTRY[key]()
                scorers.append(PairwiseMatcherScorer(
                    m, prune=prune, context=context, apply_mask=apply_mask,
                    invert=invert,
                    name=m.name + _suffix(context, apply_mask, m.apply_mask, invert),
                ))
            elif key == "shape":
                from crack_reid_baselines import crack_shape_descriptor
                fn = lambda masks: _l2(np.stack([crack_shape_descriptor(m) for m in masks]))
                scorers.append(ShapeEmbeddingScorer(
                    name="CrackShape" + _suffix(context, None, None, invert),
                    embed_fn=fn, context=context))
            elif key in ("deit", "vit", "clip", "osnet", "yolo"):
                emb = REGISTRY[key]()
                # wrap embed_batch as an embed_fn over BGR crops
                fn = lambda crops, e=emb: _l2(e.embed_batch(crops))
                if apply_mask is not None:
                    print(f"  (@mask/@nomask has no effect on '{key}': "
                          f"embedders always get the unmasked crop)")
                if context or invert:
                    scorers.append(ContextEmbeddingScorer(
                        name=emb.name + _suffix(context, None, None, invert),
                        embed_fn=fn, context=context, invert=invert,
                        input_scope="crop+ctx" if context else "crop",
                    ))
                else:
                    scorers.append(EmbeddingScorer(
                        name=emb.name, embed_fn=fn, input_scope="crop",
                    ))
            elif key == "registration":
                scorers.append(_build_registration_scorer(data, prune=prune))
            else:
                print(f"  (unknown method '{raw}', skipped)")
        except ImportError as e:
            print(f"  ({raw} skipped: {_missing(e)})")
    return scorers


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


class PrunedRegistrationScorer(RegistrationScorer):
    """RegistrationScorer that only registers image pairs the protocol can
    score. The parent registers every (query image, gallery image) pair --
    for a 74-photo test split that is 5,476 full-image SIFT+RANSAC
    registrations, of which only the same-wall ones are ever read."""

    def __init__(self, *a, prune: bool = True, **kw):
        super().__init__(*a, **kw)
        self.prune = prune

    def distance_matrix(self, queries, gallery, data):
        if not self.prune:
            return super().distance_matrix(queries, gallery, data)
        valid = valid_mask(queries, gallery)
        d = np.full((len(queries), len(gallery)), self.fail_distance, dtype=float)
        by_pair = defaultdict(list)
        for i, q in enumerate(queries):
            for j in np.nonzero(valid[i])[0]:
                by_pair[(q.image_id, gallery[j].image_id)].append((i, int(j)))

        todo, done, t0 = int(valid.sum()), 0, time.time()
        n_pairs = n_reg_fail = n_cells_unregistered = 0
        for (a_id, b_id), cells in by_pair.items():
            H = self._homography(a_id, b_id, data)
            n_pairs += 1
            if H is None:
                n_reg_fail += 1
                n_cells_unregistered += len(cells)
                done += len(cells)
                continue
            for i, j in cells:
                d[i, j] = self.chamfer_fn(queries[i], gallery[j], H, data)
                done += 1
            _progress(self.name, done, todo, t0)
        _progress(self.name, todo, todo, t0)
        # Reported separately because they mean opposite things: a failed
        # registration is a non-answer, an out-of-frame crack is a confident
        # rejection. Merging them into one `scored` column is what made
        # `scored = 0.34` read as "registration fails two thirds of the time"
        # when the true image-pair failure rate is what this records.
        self.coverage = {
            "image_pairs": n_pairs,
            "registration_failures": n_reg_fail,
            "registration_failure_rate": n_reg_fail / max(n_pairs, 1),
            "cells_unregistered": n_cells_unregistered,
            "cells_total": int(valid.sum()),
        }
        if n_pairs:
            print(f"    {self.name}: registered {n_pairs - n_reg_fail}/{n_pairs} "
                  f"image pairs ({1 - n_reg_fail / n_pairs:.0%})")
        return d


def _build_registration_scorer(data: Dataset, prune: bool = True):
    """Wire the registration pipeline into reid_eval's RegistrationScorer."""
    from crack_registration_reid import register_images, symmetric_chamfer

    def register_fn(img_a_id, img_b_id, _data):
        reg = register_images(data.image(img_a_id), data.image(img_b_id),
                              data.mask(img_a_id), data.mask(img_b_id))
        return reg.H if reg.ok else None

    # A query crack that warps OUTSIDE the gallery photo is not an
    # unanswerable pair -- it is a confident, geometrically grounded "this
    # crack is not in that photo", which no crop-scope method can produce.
    # Returning inf for it conflated two opposite events: "I could not
    # register these images" and "I registered them and the crack is not
    # there". Across the test split those are 63% and 10% of unscored
    # cells respectively, and collapsing them made `scored` unreadable and
    # handed registration an advantage disguised as a weakness.
    OUT_OF_FRAME = 1e6            # finite: ranks last, but IS an answer

    def chamfer_fn(q_ref, g_ref, H, _data):
        # warp the query instance's mask into gallery frame, then Chamfer
        inst_q = data.instance_of(q_ref)
        inst_g = data.instance_of(g_ref)
        hb, wb = data.image(g_ref.image_id).shape[:2]
        full_q = np.zeros((data.image(q_ref.image_id).shape[0],
                           data.image(q_ref.image_id).shape[1]), np.uint8)
        x, y, w, h = inst_q.bbox
        full_q[y:y + inst_q.mask_crop.shape[0], x:x + inst_q.mask_crop.shape[1]] = inst_q.mask_crop
        warped = cv2.warpPerspective(full_q, H, (wb, hb), flags=cv2.INTER_NEAREST)
        ys, xs = np.nonzero(warped)
        if len(xs) == 0:
            return OUT_OF_FRAME
        from crack_registration_reid import CrackInstance as RegInst
        q_w = RegInst(label=0, pixels=np.stack([xs, ys], 1),
                      bbox=(0, 0, wb, hb), area=len(xs),
                      centroid=(xs.mean(), ys.mean()),
                      mask=(warped > 0).astype(np.uint8))
        gx, gy, gw, gh = inst_g.bbox
        gfull = np.zeros((hb, wb), np.uint8)
        gfull[gy:gy + inst_g.mask_crop.shape[0], gx:gx + inst_g.mask_crop.shape[1]] = inst_g.mask_crop
        gys, gxs = np.nonzero(gfull)
        g_full = RegInst(label=0, pixels=np.stack([gxs, gys], 1),
                         bbox=inst_g.bbox, area=len(gxs),
                         centroid=(gxs.mean(), gys.mean()),
                         mask=(gfull > 0).astype(np.uint8))
        return symmetric_chamfer(q_w, g_full, (hb, wb))

    return PrunedRegistrationScorer(register_fn, chamfer_fn, prune=prune)


# ===========================================================================
# 4. Persisting score matrices
# ===========================================================================
# The score matrix is the only expensive artefact in this benchmark: hours
# for SuperGlue/LoFTR/registration, seconds for every metric computed from
# it. Saving it means a new metric, a bootstrap CI, a coverage-matched
# comparison, or merging runs that happened on different days all become
# free. reid_analysis.py consumes exactly these files.

def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _ref_arrays(prefix: str, refs: list[InstanceRef]) -> dict:
    """Everything needed to rebuild the validity and relevance masks later,
    so the analysis never has to reconstruct the Dataset."""
    return {
        f"{prefix}_instance_id": np.array([r.instance_id for r in refs]),
        f"{prefix}_image_id":    np.array([r.image_id for r in refs]),
        f"{prefix}_wall_id":     np.array([r.wall_id for r in refs]),
        f"{prefix}_session":     np.array([r.session for r in refs]),
        f"{prefix}_identity":    np.array([r.identity if r.identity else "" for r in refs]),
    }


def score_and_save(scorer, queries, gallery, data, out_dir: str, split: str):
    """Score once, persist, return (scores, prepare_s, score_s)."""
    t0 = time.perf_counter()
    scorer.prepare(list(dict.fromkeys(list(queries) + list(gallery))), data)
    t_prepare = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores = np.asarray(scorer.score_matrix(queries, gallery, data), dtype=float)
    t_score = time.perf_counter() - t0

    d = os.path.join(out_dir, "scores")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{_slug(scorer.name)}__{split}.npz")
    np.savez_compressed(
        path,
        scores=scores,
        method=np.array(scorer.name),
        input_scope=np.array(scorer.input_scope),
        split=np.array(split),
        prepare_seconds=np.array(t_prepare),
        score_seconds=np.array(t_score),
        **_ref_arrays("query", queries),
        **_ref_arrays("gallery", gallery),
    )
    print(f"    saved {path}")
    return scores, t_prepare, t_score


def _threshold_from_scores(scores, queries, gallery) -> float:
    """Best-F1 operating point on the validation matrix. Same logic as
    reid_eval.calibrate_threshold, but reusing a matrix we already have
    instead of recomputing it."""
    from reid_eval import build_relevance, pair_pr_curve
    valid = valid_mask(queries, gallery)
    rel = build_relevance(queries, gallery)
    pr = pair_pr_curve(np.asarray(scores, dtype=float), rel, valid)
    if not pr["f1"]:
        return 0.0
    return float(pr["thresholds"][int(np.argmax(pr["f1"]))])


# ===========================================================================
# 5. Driver
# ===========================================================================

def run(root: str, methods: list[str] | None = None,
        out_dir: str = "benchmark_out", prune: bool = True, seed: int = 0,
        min_frame_gap: int = 0, min_area: int = 200, close_px: int = 5):
    PROTOCOL["min_frame_gap"] = int(min_frame_gap)
    methods = methods or ["sift", "orb", "loftr", "superglue",
                          "deit", "vit", "clip", "osnet", "yolo",
                          "shape", "registration"]
    os.makedirs(out_dir, exist_ok=True)

    # Measured, not assumed: registration's USAC_MAGSAC returns DIFFERENT
    # homographies under different seeds, while ORB's plain RANSAC came out
    # identical across seeds in this build. So the seed matters most for the
    # method whose numbers the paper leans on. Fixing it makes a run
    # reproducible; it does not make the number stable, so quote the
    # seed-to-seed spread rather than a single decimal (run a few seeds into
    # separate --out dirs and compare).
    cv2.setRNGSeed(seed)
    np.random.seed(seed)

    data = Dataset(root, min_area=min_area, close_px=close_px)
    print(data.labelled_summary())

    splits = _load_splits(root)
    val_walls = set(splits.get("val", []))
    test_walls = set(splits.get("test", []))
    if not test_walls:
        print("\nNo test split defined; evaluating on all walls (report as such).")
        test_walls = {r.wall_id for r in data.refs}

    scorers = build_scorers(methods, data, prune=prune)
    results = []

    for scorer in scorers:
        try:
            # Calibrate the operating threshold on validation walls only, if any.
            # The val matrix is saved too: it cost the same as the test one and
            # is what any later re-calibration needs.
            thr = None
            if val_walls:
                vq, vg = data.query_gallery(val_walls)
                if vq:
                    v_scores, _, _ = score_and_save(scorer, vq, vg, data, out_dir, "val")
                    thr = _threshold_from_scores(v_scores, vq, vg)

            tq, tg = data.query_gallery(test_walls)
            t_scores, t_prep, t_sc = score_and_save(scorer, tq, tg, data, out_dir, "test")
            res = evaluate(scorer, tq, tg, data, threshold=thr,
                           scores=t_scores, timing=(t_prep, t_sc), **PROTOCOL)
            results.append(res)
            _dump_curves(res, out_dir)
        except ImportError as e:
            # Heavy backbones raise lazily, inside prepare(); skip cleanly.
            print(f"  ({scorer.name} skipped: {_missing(e)})")
        except Exception as e:
            print(f"  ({scorer.name} error: {type(e).__name__}: {e}, skipped)")

    table = results_table(results)
    print("\n" + table)
    with open(os.path.join(out_dir, "results_table.txt"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump([{k: v for k, v in r.items() if not k.startswith("_")}
                   for r in results], f, indent=2)
    print(f"\nWrote results to {out_dir}/")
    return results


def run_lowo(root: str, methods: list[str] | None = None,
             out_dir: str = "benchmark_out", prune: bool = True, seed: int = 0,
             min_frame_gap: int = 0, min_area: int = 200, close_px: int = 5):
    """Leave-one-wall-out over every wall, instead of a fixed val/test split.

    The fixed split wastes the data and reports the weaker half: validation
    carries 385 answerable queries over 6 walls while test carries 161 over
    11, four of which contribute none at all -- so the headline numbers come
    from the smaller, noisier partition while the larger one only picks a
    threshold.

    Here every wall is held out in turn, its operating threshold calibrated
    on the other 16, and the per-fold score matrices saved under
    <out>/lowo/<wall>/. Every query contributes exactly once, and the wall
    stays the unit of replication for reid_analysis's cluster bootstrap --
    which is the interval the paper should quote.
    """
    PROTOCOL["min_frame_gap"] = int(min_frame_gap)
    cv2.setRNGSeed(seed)
    np.random.seed(seed)

    data = Dataset(root, min_area=min_area, close_px=close_px)
    print(data.labelled_summary())
    walls = sorted({r.wall_id for r in data.refs})
    print(f"\nLeave-one-wall-out over {len(walls)} walls")

    per_fold = {}
    for held in walls:
        rest = set(walls) - {held}
        tq, tg = data.query_gallery({held})
        if not any(r.identity for r in tq):
            print(f"  ({held}: no labelled query, skipped)")
            continue
        fold_dir = os.path.join(out_dir, "lowo", held)
        os.makedirs(fold_dir, exist_ok=True)
        scorers = build_scorers(methods or ["sift"], data, prune=prune)
        for scorer in scorers:
            try:
                thr = None
                vq, vg = data.query_gallery(rest)
                if vq:
                    v, _, _ = score_and_save(scorer, vq, vg, data, fold_dir, "val")
                    thr = _threshold_from_scores(v, vq, vg)
                t, tp, ts = score_and_save(scorer, tq, tg, data, fold_dir, "test")
                res = evaluate(scorer, tq, tg, data, threshold=thr,
                               scores=t, timing=(tp, ts), **PROTOCOL)
                per_fold.setdefault(scorer.name, []).append((held, res))
            except Exception as e:
                print(f"  ({held}/{scorer.name} error: {type(e).__name__}: {e})")

    print("\nPooled over folds (mean +/- sd across walls):")
    for name, folds in sorted(per_fold.items()):
        r1 = np.array([r["closed_set"]["rank1"] for _, r in folds])
        mp = np.array([r["closed_set"]["mAP"] for _, r in folds])
        print(f"  {name:<24} R@1 {r1.mean():.3f}+/-{r1.std():.3f}   "
              f"mAP {mp.mean():.3f}+/-{mp.std():.3f}   ({len(folds)} folds)")
    with open(os.path.join(out_dir, "lowo_results.json"), "w") as f:
        json.dump({k: [{"wall": w, **{kk: vv for kk, vv in r.items()
                                      if not kk.startswith("_")}} for w, r in v]
                   for k, v in per_fold.items()}, f, indent=2)
    return per_fold


def _load_splits(root: str) -> dict:
    path = os.path.join(root, "splits.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _dump_curves(res: dict, out_dir: str):
    curves = res.get("_curves", {})
    with open(os.path.join(out_dir, f"curves_{res['method']}.json"), "w") as f:
        json.dump(curves, f)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="dataset root (contains walls.csv, images/, masks/, labels/)")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--out", default="benchmark_out")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for RANSAC (SIFT/ORB/SuperGlue/LoFTR/registration)")
    ap.add_argument("--min-frame-gap", type=int, default=0,
                    help="drop gallery photos within N frames of the query's own "
                         "frame. Each wall was shot in one continuous pass and the "
                         "nearest correct answer is the ADJACENT frame for 100%% of "
                         "answerable queries, so 0 measures near-duplicate retrieval. "
                         "Sweep 0..3 and report the curve.")
    ap.add_argument("--min-area", type=int, default=200,
                    help="minimum crack pixels for a connected component to count. "
                         "On a 12 MP photo the 10th-percentile component is 236 px, "
                         "which is the boundary between a hairline and segmentation "
                         "speckle -- worth sweeping rather than assuming.")
    ap.add_argument("--close-px", type=int, default=5,
                    help="morphological closing before labelling. NOTE 5 px on a "
                         "4080 px image is effectively no closing at all, so the "
                         "'one crack, many components' case this was meant to merge "
                         "is not merged; a fragmented crack becomes several "
                         "identities instead. Raise it, or merge fragments at "
                         "label time.")
    ap.add_argument("--lowo", action="store_true",
                    help="leave-one-wall-out over all walls instead of the fixed "
                         "val/test split, so every wall contributes")
    ap.add_argument("--no-prune", action="store_true",
                    help="score every query-gallery cell, including the ones the "
                         "protocol discards (slower; results are identical)")
    args = ap.parse_args()
    driver = run_lowo if args.lowo else run
    driver(args.root, methods=args.methods, out_dir=args.out,
           prune=not args.no_prune, seed=args.seed,
           min_frame_gap=args.min_frame_gap,
           min_area=args.min_area, close_px=args.close_px)