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
    def crop(self, ref: InstanceRef, apply_mask: bool = False) -> np.ndarray:
        key = f"{ref.instance_id}_{apply_mask}"
        if key not in self._crop_cache:
            inst = self._ref_to_instance[ref.instance_id]
            if apply_mask:
                self._crop_cache[key] = cv2.bitwise_and(inst.crop, inst.crop, mask=inst.mask_crop)
            else:
                self._crop_cache[key] = inst.crop
        return self._crop_cache[key]

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
PROTOCOL = dict(same_wall_only=True, exclude_same_session=True)


def valid_mask(queries, gallery) -> np.ndarray:
    """Which (query, gallery) cells the metrics will actually read.

    reid_eval masks every invalid cell out of CMC, mAP, the open-set curve
    and the pair PR curve, so scoring them is pure waste -- and for the
    genuinely pairwise matchers (SuperGlue/LoFTR) that waste is a network
    forward pass each. Delegating to reid_eval's own function means the
    two can never disagree about what "valid" means."""
    from reid_eval import build_validity_mask
    return build_validity_mask(queries, gallery, **PROTOCOL)


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

    def __init__(self, matcher, prune: bool = True):
        self.matcher = matcher
        self.name = matcher.name
        self.input_scope = "crop"
        self.prune = prune
        self._prep: dict[str, object] = {}

    def prepare(self, refs, data: Dataset):
        todo = [r for r in refs if r.instance_id not in self._prep]
        if not todo:
            return
        insts = [data.instance_of(r) for r in todo]
        # matcher.prepare works on CrackInstance lists and honours apply_mask
        preps = self.matcher.prepare(
            [_maybe_mask(inst, self.matcher.apply_mask) for inst in insts]
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


def _maybe_mask(inst: CrackInstance, apply: bool) -> CrackInstance:
    if not apply:
        return inst
    masked = cv2.bitwise_and(inst.crop, inst.crop, mask=inst.mask_crop)
    return CrackInstance(crop=masked, mask_crop=inst.mask_crop, bbox=inst.bbox)


def build_scorers(methods: list[str], data: Dataset, prune: bool = True):
    """Instantiate reid_eval scorers for the requested method keys,
    skipping any whose heavy dependency is not installed."""
    from crack_reid_baselines import REGISTRY

    scorers = []
    for key in methods:
        try:
            if key in ("sift", "orb", "superglue", "loftr"):
                scorers.append(PairwiseMatcherScorer(REGISTRY[key](), prune=prune))
            elif key in ("deit", "vit", "clip", "osnet", "yolo"):
                emb = REGISTRY[key]()
                # wrap embed_batch as an embed_fn over BGR crops
                scorers.append(EmbeddingScorer(
                    name=emb.name,
                    embed_fn=lambda crops, e=emb: _l2(e.embed_batch(crops)),
                    input_scope="crop",
                ))
            elif key == "registration":
                scorers.append(_build_registration_scorer(data, prune=prune))
            else:
                print(f"  (unknown method '{key}', skipped)")
        except ImportError as e:
            print(f"  ({key} needs '{e.name}', skipped)")
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
        for (a_id, b_id), cells in by_pair.items():
            H = self._homography(a_id, b_id, data)
            if H is None:
                done += len(cells)
                continue
            for i, j in cells:
                d[i, j] = self.chamfer_fn(queries[i], gallery[j], H, data)
                done += 1
            _progress(self.name, done, todo, t0)
        _progress(self.name, todo, todo, t0)
        return d


def _build_registration_scorer(data: Dataset, prune: bool = True):
    """Wire the registration pipeline into reid_eval's RegistrationScorer."""
    from crack_registration_reid import register_images, symmetric_chamfer

    def register_fn(img_a_id, img_b_id, _data):
        reg = register_images(data.image(img_a_id), data.image(img_b_id),
                              data.mask(img_a_id), data.mask(img_b_id))
        return reg.H if reg.ok else None

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
            return np.inf
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
# 4. Driver
# ===========================================================================

def run(root: str, methods: list[str] | None = None,
        out_dir: str = "benchmark_out", prune: bool = True):
    methods = methods or ["sift", "orb", "loftr", "superglue",
                          "deit", "vit", "clip", "osnet", "yolo", "registration"]
    os.makedirs(out_dir, exist_ok=True)

    data = Dataset(root)
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
            thr = None
            if val_walls:
                vq, vg = data.query_gallery(val_walls)
                if vq:
                    thr = calibrate_threshold(scorer, vq, vg, data, **PROTOCOL)

            tq, tg = data.query_gallery(test_walls)
            res = evaluate(scorer, tq, tg, data, threshold=thr, **PROTOCOL)
            results.append(res)
            _dump_curves(res, out_dir)
        except ImportError as e:
            # Heavy backbones raise lazily, inside prepare(); skip cleanly.
            print(f"  ({scorer.name} needs '{e.name}', skipped)")
        except Exception as e:
            print(f"  ({scorer.name} error: {type(e).__name__}: {e}, skipped)")

    table = results_table(results)
    print("\n" + table)
    with open(os.path.join(out_dir, "results_table.txt"), "w") as f:
        f.write(table + "\n")
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "_curves"} for r in results], f, indent=2)
    print(f"\nWrote results to {out_dir}/")
    return results


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
    ap.add_argument("--no-prune", action="store_true",
                    help="score every query-gallery cell, including the ones the "
                         "protocol discards (slower; results are identical)")
    args = ap.parse_args()
    run(args.root, methods=args.methods, out_dir=args.out, prune=not args.no_prune)