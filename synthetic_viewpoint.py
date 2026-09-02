"""
Synthetic-viewpoint ablation: apply a KNOWN homography to real photographs
and measure retrieval accuracy as a function of the applied transform,
instead of the transform *inferred* by viewpoint.py's AKAZE/ECC front end.

Why this exists (see REVIEW_RESPONSE.md S6 / SS6 "still open"): the paper's
viewpoint-sensitivity claim currently rests on a covariate ESTIMATED from
whatever pairs happened to occur during capture -- inferred, not designed.
A real second capture session at marked distances/angles is the clean fix
but costs a field trip. This script gets most of the same evidence for the
cost of a script run: every OTHER variable stays identical to the real
evaluation -- real crack appearance, real photometric noise, real masks --
and only the viewpoint transform is synthesized, with EXACT ground truth
instead of an estimate.

Method
------
For each admitted source photo, synthesize a QUERY image by warping the
photo, its mask, and its label points by an affine transform with a chosen
isotropic scale, in-plane rotation, and out-of-plane tilt -- parameterised
EXACTLY the way viewpoint.decompose() reads them back out of an estimated
homography. That makes the ground truth self-checking: decomposing the
transform we constructed must recover the parameters we asked for, up to
floating point, or the synthesis has a bug. See `_selftest()`.

The GALLERY stays 100% real: every other real photo of the same wall,
untouched. So every method is still matching a warped-but-real crop
against a real photograph -- nothing about crack appearance, texture, or
sensor noise is synthesized, only the geometric relationship between the
two views.

Tilt is applied as anisotropic in-plane scaling (the weak-perspective
approximation), NOT a true projective warp -- consistent with
viewpoint.decompose(), which is deliberately intrinsics-free because this
dataset's two handsets don't carry reliable focal-length EXIF. A
consequence worth knowing: the synthesized transforms are exactly affine,
so `projectivity` comes back ~1.0 by construction. That's fine for
isolating scale/rotation/tilt sensitivity, but it means this ablation
cannot speak to the projectivity column in the real viewpoint table --
say so if you quote both side by side.

Usage
-----
    python synthetic_viewpoint.py dataset --out synth_sweep_out \
        --scales 1.0,1.5,2.0,3.0 --rotations 0,15,30,45 --tilts 0,30,60 \
        --methods registration sift orb --min-sharpness 10

Requires the same deps as benchmark.py for whichever --methods you pass.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict

import cv2
import numpy as np

from benchmark import Dataset, Photo, PROTOCOL, PairwiseMatcherScorer, _build_registration_scorer
from crack_reid_baselines import REGISTRY as MATCHERS
from reid_eval import evaluate, results_table, InstanceRef
from viewpoint import decompose


# ===========================================================================
# 1. Constructing a homography with EXACT, known scale/rotation/tilt
# ===========================================================================

def make_transform(scale: float, rotation_deg: float, tilt_deg: float,
                    w: int, h: int) -> np.ndarray:
    """3x3 affine H (bottom row [0,0,1]) with the requested viewpoint
    components, applied about the image centre.

    Built so that `viewpoint.decompose(H, (h, w))` recovers exactly
    `scale`, `rotation_deg`, `tilt_deg` back out -- see `_selftest()`.
    Constant Jacobian everywhere (it's affine), so `projectivity` == 1.0.
    """
    if not (0.0 <= tilt_deg < 90.0):
        raise ValueError("tilt_deg must be in [0, 90)")
    theta = np.radians(rotation_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    aniso = 1.0 / np.cos(np.radians(tilt_deg))      # >= 1
    sigma1, sigma2 = scale * np.sqrt(aniso), scale / np.sqrt(aniso)
    J = R @ np.diag([sigma1, sigma2])

    cx, cy = w / 2.0, h / 2.0
    A = np.eye(3)
    A[:2, :2] = J
    A[:2, 2] = [cx, cy] - J @ [cx, cy]
    return A


def _selftest():
    """Round-trip check: build a transform, decompose it, compare."""
    w, h = 4000, 3000
    for scale in (0.6, 1.0, 1.8, 3.0):
        for rot in (0, 15, 40, -30):
            for tilt in (0, 20, 55, 75):
                H = make_transform(scale, rot, tilt, w, h)
                d = decompose(H, (h, w))
                assert d is not None
                assert abs(d["scale"] - scale) < 1e-6, (scale, d["scale"])
                # rotation can come back mod 360 with opposite sign convention
                dr = abs(((d["rotation_deg"] - rot) + 180) % 360 - 180)
                assert dr < 1e-4, (rot, d["rotation_deg"])
                assert abs(d["tilt_deg"] - tilt) < 1e-4, (tilt, d["tilt_deg"])
                assert abs(d["projectivity"] - 1.0) < 1e-6, d["projectivity"]
    print("synthetic_viewpoint self-test: OK")


# ===========================================================================
# 2. A Dataset that serves synthetic query images alongside the real ones
# ===========================================================================

class SyntheticViewpointDataset(Dataset):
    """Adds synthetic query photos on top of a real Dataset.

    image()/mask()/_load_points() are overridden to synthesize on first
    access and cache -- everything else (instances(), crop(), _window(),
    the registration/embedding scorers) goes through those two accessors
    polymorphically and needs no changes at all.
    """

    def __init__(self, *a, **kw):
        self._synth_recipe: dict[str, tuple[str, np.ndarray]] = {}
        self._synth_points: dict[str, list[dict]] = {}
        self.synth_transform: dict[str, dict] = {}
        super().__init__(*a, **kw)

    def image(self, image_id: str) -> np.ndarray:
        if image_id in self._synth_recipe and image_id not in self._img_cache:
            src_id, H = self._synth_recipe[image_id]
            img = super().image(src_id)
            self._img_cache[image_id] = cv2.warpAffine(
                img, H[:2, :], (img.shape[1], img.shape[0]),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return super().image(image_id)

    def mask(self, image_id: str) -> np.ndarray:
        if image_id in self._synth_recipe and image_id not in self._mask_cache:
            src_id, H = self._synth_recipe[image_id]
            m = super().mask(src_id)
            self._mask_cache[image_id] = cv2.warpAffine(
                m, H[:2, :], (m.shape[1], m.shape[0]),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return super().mask(image_id)

    def _load_points(self, image_id: str) -> list[dict]:
        if image_id in self._synth_points:
            return self._synth_points[image_id]
        return super()._load_points(image_id)

    def add_synthetic(self, source_image_id: str, scale: float,
                      rotation_deg: float, tilt_deg: float) -> str | None:
        """Register one synthetic query derived from `source_image_id`.

        Returns the new image_id, or None if the source has no labelled
        points (nothing to build a query out of).
        """
        src_points = self._load_points(source_image_id)
        if not src_points:
            return None
        src_photo = self.photos[source_image_id]
        img = self.image(source_image_id)
        h, w = img.shape[:2]
        H = make_transform(scale, rotation_deg, tilt_deg, w, h)

        tag = f"s{scale:g}_r{rotation_deg:g}_t{tilt_deg:g}"
        synth_id = f"{source_image_id}__SYN__{tag}"
        if synth_id in self.photos:
            return synth_id

        self.photos[synth_id] = Photo(
            image_id=synth_id, wall_id=src_photo.wall_id,
            session=f"{src_photo.session}__synth_{tag}",
            img_path="<synthetic>", mask_path="<synthetic>", label_path="<synthetic>")
        self._synth_recipe[synth_id] = (source_image_id, H)
        self.synth_transform[synth_id] = dict(
            source_image_id=source_image_id, scale=scale,
            rotation_deg=rotation_deg, tilt_deg=tilt_deg)

        pts = np.float32([p["xy"] for p in src_points]).reshape(-1, 1, 2)
        warped = cv2.transform(pts, H[:2, :]).reshape(-1, 2)
        self._synth_points[synth_id] = [
            {"identity": p["identity"], "xy": [int(round(x)), int(round(y))]}
            for p, (x, y) in zip(src_points, warped)
        ]

        insts = self.instances(synth_id, apply_mask=False)
        points = self._synth_points[synth_id]
        for c_idx, inst in enumerate(insts):
            identity = self._resolve_identity(inst, points)
            ref = InstanceRef(
                instance_id=f"{synth_id}_c{c_idx:02d}", image_id=synth_id,
                wall_id=src_photo.wall_id, session=self.photos[synth_id].session,
                identity=identity)
            self.refs.append(ref)
            self._ref_to_instance[ref.instance_id] = inst
        return synth_id


# ===========================================================================
# 3. Sweep + evaluate
# ===========================================================================

def build_scorers(names: list[str], data: Dataset) -> dict:
    scorers = {}
    for name in names:
        if name == "registration":
            scorers[name] = _build_registration_scorer(data, prune=True)
        elif name in MATCHERS:
            scorers[name] = PairwiseMatcherScorer(MATCHERS[name](), prune=True)
        else:
            raise ValueError(f"unknown method '{name}' (known: registration, {list(MATCHERS)})")
    return scorers


def run_sweep(root: str, scales, rotations, tilts, methods: list[str],
             min_sharpness: float | None = 10, max_sources_per_wall: int | None = 6,
             seed: int = 0) -> list[dict]:
    _selftest()

    data = SyntheticViewpointDataset(root, min_sharpness=min_sharpness)
    scorers = build_scorers(methods, data)

    rng = np.random.default_rng(seed)
    by_wall = defaultdict(list)
    for image_id, photo in data.photos.items():
        if data._load_points(image_id):
            by_wall[photo.wall_id].append(image_id)
    sources = []
    for wall_id, ids in by_wall.items():
        ids = sorted(ids)
        if max_sources_per_wall and len(ids) > max_sources_per_wall:
            ids = list(rng.choice(ids, size=max_sources_per_wall, replace=False))
        sources.extend(ids)

    bins = list(itertools.product(scales, rotations, tilts))
    print(f"{len(sources)} source photos x {len(bins)} transform bins x "
          f"{len(scorers)} methods = {len(sources) * len(bins) * len(scorers)} evaluations")

    rows = []
    for si, source_id in enumerate(sources):
        wall_id = data.photos[source_id].wall_id
        # Real, non-synthetic instances from every OTHER real photo of this
        # wall -- the source photo itself is excluded so a query can't
        # trivially match its own un-warped instance.
        real_gallery = [r for r in data.refs
                        if r.wall_id == wall_id
                        and r.image_id not in data._synth_recipe
                        and r.image_id != source_id]
        if not real_gallery:
            continue
        for scale, rot, tilt in bins:
            synth_id = data.add_synthetic(source_id, scale, rot, tilt)
            if synth_id is None:
                continue
            query_refs = [r for r in data.refs
                         if r.image_id == synth_id and r.identity is not None]
            if not query_refs:
                continue
            for method, scorer in scorers.items():
                res = evaluate(scorer, query_refs, real_gallery, data, **PROTOCOL)
                rows.append({
                    "method": method, "source_image_id": source_id,
                    "scale": scale, "rotation_deg": rot, "tilt_deg": tilt,
                    "n_queries": res["closed_set"]["n_queries"],
                    "rank1": res["closed_set"]["rank1"],
                    "mAP": res["closed_set"]["mAP"],
                    "dir_at_far10": res["open_set_dir_at_far10"],
                    "scoreable_pair_rate": res["scoreable_pair_rate"],
                })
        print(f"\r  {si + 1}/{len(sources)} source photos done", end="", flush=True)
    print()
    return rows


def summarize(rows: list[dict]) -> str:
    """One row per (method, scale, rotation, tilt) bin, averaged over
    source photos -- the designed analogue of reid_analysis's inferred
    viewpoint_sweep()."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["scale"], r["rotation_deg"], r["tilt_deg"])].append(r)

    hdr = f"{'method':<16}{'scale':>7}{'rot':>7}{'tilt':>7}{'n':>5}{'R@1':>7}{'mAP':>7}{'DIR.1':>7}{'scored':>8}"
    lines = [hdr, "-" * len(hdr)]
    for key in sorted(groups):
        method, scale, rot, tilt = key
        g = groups[key]
        n = sum(r["n_queries"] for r in g)
        w = np.array([r["n_queries"] for r in g], dtype=float)
        w = w / w.sum() if w.sum() else w
        rank1 = float(np.sum(w * [r["rank1"] for r in g]))
        mAP = float(np.sum(w * [r["mAP"] for r in g]))
        dirf = float(np.mean([r["dir_at_far10"] for r in g]))
        scored = float(np.mean([r["scoreable_pair_rate"] for r in g]))
        lines.append(f"{method:<16}{scale:>7.2f}{rot:>7.1f}{tilt:>7.1f}{n:>5d}"
                     f"{rank1:>7.3f}{mAP:>7.3f}{dirf:>7.3f}{scored:>8.2f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="synth_sweep_out")
    ap.add_argument("--scales", default="1.0,1.5,2.0,3.0")
    ap.add_argument("--rotations", default="0,15,30,45")
    ap.add_argument("--tilts", default="0,30,60")
    ap.add_argument("--methods", nargs="+", default=["registration", "sift", "orb"])
    ap.add_argument("--min-sharpness", type=float, default=10)
    ap.add_argument("--max-sources-per-wall", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scales = [float(x) for x in args.scales.split(",")]
    rotations = [float(x) for x in args.rotations.split(",")]
    tilts = [float(x) for x in args.tilts.split(",")]

    rows = run_sweep(args.root, scales, rotations, tilts, args.methods,
                     min_sharpness=args.min_sharpness,
                     max_sources_per_wall=args.max_sources_per_wall, seed=args.seed)

    import os
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "synthetic_viewpoint_rows.json"), "w") as f:
        json.dump(rows, f, indent=2)
    table = summarize(rows)
    print(table)
    with open(os.path.join(args.out, "synthetic_viewpoint_table.txt"), "w") as f:
        f.write(table + "\n")


if __name__ == "__main__":
    main()
