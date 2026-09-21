"""
Hybrid ReID evaluation on two designed "revisit" datasets.

The real dataset has one walk-around per wall: there are no true temporal
revisits, so a revisit-protocol experiment has to be SYNTHESIZED with exact
ground truth.  This is the same design decision as synthetic_viewpoint.py --
real crack appearance, real photometric noise, real masks, but the
query↔reference relationship is constructed, not inferred.

Two datasets (10 sources each; the same sources are typically reused so the
comparison is apples-to-apples):

  Dataset A - damage evolution.
      The query is the source photo warped by a modest viewpoint transform
      (the "later visit") whose mask additionally GROWS: one crack tip is
      extended and the stroke is thickened near it.  The pre-existing region
      is preserved by construction.  Reidentification must not collapse
      under legitimate evolution -- the old damage is still the old damage.

  Dataset B - significant appearance change, identical structure.
      The query is the source photo warped AND photometrically corrupted
      (blur, noise, contrast/brightness shift).  The mask is unchanged from
      the pure warp, so the structure is identical.  Appearance-based
      embeddings should degrade; the structural/hybrid path should hold.

Protocol
--------
For every synthetic query against a REAL gallery (all other real photos of
the same wall, source excluded), run the benchmark protocol (same-wall,
exclude-same-session, min-frame-gap 0) with closed-set Top-1/mAP and
open-set DIR at 10% FAR, plus scoreable-pair rate.  The hybrid scorer also
reports per-cell diagnostics: how often the verdict rested on a homography,
the skeleton fallback, or a blend, and the confidence distribution.

Usage
-----
    python hybrid_eval.py dataset --out hybrid_eval_out \\
        --methods hybrid registration skeleton osnet@ctx1 --seed 0

Only the passed methods are run; the defaults mix the hybrid with the
geometric baseline, the crop structural matcher, and an appearance
embedding to make the A/B contrast informative.

Resuming an interrupted run
---------------------------
Every evaluated cell is appended and flushed to
<out>/hybrid_eval_rows.jsonl as it completes, so a killed run loses at
most the current cell.  Re-run with the SAME --out, --methods, --seed and
--n-sources plus --resume (alias --continue) to replay the checkpoint and
skip the finished cells:

    python hybrid_eval.py dataset --out hybrid_eval_out \\
        --methods hybrid registration skeleton osnet@ctx1 \\
        --seed 0 --resume

Resume requires the same dataset recipe as the interrupted run (same
methods, sources and seed); changing them would mix new recipes into the
replayed old cells.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import cv2
import numpy as np

from hybrid_reid import HybridReIDScorer
from benchmark import Photo, PROTOCOL, build_scorers as build_benchmark_scorers
from reid_eval import evaluate, InstanceRef
from synthetic_viewpoint import SyntheticViewpointDataset, make_transform


# ===========================================================================
# 1. Mask growth / appearance corruption recipes
# ===========================================================================

def draw_line_mask(mask: np.ndarray, p0: tuple[int, int],
                   p1: tuple[int, int], thickness: int) -> np.ndarray:
    """Draw a thick line into a 1-channel uint8 mask deterministically.

    cv2.line() is unusable here: on 1-channel uint8 arrays several
    opencv-python 5.0.0 wheels silently fill the frame (a 40-px line wrote
    ~48k pixels), and the failure flips unpredictably with dtype.  This
    numpy rasterizer (distance-to-segment over a local bbox) is exact and
    wheel-independent.
    """
    h, w = mask.shape
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    d = p1 - p0
    L2 = float(d @ d)
    R = max(thickness / 2.0, 0.5)
    x0 = max(int(min(p0[0], p1[0]) - R), 0)
    x1 = min(int(max(p0[0], p1[0]) + R) + 1, w)
    y0 = max(int(min(p0[1], p1[1]) - R), 0)
    y1 = min(int(max(p0[1], p1[1]) + R) + 1, h)
    if x1 <= x0 or y1 <= y0:
        return mask
    ys, xs = np.mgrid[y0:y1, x0:x1]
    P = np.dstack([xs.astype(float), ys.astype(float)])
    t = np.clip(((P - p0) @ d) / L2, 0.0, 1.0) if L2 else np.zeros(P.shape[:2])
    proj = p0 + t[..., None] * d
    dist = np.sqrt(((P - proj) ** 2).sum(-1))
    mask[y0:y1, x0:x1][dist <= R] = 255
    return mask


def extend_crack_tip(mask: np.ndarray, growth_px: float, width_px: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Extend one endpoint of the largest crack component by `growth_px`.

    Only ADDS mask material -- the pre-existing region is byte-identical, by
    construction.  The tip chosen is the skeleton endpoint farthest from the
    component centroid, and the extension follows its outward direction so
    the growth looks like the crack kept propagating.
    """
    from skimage.morphology import skeletonize
    out = (mask > 0).astype(np.uint8).copy()
    binary = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n < 2:
        return out
    areas = stats[1:, cv2.CC_STAT_AREA]
    main_lab = int(np.argmax(areas))
    comp = (labels == (main_lab + 1)).astype(np.uint8)

    skel = skeletonize(comp > 0)
    ys, xs = np.nonzero(skel)
    if len(ys) < 4:
        return out
    degree = cv2.filter2D(skel.astype(np.uint8), cv2.CV_16S,
                          np.ones((3, 3), np.uint8)).astype(np.int16)
    degree -= skel.astype(np.int16)
    deg1 = np.c_[np.nonzero((skel > 0) & (degree == 1))[1],
                 np.nonzero((skel > 0) & (degree == 1))[0]]
    cy, cx = np.mean(ys), np.mean(xs)
    if len(deg1):
        tip = deg1[np.argmax((deg1[:, 0] - cx) ** 2 + (deg1[:, 1] - cy) ** 2)]
    else:
        tip = np.array([xs[np.argmax((xs - cx) ** 2 + (ys - cy) ** 2)],
                        ys[np.argmax((xs - cx) ** 2 + (ys - cy) ** 2)]])
    dirv = np.array([tip[0] - cx, tip[1] - cy], dtype=float)
    norm = np.linalg.norm(dirv)
    if norm < 1e-6:
        return out
    dirv = dirv / norm
    tip_pt = (int(round(tip[0])), int(round(tip[1])))
    end_pt = (int(round(tip[0] + dirv[0] * (growth_px + width_px))),
              int(round(tip[1] + dirv[1] * (growth_px + width_px))))
    draw_line_mask(out, tip_pt, end_pt, max(1, width_px))
    out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return out


def corrupt_appearance(img: np.ndarray, rng: np.random.Generator,
                       blur_sigma: float = 1.6, noise_sigma: float = 14.0,
                       gain: float = 0.85, offset: float = 18.0) -> np.ndarray:
    """Photometric corruption for Dataset B: blur + noise + contrast/shift.

    The geometry (and therefore the mask) is untouched.
    """
    g = float(rng.uniform(0.7, 1.0) + (gain / 2))
    o = float(rng.uniform(-offset * 0.7, offset * 0.7))
    img = cv2.GaussianBlur(img, (0, 0), blur_sigma)
    img = np.clip(g * img.astype(np.float32) + o, 0, 255).astype(np.uint8)
    noise = rng.normal(0.0, noise_sigma, size=img.shape)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return img


# ===========================================================================
# 2. Synthetic "revisit" dataset
# ===========================================================================

class HybridRevisitDataset(SyntheticViewpointDataset):
    """Synthetic query photos that optionally grow the mask / corrupt the
    appearance on top of the viewpoint warp."""

    def __init__(self, *a, **kw):
        self._revisit_recipe: dict[str, tuple] = {}
        super().__init__(*a, **kw)

    def image(self, image_id: str) -> np.ndarray:
        if image_id in self._revisit_recipe and image_id not in self._img_cache:
            src_id, H, _mask_edit, appearance = self._revisit_recipe[image_id]
            img = super().image(src_id)
            warped = cv2.warpAffine(img, H[:2, :], (img.shape[1], img.shape[0]),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
            if appearance is not None:
                warped = appearance(warped, np.random.default_rng(1729 + len(warped)))
            self._img_cache[image_id] = warped
        return super().image(image_id)

    def mask(self, image_id: str) -> np.ndarray:
        if image_id in self._revisit_recipe and image_id not in self._mask_cache:
            src_id, H, mask_edit, _appearance = self._revisit_recipe[image_id]
            m = super().mask(src_id)
            warped = cv2.warpAffine(m, H[:2, :], (m.shape[1], m.shape[0]),
                                    flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if mask_edit is not None:
                warped = mask_edit(warped, np.random.default_rng(2718 + len(warped)))
            self._mask_cache[image_id] = warped
        return super().mask(image_id)

    def add_revisit(self, source_image_id: str, scale: float,
                    rotation_deg: float, tilt_deg: float,
                    tag: str, mask_edit=None, appearance=None) -> str | None:
        src_points = self._load_points(source_image_id)
        if not src_points:
            return None
        src_photo = self.photos[source_image_id]
        img = self.image(source_image_id)
        h, w = img.shape[:2]
        H = make_transform(scale, rotation_deg, tilt_deg, w, h)

        synth_id = f"{source_image_id}__{tag}"
        if synth_id in self.photos:
            return synth_id

        self.photos[synth_id] = Photo(
            image_id=synth_id, wall_id=src_photo.wall_id,
            session=f"{src_photo.session}__{tag.lower()}",
            img_path="<synthetic>", mask_path="<synthetic>", label_path="<synthetic>")
        self._synth_recipe[synth_id] = (source_image_id, H)
        self._revisit_recipe[synth_id] = (source_image_id, H, mask_edit, appearance)
        self.synth_transform[synth_id] = dict(
            source_image_id=source_image_id, scale=scale,
            rotation_deg=rotation_deg, tilt_deg=tilt_deg, tag=tag)

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
# 3. Evaluation
# ===========================================================================

def build_scorers(names: list[str], data) -> dict:
    return {scorer.name: scorer for scorer in build_benchmark_scorers(names, data, prune=True)}


def pick_sources(data: HybridRevisitDataset, n: int, seed: int = 0) -> list[str]:
    """Sample sources whose cracks also appear on ANOTHER real photo of the
    same wall, so the closed-set gallery actually contains the answer."""
    rng = np.random.default_rng(seed)
    by_wall: dict[str, list[str]] = defaultdict(list)
    ids_by_img: dict[str, set] = {}
    for image_id, photo in data.photos.items():
        ids = {p["identity"] for p in data._load_points(image_id)}
        ids.discard(None)
        if not ids:
            continue
        ids_by_img[image_id] = ids
        by_wall[photo.wall_id].append(image_id)

    eligible: list[str] = []
    for wall_id, imgs in by_wall.items():
        for image_id in imgs:
            foreign = set().union(*(ids_by_img[o] for o in imgs if o != image_id))
            if ids_by_img[image_id] & foreign:
                eligible.append(image_id)
    eligible = sorted(eligible)
    pool = eligible or [img for imgs in by_wall.values() for img in imgs]
    if len(pool) <= n:
        return sorted(pool)
    return sorted(rng.choice(pool, size=n, replace=False).tolist())


def _cell_key(kind: str, method: str, dataset: str,
              source_image_id: str) -> tuple:
    """Identity of one evaluation/diagnostic cell, for resume bookkeeping."""
    return (kind, method, dataset, source_image_id)


def _completed_keys(path: str) -> set[tuple]:
    """Replay the checkpoint JSONL into the set of finished cell keys.

    Each line is an eval or diagnostic record tagged with "kind"; eval and
    diagnostic cells are tracked independently so a cell is never run twice
    but the diagnostic summary can still be rebuilt from the checkpoint.
    Records are returned in their routed (rows, diagnostics) lists by the
    caller via the same file re-read.
    """
    done: set[tuple] = set()
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                done.add(_cell_key(r["kind"], r["method"], r["dataset"],
                                   r["source_image_id"]))
    return done


def run_eval(root: str, methods: list[str], n_sources: int = 10,
             seed: int = 0, min_sharpness: float | None = 10,
             out_dir: str | None = None, resume: bool = False) -> dict:
    data = HybridRevisitDataset(root, min_sharpness=min_sharpness)
    scorers = build_scorers(methods, data)
    sources = pick_sources(data, n_sources, seed)
    print(f"{len(sources)} sources x ({len(scorers)} methods) x "
          f"(Dataset A evolution, Dataset B appearance)")

    # Checkpoint JSONL: every emitted record is appended and flushed, so a
    # killed run keeps everything before the current cell.  On resume the
    # same file is replayed into `rows`/`diag_rows` and the completed-key
    # set, and finished cells are skipped.
    ckpt_path = os.path.join(out_dir, "hybrid_eval_rows.jsonl") if out_dir else None
    if ckpt_path:
        os.makedirs(out_dir, exist_ok=True)
    rows: list[dict] = []
    diag_rows: list[dict] = []
    completed: set[tuple] = set()
    if resume and ckpt_path and os.path.isfile(ckpt_path):
        with open(ckpt_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.pop("kind") == "diag":
                    diag_rows.append(rec)
                else:
                    rows.append(rec)
        completed = _completed_keys(ckpt_path)
        print(f"resume: replayed {len(rows)} eval rows and "
              f"{len(diag_rows)} diagnostic rows, "
              f"{len(completed)} finished cells will be skipped")
    ckpt = open(ckpt_path, "a") if ckpt_path else None

    def emit(rec: dict) -> None:
        rows.append(rec)
        if ckpt is not None:
            ckpt.write(json.dumps({"kind": "eval", **rec}) + "\n")
            ckpt.flush()

    def emit_diag(rec: dict) -> None:
        diag_rows.append(rec)
        if ckpt is not None:
            ckpt.write(json.dumps({"kind": "diag", **rec}) + "\n")
            ckpt.flush()

    skipped = 0
    evaluated = 0
    for si, source_id in enumerate(sources):
        wall_id = data.photos[source_id].wall_id
        gallery = [r for r in data.refs
                   if r.wall_id == wall_id
                   and r.image_id not in data._synth_recipe
                   and r.image_id != source_id]
        if not gallery:
            continue

        # Dataset A: growth (pre-existing region preserved).
        growth = 0.18 * max(16.0, float(data.image(source_id).shape[1]) / 14)
        synth_a = data.add_revisit(
            source_id, scale=1.08, rotation_deg=6.0, tilt_deg=4.0,
            tag="EVO_A",
            mask_edit=lambda m, rng: extend_crack_tip(m, growth, 3, rng))
        # Dataset B: appearance corruption, identical structure.
        synth_b = data.add_revisit(
            source_id, scale=1.08, rotation_deg=6.0, tilt_deg=4.0,
            tag="APP_B", appearance=corrupt_appearance)

        for tag, synth_id in (("A", synth_a), ("B", synth_b)):
            if synth_id is None:
                continue
            q_refs = [r for r in data.refs if r.image_id == synth_id and r.identity is not None]
            if not q_refs:
                continue
            for name, scorer in scorers.items():
                if _cell_key("eval", name, tag, source_id) in completed:
                    skipped += 1
                    continue
                res = evaluate(scorer, q_refs, gallery, data, **PROTOCOL)
                evaluated += 1
                emit({
                    "dataset": tag, "method": name,
                    "source_image_id": source_id,
                    "n_queries": res["closed_set"]["n_queries"],
                    "rank1": res["closed_set"]["rank1"],
                    "mAP": res["closed_set"]["mAP"],
                    "dir_at_far10": res["open_set_dir_at_far10"],
                    "scoreable_pair_rate": res["scoreable_pair_rate"],
                    "pair_f1": res["pair_f1_at_threshold"],
                    "assign_f1": res["assignment"]["f1"],
                })
                if isinstance(scorer, HybridReIDScorer):
                    emit_diag({"dataset": tag, "method": name,
                               "source_image_id": source_id,
                               **scorer.diagnostics()})
        print(f"\r  {si + 1}/{len(sources)} sources done", end="", flush=True)
    if ckpt is not None:
        ckpt.close()
    print()
    if resume:
        print(f"resume: evaluated {evaluated} new cells, skipped {skipped} "
              f"already-finished cells")
    return {"rows": rows, "diagnostics": diag_rows}


def summarize(rows: list[dict]) -> str:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["method"])].append(r)
    hdr = (f"{'ds':>3}{'method':<22}{'n':>5}{'R@1':>7}{'mAP':>7}"
           f"{'DIR.1':>7}{'pF1':>7}{'aF1':>7}{'scored':>8}")
    lines = [hdr, "-" * len(hdr)]
    for key in sorted(groups):
        ds, method = key
        g = groups[key]
        n = sum(r["n_queries"] for r in g)
        w = np.array([r["n_queries"] for r in g], dtype=float)
        w = w / w.sum() if w.sum() else w
        rank1 = float(np.sum(w * [r["rank1"] for r in g]))
        mAP = float(np.sum(w * [r["mAP"] for r in g]))
        dirf = float(np.mean([r["dir_at_far10"] for r in g]))
        scored = float(np.mean([r["scoreable_pair_rate"] for r in g]))
        pf1 = float(np.sum(w * [r.get("pair_f1") or 0.0 for r in g]))
        af1 = float(np.sum(w * [r.get("assign_f1") or 0.0 for r in g]))
        lines.append(f"{ds:>3}{method:<22}{n:>5d}"
                     f"{rank1:>7.3f}{mAP:>7.3f}{dirf:>7.3f}"
                     f"{pf1:>7.3f}{af1:>7.3f}{scored:>8.2f}")
    return "\n".join(lines)


def summarize_diagnostics(diag_rows: list[dict]) -> str:
    groups = defaultdict(list)
    for r in diag_rows:
        groups[(r["dataset"], r["method"])].append(r)
    hdr = f"{'ds':>3}{'method':<22}{'cells':>7}{'H%':>7}{'SK%':>7}{'HY%':>7}{'conf':>7}"
    lines = [hdr, "-" * len(hdr)]
    for key in sorted(groups):
        ds, method = key
        g = groups[key]
        cells = sum(r["cells"] for r in g)
        fr = defaultdict(float)
        conf = []
        for r in g:
            for k, v in (r.get("method_fracs") or {}).items():
                fr[k] += v * r["cells"]
            if r["cells"]:
                conf.extend([r["confidence_mean"]] * r["cells"])
        if cells:
            fr = {k: v / cells for k, v in fr.items()}
        confm = float(np.mean(conf)) if conf else float("nan")
        lines.append(f"{ds:>3}{method:<22}{cells:>7d}"
                     f"{fr.get('HOMOGRAPHY', 0) * 100:>7.1f}"
                     f"{fr.get('SKELETON', 0) * 100:>7.1f}"
                     f"{fr.get('HYBRID', 0) * 100:>7.1f}"
                     f"{confm:>7.3f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--out", default="hybrid_eval_out")
    ap.add_argument("--methods", nargs="+",
                    default=["hybrid", "registration", "skeleton", "osnet@ctx1"])
    ap.add_argument("--n-sources", type=int, default=10)
    ap.add_argument("--min-sharpness", type=float, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", "--continue", dest="resume",
                    action="store_true",
                    help="Replay <out>/hybrid_eval_rows.jsonl and skip "
                         "already-finished cells. Requires the same "
                         "--methods, --seed and --n-sources as the "
                         "interrupted run.")
    args = ap.parse_args()

    result = run_eval(args.root, args.methods, n_sources=args.n_sources,
                      min_sharpness=args.min_sharpness, seed=args.seed,
                      out_dir=args.out, resume=args.resume)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "hybrid_eval_rows.json"), "w") as f:
        json.dump(result["rows"], f, indent=2)
    with open(os.path.join(args.out, "hybrid_diagnostics.json"), "w") as f:
        json.dump(result["diagnostics"], f, indent=2)

    table = summarize(result["rows"])
    print(table)
    with open(os.path.join(args.out, "hybrid_eval_table.txt"), "w") as f:
        f.write(table + "\n")

    if result["diagnostics"]:
        dtable = summarize_diagnostics(result["diagnostics"])
        print("\nPer-cell method usage (of scored cells):")
        print(dtable)
        with open(os.path.join(args.out, "hybrid_diagnostics_table.txt"), "w") as f:
            f.write(dtable + "\n")


if __name__ == "__main__":
    main()