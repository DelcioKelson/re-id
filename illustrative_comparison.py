"""Qualitative Skeleton vs. OSNet+ctx1 comparison on GIMP illustrations.

This intentionally does NOT use ``benchmark.py`` or ``reid_eval.py``.  The
images in illustrative_synthetic/ are near-clones of their listed source
photographs, edited to shorten one crack; they are useful for inspecting score
behaviour under a controlled structural loss, but are not independent revisits
and must never contribute to a reported retrieval metric.

The illustrations have no segmentation masks.  For a reproducible visual
comparison this script creates a *mask proxy* from the source segmentation and
the disclosed edit boundary in manifest.json: it keeps the target connected
component above the end of the GIMP taper and removes it below that boundary.
The proxy is stated in the output JSON and is not a claim that a segmenter
would make the same prediction on the edited JPEG.

Usage:
    python3 illustrative_comparison.py --out illustrative_comparison_out
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from crack_reid_baselines import CrackInstance, SkeletonLoFTRMatcher


ROOT = os.path.dirname(os.path.abspath(__file__))
ILLUSTRATIONS = os.path.join(ROOT, "illustrative_synthetic")


def _target_component(mask: np.ndarray, cutoff_y: int) -> np.ndarray:
    """Choose the edited component: largest one that crosses the cut line."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    candidates = []
    for label in range(1, n):
        y0 = stats[label, cv2.CC_STAT_TOP]
        y1 = y0 + stats[label, cv2.CC_STAT_HEIGHT]
        if y0 <= cutoff_y < y1:
            candidates.append((stats[label, cv2.CC_STAT_AREA], label))
    if not candidates:
        # A manifest without a crossing component is malformed for this use;
        # using the largest component keeps the failure deterministic/visible.
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    else:
        label = max(candidates)[1]
    return (labels == label).astype(np.uint8) * 255


def _largest_instance(image: np.ndarray, mask: np.ndarray) -> CrackInstance | None:
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n <= 1:
        return None
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = stats[label]
    crop = image[y:y + h, x:x + w].copy()
    return CrackInstance(crop=crop, mask_crop=(labels[y:y + h, x:x + w] == label).astype(np.uint8) * 255,
                         bbox=(int(x), int(y), int(w), int(h)))


def _context_crop(image: np.ndarray, inst: CrackInstance, context: float = 1.0) -> np.ndarray:
    x, y, w, h = inst.bbox
    x0, y0 = max(0, round(x - context * w)), max(0, round(y - context * h))
    x1, y1 = min(image.shape[1], round(x + (1 + context) * w)), min(image.shape[0], round(y + (1 + context) * h))
    return image[y0:y1, x0:x1].copy()


def _load_osnet():
    """Load OSNet once for the entire illustrative set, not once per wall."""
    try:
        from crack_reid_baselines import REGISTRY
        return REGISTRY["osnet"](), None
    except (ImportError, RuntimeError, OSError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _osnet_score(source_img, source_inst, edited_img, edited_inst, matcher) -> float:
    """Return cosine similarity for OSNet with one bbox-width of context.

    ``matcher`` is deliberately injected: construction downloads/loads weights,
    so doing it inside the per-wall loop turns a 17-pair comparison into 17
    model initialisations.
    """
    emb = matcher.embed_batch([
        _context_crop(source_img, source_inst),
        _context_crop(edited_img, edited_inst),
    ])
    # The OSNet embedder returns raw descriptors. Mirror benchmark.py's L2
    # normalisation so this field is truly a cosine similarity, not a raw dot.
    denom = max(float(np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])), 1e-12)
    return float(np.dot(emb[0], emb[1]) / denom)


def compare(illustrations: str, out_dir: str, skip_osnet: bool = False) -> list[dict]:
    with open(os.path.join(illustrations, "manifest.json")) as f:
        manifest = json.load(f)
    rows = []
    skeleton = SkeletonLoFTRMatcher()
    osnet, osnet_load_error = ((None, "skipped by --skip-osnet") if skip_osnet
                               else _load_osnet())
    for edited_name, recipe in sorted(manifest.items()):
        source_path = os.path.join(ROOT, recipe["source"])
        source_id = os.path.splitext(os.path.basename(source_path))[0]
        mask_path = os.path.join(ROOT, "dataset", "masks", source_id + ".png")
        edited_path = os.path.join(illustrations, edited_name)
        source_img = cv2.imread(source_path, cv2.IMREAD_COLOR)
        edited_img = cv2.imread(edited_path, cv2.IMREAD_COLOR)
        source_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if source_img is None or edited_img is None or source_mask is None:
            raise FileNotFoundError(f"missing source/image/mask for {edited_name}")
        if source_mask.shape != source_img.shape[:2]:
            source_mask = cv2.resize(source_mask, (source_img.shape[1], source_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # The taper end is the first fully erased row in the disclosed GIMP
        # recipe.  Only the selected target component is shortened.
        cutoff = int(recipe["taper_zone_y"][1])
        target = _target_component(source_mask, cutoff)
        shortened = target.copy()
        shortened[cutoff:] = 0
        a = _largest_instance(source_img, target)
        b = _largest_instance(edited_img, shortened)
        if a is None or b is None:
            raise RuntimeError(f"no target instance after shortening {edited_name}")
        structural = skeleton.explain_pair(skeleton.prepare([a])[0], skeleton.prepare([b])[0])
        osnet_score = None if osnet is None else _osnet_score(source_img, a, edited_img, b, osnet)
        rows.append({
            "illustration": edited_name, "source": recipe["source"],
            "disclosure": "QUALITATIVE ONLY: GIMP-edited near-clone, not a revisit or benchmark datum.",
            "mask_proxy": "source target component truncated below disclosed taper end",
            "cutoff_y": cutoff, "skeleton_loftr": structural,
            "osnet_ctx1_cosine": osnet_score, "osnet_status": osnet_load_error,
        })
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "illustrative_comparison.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print("QUALITATIVE ONLY — no benchmark metrics were computed.")
    print(f"wrote {os.path.join(out_dir, 'illustrative_comparison.json')}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Qualitative comparison on GIMP illustrative images only.")
    ap.add_argument("--illustrations", default=ILLUSTRATIONS)
    ap.add_argument("--out", default="illustrative_comparison_out")
    ap.add_argument("--skip-osnet", action="store_true")
    args = ap.parse_args()
    compare(args.illustrations, args.out, args.skip_osnet)
