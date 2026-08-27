"""
The dependency every method in the benchmark shares, and nobody measures.

Masks come from one tiled UNet++ pass. They define the crops the embedders
see, the pixel sets the Chamfer distance measures, the components the
click points resolve to, and the regions excluded from SIFT detection. A
missed 2 px hairline is invisible to all eleven methods equally -- which
keeps the COMPARISON fair, and leaves the ABSOLUTE numbers uninterpretable,
because no reader can tell whether R@1 = 0.920 reflects matching quality or
segmentation quality.

Two things fix that, and this module does both.

  1. QUALITY AGAINST HAND-DRAWN MASKS on a held-out subset. Draw masks for
     a handful of photos by hand, put them in <root>/masks_gt/ under the
     same stem, and `--gt` reports IoU, precision, recall and boundary F1
     against them. Ten photos is enough to quote a number; the point is
     that a number exists.

  2. SENSITIVITY OF THE HEADLINE METRICS to min_area and close_px. If the
     ranking is stable across settings, say so -- that is a robustness
     result and it costs one afternoon of compute. If it is not stable,
     the paper needs to know before a reviewer finds out.

     NOTE, because an earlier review got this wrong: this is NOT a re-read
     of saved score matrices. min_area and close_px change which connected
     components exist, so they change the instance set, the identities
     resolved from click points, and the SHAPE of every matrix. Each
     setting is a genuine re-run. Use the cheap methods for the sweep --
     `--methods sift orb shape osnet` answers the stability question for a
     fraction of the cost of the full table.

    python segmentation_audit.py dataset --gt
    python segmentation_audit.py dataset --sweep --out benchmark_out \\
        --methods sift orb shape
"""

from __future__ import annotations

import csv
import itertools
import json
import os

import cv2
import numpy as np


# ===========================================================================
# 1. Mask quality against hand-drawn ground truth
# ===========================================================================

def _boundary(mask: np.ndarray, tol: int = 2) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    return cv2.dilate(mask, k)


def mask_scores(pred: np.ndarray, gt: np.ndarray, tol: int = 2) -> dict:
    """IoU, precision, recall and a tolerance-`tol` boundary F1.

    Boundary F1 with a tolerance is the honest measure for a structure one
    or two pixels wide: plain IoU on a hairline is dominated by whether the
    mask is offset by a single pixel, which is not what anyone means by a
    segmentation error here.
    """
    p = (pred > 0).astype(np.uint8)
    g = (gt > 0).astype(np.uint8)
    inter = int((p & g).sum())
    union = int((p | g).sum())
    tp_p = int((p & _boundary(g, tol)).sum())        # pred pixels near a gt pixel
    tp_g = int((g & _boundary(p, tol)).sum())        # gt pixels near a pred pixel
    prec = tp_p / max(int(p.sum()), 1)
    rec = tp_g / max(int(g.sum()), 1)
    return {
        "iou": inter / union if union else float("nan"),
        "precision": prec,
        "recall": rec,
        "boundary_f1": 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0,
        "gt_pixels": int(g.sum()),
        "pred_pixels": int(p.sum()),
    }


def gt_report(root: str, gt_dir: str = "masks_gt", tol: int = 2) -> str:
    d = os.path.join(root, gt_dir)
    if not os.path.isdir(d):
        return (f"\nno {d}/ -- draw masks by hand for ~10 photos, save them there under the\n"
                f"same stem as the photo, and re-run. Without this the paper cannot say "
                f"whether\nR@1 measures matching or segmentation, and a reviewer will ask.")
    rows = []
    for fn in sorted(os.listdir(d)):
        stem = os.path.splitext(fn)[0]
        gt = cv2.imread(os.path.join(d, fn), cv2.IMREAD_GRAYSCALE)
        pr = cv2.imread(os.path.join(root, "masks", f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if gt is None or pr is None:
            continue
        if pr.shape != gt.shape:
            pr = cv2.resize(pr, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        rows.append((stem, mask_scores(pr, gt, tol)))
    if not rows:
        return f"\n{d}/ exists but no photo in it matches a mask in {root}/masks/"

    L = ["", "=" * 84,
         f"SEGMENTATION QUALITY vs HAND-DRAWN MASKS  ({len(rows)} photos, "
         f"boundary tolerance {tol} px)",
         "=" * 84,
         f"{'photo':<26}{'IoU':>8}{'prec':>8}{'recall':>8}{'bF1':>8}{'gt px':>9}"]
    L.append("-" * len(L[-1]))
    for stem, m in rows:
        L.append(f"{stem:<26}{m['iou']:>8.3f}{m['precision']:>8.3f}"
                 f"{m['recall']:>8.3f}{m['boundary_f1']:>8.3f}{m['gt_pixels']:>9d}")
    agg = {k: float(np.mean([m[k] for _, m in rows]))
           for k in ("iou", "precision", "recall", "boundary_f1")}
    L.append("-" * 67)
    L.append(f"{'mean':<26}{agg['iou']:>8.3f}{agg['precision']:>8.3f}"
             f"{agg['recall']:>8.3f}{agg['boundary_f1']:>8.3f}")
    L += ["", "Quote the mean boundary F1 in the paper next to the headline metric, and say",
          "how many photos it was measured on. Recall is the number that bounds re-ID: a",
          "crack the segmenter misses cannot be re-identified by ANY of the eleven methods."]
    return "\n".join(L)


# ===========================================================================
# 2. Sensitivity of the ranking to the segmentation parameters
# ===========================================================================

def sweep(root: str, out_dir: str = "benchmark_out",
          methods: list[str] | None = None,
          min_areas=(100, 200, 400), close_pxs=(5, 15),
          min_sharpness: float | None = None, seed: int = 0) -> dict:
    """Re-run the benchmark at each (min_area, close_px) and collect mAP."""
    from benchmark import run

    results = {}
    for ma, cp in itertools.product(min_areas, close_pxs):
        tag = f"minarea{ma}_close{cp}"
        sub = os.path.join(out_dir, "seg", tag)
        print(f"\n=== segmentation setting {tag} ===")
        res = run(root, methods=methods, out_dir=sub, seed=seed,
                  min_area=ma, close_px=cp, min_sharpness=min_sharpness)
        results[tag] = {r["method"]: {"mAP": r["closed_set"]["mAP"],
                                      "rank1": r["closed_set"]["rank1"],
                                      "n_queries": r["closed_set"]["n_queries"]}
                        for r in res}
    with open(os.path.join(out_dir, "seg_sensitivity.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def format_sweep(results: dict) -> str:
    settings = list(results)
    methods = sorted({m for r in results.values() for m in r})
    L = ["", "=" * 100,
         "SEGMENTATION SENSITIVITY -- does the ranking survive a different min_area/close_px?",
         "=" * 100,
         f"{'method':<24}" + "".join(f"{s.replace('minarea', 'a'):>18}" for s in settings)]
    L.append("-" * len(L[-1]))
    orders = {}
    for s in settings:
        orders[s] = [m for m in sorted(results[s], key=lambda m: -results[s][m]["mAP"])]
    for m in sorted(methods, key=lambda m: -np.mean(
            [results[s][m]["mAP"] for s in settings if m in results[s]] or [0])):
        cells = ""
        for s in settings:
            r = results[s].get(m)
            cells += (f"{r['mAP']:>10.3f}/{r['n_queries']:<8d}" if r else f"{'-':>18}")
        L.append(f"{m:<24}{cells}")

    base = orders[settings[0]]
    moved = [m for m in base
             if any(m in orders[s] and orders[s].index(m) != base.index(m)
                    for s in settings[1:])]
    L += ["", "cells are mAP / queries with an answer (the query count MOVES because these",
          "settings change which connected components exist -- that is why this is a re-run",
          "and not a re-read of saved matrices)."]
    if moved:
        L.append(f"\n!! ranking is NOT stable: {moved} change position across settings.")
        L.append("   Report the sweep, not a single setting, and say which conclusions survive.")
    else:
        L.append("\nRanking is IDENTICAL at every setting. Say so in the paper -- it is a")
        L.append("robustness result, and it costs one paragraph to claim.")
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--gt", action="store_true", help="score masks/ against masks_gt/")
    ap.add_argument("--gt-dir", default="masks_gt")
    ap.add_argument("--tol", type=int, default=2, help="boundary tolerance in px")
    ap.add_argument("--sweep", action="store_true",
                    help="re-run the benchmark at several min_area/close_px settings")
    ap.add_argument("--out", default="benchmark_out")
    ap.add_argument("--methods", nargs="*", default=["sift", "orb", "shape"],
                    help="keep this cheap: the question is whether the RANKING moves")
    ap.add_argument("--min-areas", nargs="*", type=int, default=[100, 200, 400])
    ap.add_argument("--close-pxs", nargs="*", type=int, default=[5, 15])
    ap.add_argument("--min-sharpness", type=float, default=None)
    args = ap.parse_args()

    did = False
    if args.gt:
        print(gt_report(args.root, args.gt_dir, args.tol))
        did = True
    if args.sweep:
        res = sweep(args.root, args.out, methods=args.methods,
                    min_areas=tuple(args.min_areas), close_pxs=tuple(args.close_pxs),
                    min_sharpness=args.min_sharpness)
        print(format_sweep(res))
        did = True
    if not did:
        ap.print_help()
