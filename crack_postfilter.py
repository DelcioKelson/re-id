"""
Post-segmentation crack filter.

The UNet++ mask, run at native resolution, fires not only on cracks but on
other thin dark linear structures: architectural trim / coving edges, paint
lines, and surface marks. Those are genuine confusions — a crack detector
keys on dark elongated features, and a moulding edge is exactly that.

They are separable from real cracks by SHAPE, not appearance:

  * a real crack MEANDERS. Fit a line to the component; a crack leaves
    clear residual off that line, a trim edge / mark sits almost perfectly
    on it (straightness ~= 1.000 in the measurements).
  * a trim edge is THICKER. Cracks here are ~2 px wide; the moulding
    rendered ~6-7 px wide.
  * marks are small and/or round (low elongation).

This filter removes components that are too thick, too perfectly straight
(while also long — a short straight segment can be part of a crack), too
small, or too round, and keeps the rest. Thresholds are exposed and were
read off the real mask, not guessed; sweep them on a few labelled masks.

    pip install opencv-python numpy
"""

from __future__ import annotations

import cv2
import numpy as np


def _component_features(comp_mask: np.ndarray, dt: np.ndarray):
    ys, xs = np.nonzero(comp_mask)
    pts = np.stack([xs, ys], 1).astype(np.float32)
    c = pts - pts.mean(0)
    cov = (c.T @ c) / max(len(c), 1)
    evals, _ = np.linalg.eigh(cov)          # ascending: evals[0]=minor, [1]=major
    minor, major = float(evals[0]), float(evals[1])
    elong = (major / max(minor, 1e-6)) ** 0.5
    straight = 1.0 - minor / max(major, 1e-6)     # ~1.0 = lies on a line
    length = 4.0 * (major ** 0.5)                  # ~ extent along major axis
    width = 2.0 * float(dt[comp_mask].mean())      # mean thickness in px
    return dict(area=int(comp_mask.sum()), elong=elong, straight=straight,
                length=length, width=width)


def filter_crack_mask(mask: np.ndarray,
                      min_area: int = 60,
                      max_width: float = 4.5,
                      straight_max: float = 0.9995,
                      straight_len_min: float = 150.0,
                      min_elong: float = 3.0,
                      return_report: bool = False):
    """Keep crack-like components, drop trim edges / marks / blobs.

    min_area:         drop specks below this pixel count.
    max_width:        drop components thicker than this (px). Cracks here are
                      ~2 px; the moulding edge was ~6-7 px.
    straight_max +    drop a component only if it is BOTH almost perfectly
    straight_len_min: straight (>= straight_max) AND long (>= straight_len_min
                      px). This removes long trim edges without killing the
                      locally-straight segments of a real crack.
    min_elong:        drop round blobs (major/minor axis ratio below this).
    """
    binary = (mask > 0).astype(np.uint8)
    dt = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

    keep = np.zeros_like(binary)
    report = []
    for i in range(1, n):
        comp = labels == i
        f = _component_features(comp, dt)
        reason = None
        if f["area"] < min_area:
            reason = "small"
        elif f["width"] > max_width:
            reason = "thick (trim/edge)"
        elif f["elong"] < min_elong:
            reason = "round (mark)"
        elif f["straight"] >= straight_max and f["length"] >= straight_len_min:
            reason = "straight+long (trim)"
        if reason is None:
            keep[comp] = 255
        report.append((i, reason or "KEEP", f))

    if return_report:
        return keep, report
    return keep


def overlay(image_bgr, mask, color=(0, 0, 255)):
    out = image_bgr.copy()
    m = mask.astype(bool)
    out[m] = (0.45 * out[m] + 0.55 * np.array(color, np.float32)).astype(np.uint8)
    return out


if __name__ == "__main__":
    import sys
    mask_path = sys.argv[1] if len(sys.argv) > 1 else "mask.png"
    img_path = sys.argv[2] if len(sys.argv) > 2 else None

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    cleaned, report = filter_crack_mask(mask, return_report=True)

    kept = [r for r in report if r[1] == "KEEP"]
    dropped = [r for r in report if r[1] != "KEEP"]
    print(f"kept {len(kept)} component(s), dropped {len(dropped)}")
    for i, reason, f in sorted(dropped, key=lambda r: -r[2]["area"])[:10]:
        print(f"  drop c{i}: {reason:22s} area={f['area']:6d} width={f['width']:.1f} "
              f"straight={f['straight']:.3f} len={f['length']:.0f} elong={f['elong']:.1f}")

    cv2.imwrite("mask_filtered.png", cleaned)
    if img_path:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        cv2.imwrite("mask_filtered_overlay.jpg", overlay(img, cleaned))
    print("wrote mask_filtered.png")