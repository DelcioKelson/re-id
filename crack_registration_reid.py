"""
Registration-based crack re-identification.

Instead of matching crack crops against each other (which fails because
cracks are thin, low-texture and self-similar), this pipeline:

  1. Estimates a single homography between the two FULL images, using
     wall texture rather than the cracks themselves. Walls are planar,
     so a homography is the correct model.
  2. Warps mask A into image B's coordinate frame.
  3. Matches crack instances geometrically (Chamfer distance between
     mask pixels, plus dilated IoU), solved as a global assignment.
  4. Reports unmatched instances explicitly as new / disappeared,
     instead of forcing every query to an argmax.

Requires: opencv-contrib-python, numpy, scipy
    pip install opencv-contrib-python numpy scipy
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment


# ===========================================================================
# 1. Global registration
# ===========================================================================

@dataclass
class Registration:
    H: np.ndarray | None          # 3x3 homography mapping image A -> image B
    n_inliers: int
    n_matches: int
    inlier_ratio: float
    ok: bool
    reason: str = ""


def _exclude_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    """Return a uint8 mask of pixels ALLOWED for keypoint detection.

    Crack pixels are excluded (dilated a little) so that the homography is
    driven by wall texture. Otherwise a long crack can dominate the
    correspondence set and pull the fit toward matching the wrong crack.
    """
    if mask is None:
        return None
    crack = (mask > 0).astype(np.uint8)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        crack = cv2.dilate(crack, k)
    return (1 - crack) * 255


def _homography_is_sane(H: np.ndarray, shape: tuple[int, int],
                        max_scale: float = 4.0) -> tuple[bool, str]:
    """Reject degenerate homographies that RANSAC will happily return.

    Near-collinear or clustered correspondences produce matrices that fit
    the inliers but fold, mirror or wildly rescale the image. Cheapest
    reliable check: warp the image corners and inspect the resulting quad.
    """
    if H is None or not np.all(np.isfinite(H)):
        return False, "homography is None or non-finite"

    h, w = shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    # Convexity / orientation: signed area of the warped quad
    x, y = warped[:, 0], warped[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if area <= 0:
        return False, "warped image is mirrored or folded"

    scale = abs(area) / float(w * h)
    if not (1.0 / (max_scale ** 2) < scale < max_scale ** 2):
        return False, f"implausible area scale factor ({scale:.3f})"

    # Side lengths should not collapse
    sides = np.linalg.norm(np.roll(warped, -1, axis=0) - warped, axis=1)
    if sides.min() < 1e-3 * max(w, h):
        return False, "warped quad has a collapsed side"

    return True, ""


def register_images(img_a: np.ndarray, img_b: np.ndarray,
                    mask_a: np.ndarray | None = None,
                    mask_b: np.ndarray | None = None,
                    exclude_cracks: bool = True,
                    exclude_dilate: int = 15,
                    max_dim: int = 1600,
                    ratio_thresh: float = 0.75,
                    ransac_thresh: float = 4.0,
                    min_inliers: int = 25) -> Registration:
    """Estimate the homography mapping img_a into img_b's frame.

    Detection runs on downscaled copies for speed; the resulting
    homography is rescaled back to full resolution at the end.
    """
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY) if img_a.ndim == 3 else img_a
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY) if img_b.ndim == 3 else img_b

    scale_a = min(1.0, max_dim / max(gray_a.shape[:2]))
    scale_b = min(1.0, max_dim / max(gray_b.shape[:2]))

    small_a = cv2.resize(gray_a, None, fx=scale_a, fy=scale_a, interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(gray_b, None, fx=scale_b, fy=scale_b, interpolation=cv2.INTER_AREA)

    det_mask_a = det_mask_b = None
    if exclude_cracks:
        if mask_a is not None:
            m = cv2.resize(mask_a, (small_a.shape[1], small_a.shape[0]), interpolation=cv2.INTER_NEAREST)
            det_mask_a = _exclude_mask(m, exclude_dilate)
        if mask_b is not None:
            m = cv2.resize(mask_b, (small_b.shape[1], small_b.shape[0]), interpolation=cv2.INTER_NEAREST)
            det_mask_b = _exclude_mask(m, exclude_dilate)

    # CLAHE helps a lot when the two captures differ in exposure, which is
    # the normal case for inspections taken weeks or months apart.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    small_a = clahe.apply(small_a)
    small_b = clahe.apply(small_b)

    sift = cv2.SIFT_create(nfeatures=8000)
    kp_a, des_a = sift.detectAndCompute(small_a, det_mask_a)
    kp_b, des_b = sift.detectAndCompute(small_b, det_mask_b)

    if des_a is None or des_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        return Registration(None, 0, 0, 0.0, False, "too few keypoints on the wall")

    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    knn = matcher.knnMatch(des_a, des_b, k=2)

    good = [m for pair in knn if len(pair) == 2
            for m, n in [pair] if m.distance < ratio_thresh * n.distance]

    if len(good) < min_inliers:
        return Registration(None, 0, len(good), 0.0, False,
                            f"only {len(good)} ratio-test matches")

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    H_small, inliers = cv2.findHomography(pts_a, pts_b, method, ransac_thresh,
                                          maxIters=10000, confidence=0.9999)

    if H_small is None or inliers is None:
        return Registration(None, 0, len(good), 0.0, False, "RANSAC found no model")

    n_in = int(inliers.sum())
    ratio = n_in / len(good)

    if n_in < min_inliers:
        return Registration(None, n_in, len(good), ratio, False,
                            f"only {n_in} geometric inliers")

    # Undo the downscaling: H_full = S_b^-1 @ H_small @ S_a
    S_a = np.diag([scale_a, scale_a, 1.0]).astype(np.float64)
    S_b_inv = np.diag([1.0 / scale_b, 1.0 / scale_b, 1.0]).astype(np.float64)
    H = S_b_inv @ H_small @ S_a
    H /= H[2, 2]

    sane, reason = _homography_is_sane(H, gray_a.shape[:2])
    if not sane:
        return Registration(None, n_in, len(good), ratio, False, reason)

    return Registration(H, n_in, len(good), ratio, True)


# ===========================================================================
# 2. Instance extraction
# ===========================================================================

@dataclass
class CrackInstance:
    label: int
    pixels: np.ndarray            # (N, 2) array of (x, y) coordinates
    bbox: tuple                   # (x, y, w, h)
    area: int
    centroid: tuple
    mask: np.ndarray = field(repr=False, default=None)   # full-frame binary


def extract_instances(mask: np.ndarray, min_area: int = 200,
                      close_px: int = 5) -> list[CrackInstance]:
    """Connected components, after closing small gaps.

    The closing step matters: a crack that is continuous in one capture
    often breaks into several components in another when a thin section
    falls below the segmentation threshold. Without it, instance counts
    are not comparable between images.
    """
    binary = (mask > 0).astype(np.uint8)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    instances = []
    for lid in range(1, n_labels):
        area = int(stats[lid, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = (labels == lid)
        ys, xs = np.nonzero(comp)
        instances.append(CrackInstance(
            label=lid,
            pixels=np.stack([xs, ys], axis=1),
            bbox=(int(stats[lid, cv2.CC_STAT_LEFT]), int(stats[lid, cv2.CC_STAT_TOP]),
                  int(stats[lid, cv2.CC_STAT_WIDTH]), int(stats[lid, cv2.CC_STAT_HEIGHT])),
            area=area,
            centroid=(float(centroids[lid][0]), float(centroids[lid][1])),
            mask=comp.astype(np.uint8),
        ))
    return instances


# ===========================================================================
# 3. Geometric cost between instances
# ===========================================================================

def symmetric_chamfer(inst_a: CrackInstance, inst_b: CrackInstance,
                      shape: tuple[int, int], cap: float = 1e4) -> float:
    """Mean symmetric distance (px) between two instances' pixel sets.

    Chamfer distance is the primary criterion rather than IoU because
    cracks are only a few pixels wide: a 3px registration error can drive
    IoU to nearly zero for a pair that is in fact correctly aligned.
    """
    h, w = shape

    # Cheap rejection: if bounding boxes are far apart, skip the transform.
    ax, ay, aw, ah = inst_a.bbox
    bx, by, bw, bh = inst_b.bbox
    gap_x = max(0, max(ax - (bx + bw), bx - (ax + aw)))
    gap_y = max(0, max(ay - (by + bh), by - (ay + ah)))
    if np.hypot(gap_x, gap_y) > cap:
        return cap

    def dt_of(inst):
        occupied = np.zeros((h, w), np.uint8)
        occupied[inst.pixels[:, 1], inst.pixels[:, 0]] = 1
        return cv2.distanceTransform(1 - occupied, cv2.DIST_L2, 3)

    dt_b = dt_of(inst_b)
    d_ab = dt_b[inst_a.pixels[:, 1], inst_a.pixels[:, 0]].mean()

    dt_a = dt_of(inst_a)
    d_ba = dt_a[inst_b.pixels[:, 1], inst_b.pixels[:, 0]].mean()

    return float(min(0.5 * (d_ab + d_ba), cap))


def dilated_iou(inst_a: CrackInstance, inst_b: CrackInstance, dilate_px: int = 7) -> float:
    """IoU after dilating both masks, to tolerate small registration error."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    a = cv2.dilate(inst_a.mask, k).astype(bool)
    b = cv2.dilate(inst_b.mask, k).astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


# ===========================================================================
# 4. Matching
# ===========================================================================

@dataclass
class MatchResult:
    pairs: list                   # (idx_a, idx_b, chamfer_px, iou)
    unmatched_a: list             # disappeared / not re-observed
    unmatched_b: list             # newly appeared
    cost_matrix: np.ndarray
    iou_matrix: np.ndarray


def match_instances(instances_a: list[CrackInstance],
                    instances_b: list[CrackInstance],
                    shape: tuple[int, int],
                    max_chamfer_px: float = 20.0,
                    min_iou: float = 0.0) -> MatchResult:
    """Global one-to-one assignment, with an explicit rejection threshold.

    Solving the assignment globally (rather than taking an argmax per
    query) prevents two cracks in A from both claiming the same crack in
    B, which per-query matching cannot avoid.
    """
    n_a, n_b = len(instances_a), len(instances_b)
    cost = np.full((n_a, n_b), np.inf, dtype=np.float64)
    ious = np.zeros((n_a, n_b), dtype=np.float64)

    for i, a in enumerate(instances_a):
        for j, b in enumerate(instances_b):
            d = symmetric_chamfer(a, b, shape)
            cost[i, j] = d
            if d <= max_chamfer_px * 3:      # only compute IoU where plausible
                ious[i, j] = dilated_iou(a, b)

    if n_a == 0 or n_b == 0:
        return MatchResult([], list(range(n_a)), list(range(n_b)), cost, ious)

    # Hungarian cannot handle inf; substitute a large finite penalty.
    finite = np.where(np.isfinite(cost), cost, 0.0)
    big = float(finite.max()) * 10 + 1e3
    solvable = np.where(np.isfinite(cost), cost, big)

    rows, cols = linear_sum_assignment(solvable)

    pairs, matched_a, matched_b = [], set(), set()
    for i, j in zip(rows, cols):
        if cost[i, j] <= max_chamfer_px and ious[i, j] >= min_iou:
            pairs.append((int(i), int(j), float(cost[i, j]), float(ious[i, j])))
            matched_a.add(int(i))
            matched_b.add(int(j))

    return MatchResult(
        pairs=pairs,
        unmatched_a=[i for i in range(n_a) if i not in matched_a],
        unmatched_b=[j for j in range(n_b) if j not in matched_b],
        cost_matrix=cost,
        iou_matrix=ious,
    )


# ===========================================================================
# 5. End-to-end
# ===========================================================================

def reidentify(img_a, img_b, mask_a, mask_b, min_area: int = 200,
               max_chamfer_px: float = 20.0, close_px: int = 5):
    """Full pipeline. Returns (Registration, instances_a_warped, instances_b, MatchResult)."""
    if mask_a.shape[:2] != img_a.shape[:2]:
        mask_a = cv2.resize(mask_a, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_NEAREST)
    if mask_b.shape[:2] != img_b.shape[:2]:
        mask_b = cv2.resize(mask_b, (img_b.shape[1], img_b.shape[0]), interpolation=cv2.INTER_NEAREST)

    reg = register_images(img_a, img_b, mask_a, mask_b)
    if not reg.ok:
        return reg, [], [], None

    h_b, w_b = img_b.shape[:2]
    mask_a_warped = cv2.warpPerspective(
        (mask_a > 0).astype(np.uint8), reg.H, (w_b, h_b), flags=cv2.INTER_NEAREST
    )

    inst_a = extract_instances(mask_a_warped, min_area=min_area, close_px=close_px)
    inst_b = extract_instances(mask_b, min_area=min_area, close_px=close_px)

    result = match_instances(inst_a, inst_b, (h_b, w_b), max_chamfer_px=max_chamfer_px)
    return reg, inst_a, inst_b, result


def visualize(img_b, inst_a, inst_b, result, out_path: str = "crack_reid.png"):
    """Green = matched pair, red = only in A (gone), blue = only in B (new)."""
    canvas = img_b.copy()
    overlay = np.zeros_like(canvas)

    matched_a = {p[0] for p in result.pairs}
    matched_b = {p[1] for p in result.pairs}

    for i, inst in enumerate(inst_a):
        color = (0, 255, 0) if i in matched_a else (0, 0, 255)
        overlay[inst.mask.astype(bool)] = color
    for j, inst in enumerate(inst_b):
        if j not in matched_b:
            overlay[inst.mask.astype(bool)] = (255, 0, 0)

    canvas = cv2.addWeighted(canvas, 1.0, overlay, 0.7, 0)

    for i, j, d, iou in result.pairs:
        cx, cy = inst_b[j].centroid
        cv2.putText(canvas, f"{i}->{j} d={d:.1f}", (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(out_path, canvas)
    return canvas


# ===========================================================================
# Example usage
# ===========================================================================

if __name__ == "__main__":
    img_a = cv2.imread("/content/inspection_2025.jpg", cv2.IMREAD_COLOR)
    img_b = cv2.imread("/content/inspection_2026.jpg", cv2.IMREAD_COLOR)
    mask_a = cv2.imread("/content/mask_2025.png", cv2.IMREAD_GRAYSCALE)
    mask_b = cv2.imread("/content/mask_2026.png", cv2.IMREAD_GRAYSCALE)

    for name, arr in [("img_a", img_a), ("img_b", img_b), ("mask_a", mask_a), ("mask_b", mask_b)]:
        if arr is None:
            raise FileNotFoundError(f"Failed to load {name} - check the path")

    reg, inst_a, inst_b, result = reidentify(img_a, img_b, mask_a, mask_b)

    print(f"Registration: ok={reg.ok} inliers={reg.n_inliers}/{reg.n_matches} "
          f"ratio={reg.inlier_ratio:.2f} {reg.reason}")

    if not reg.ok:
        raise SystemExit("Registration failed - fall back to appearance matching or reject the pair")

    print(f"{len(inst_a)} instance(s) in A (warped), {len(inst_b)} in B\n")

    print("=== Matches ===")
    for i, j, d, iou in result.pairs:
        growth = (inst_b[j].area - inst_a[i].area) / max(inst_a[i].area, 1) * 100
        print(f"A{i} <-> B{j}   chamfer={d:5.1f}px  iou={iou:.3f}  area change={growth:+.1f}%")

    print(f"\nOnly in A (not re-observed): {result.unmatched_a}")
    print(f"Only in B (newly appeared):  {result.unmatched_b}")

    visualize(img_b, inst_a, inst_b, result)
