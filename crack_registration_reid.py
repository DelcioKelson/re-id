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
    #: Which front-end produced H: "sift", "sift-nocrackmask", or "ecc".
    method: str = "sift"
    #: RMS reprojection error of the inliers, in full-resolution px. Use it
    #: to set the tolerance of downstream shape comparisons instead of a
    #: hard-coded constant -- see chamfer_coverage(tau=...).
    inlier_rms: float = float("nan")


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


MIN_WALL_KEYPOINTS = 300      # below this, the wall has no texture to key on


def _ecc_homography(gray_a: np.ndarray, gray_b: np.ndarray,
                    levels=(160, 320, 640), iters: int = 300,
                    eps: float = 1e-6) -> tuple[np.ndarray | None, float]:
    """Pyramid ECC alignment: coarse Euclidean, then homography.

    Direct/photometric rather than feature-based, so it aligns on smooth
    shading gradients -- exactly the signal that survives on a flat
    painted wall, where corner detectors find nothing at all.

    Measured on wall13, where SIFT and ORB together register 0 of 72
    image pairs: ECC recovers 0001->0002 at corr 0.955, 0006->0007 at
    0.901 and 0007->0008 at 0.925, and the warped cracks land on their
    counterparts. Caveat: ECC is driven by large-scale shading, so it
    aligns the SCENE well without necessarily being accurate to hairline
    precision -- treat its homographies as coarser than a RANSAC fit and
    set downstream tolerances from `inlier_rms` accordingly.
    """
    def prep(g, width):
        s = width / max(g.shape[:2])
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        g = cv2.createCLAHE(2.0, (8, 8)).apply(g)
        return cv2.GaussianBlur(g, (0, 0), 1.2).astype(np.float32) / 255.0, s

    W = np.eye(3, dtype=np.float32)
    prev_s, cc = None, 0.0
    for li, width in enumerate(levels):
        A, sa = prep(gray_a, width)
        B, _ = prep(gray_b, width)
        if prev_s is not None:                       # rescale warp to this level
            r = sa / prev_s
            S = np.diag([r, r, 1]).astype(np.float32)
            W = (S @ W @ np.linalg.inv(S)).astype(np.float32)
            W /= W[2, 2]
        mode = cv2.MOTION_EUCLIDEAN if li == 0 else cv2.MOTION_HOMOGRAPHY
        warp = W[:2].copy() if mode == cv2.MOTION_EUCLIDEAN else W.copy()
        try:
            cc, warp = cv2.findTransformECC(
                A, B, warp, mode,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps), None, 5)
        except cv2.error:
            return None, 0.0
        W = (np.vstack([warp, [0, 0, 1]]) if mode == cv2.MOTION_EUCLIDEAN
             else warp).astype(np.float32)
        prev_s = sa
    S = np.diag([prev_s, prev_s, 1.0])
    H = np.linalg.inv(S) @ W @ S
    return (H / H[2, 2]), float(cc)


def register_images(img_a: np.ndarray, img_b: np.ndarray,
                    mask_a: np.ndarray | None = None,
                    mask_b: np.ndarray | None = None,
                    exclude_cracks: bool = True,
                    exclude_dilate: int = 15,
                    max_dim: int = 1600,
                    ratio_thresh: float = 0.75,
                    ransac_thresh: float = 4.0,
                    min_inliers: int = 25,
                    ecc_fallback: bool = True,
                    ecc_min_corr: float = 0.80) -> Registration:
    """Estimate the homography mapping img_a into img_b's frame.

    Detection runs on downscaled copies for speed; the resulting
    homography is rescaled back to full resolution at the end.

    Three stages, tried in order, because this dataset contains two very
    different surfaces and one front-end cannot serve both:

      1. SIFT on wall texture, cracks masked out of detection.
      2. SIFT with the cracks put BACK IN, if step 1 found almost no
         keypoints. `exclude_cracks` is right for stippled render, where
         a long crack would otherwise dominate the correspondence set --
         but half these walls are smooth painted plaster photographed at
         cornice junctions, where the crack is the only feature in the
         frame and masking it leaves nothing at all.
      3. ECC, which needs no keypoints whatsoever.

    Measured across the 602 same-wall test pairs, stage 1 alone succeeds
    on 28%, and on walls 10/13/14/16/17 it succeeds on exactly none of
    330 pairs -- not marginally, but at zero, which is the signature of a
    correspondence problem rather than a geometry one.
    """
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY) if img_a.ndim == 3 else img_a
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY) if img_b.ndim == 3 else img_b

    scale_a = min(1.0, max_dim / max(gray_a.shape[:2]))
    scale_b = min(1.0, max_dim / max(gray_b.shape[:2]))

    small_a = cv2.resize(gray_a, None, fx=scale_a, fy=scale_a, interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(gray_b, None, fx=scale_b, fy=scale_b, interpolation=cv2.INTER_AREA)

    # CLAHE helps a lot when the two captures differ in exposure, which is
    # the normal case for inspections taken weeks or months apart.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    small_a = clahe.apply(small_a)
    small_b = clahe.apply(small_b)

    def det_masks(use_exclusion):
        if not use_exclusion:
            return None, None
        da = db = None
        if mask_a is not None:
            m = cv2.resize(mask_a, (small_a.shape[1], small_a.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
            da = _exclude_mask(m, exclude_dilate)
        if mask_b is not None:
            m = cv2.resize(mask_b, (small_b.shape[1], small_b.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
            db = _exclude_mask(m, exclude_dilate)
        return da, db

    sift = cv2.SIFT_create(nfeatures=8000)
    attempts = [True, False] if exclude_cracks else [False]
    kp_a = kp_b = des_a = des_b = None
    used_exclusion = exclude_cracks
    for use_exclusion in attempts:
        da, db = det_masks(use_exclusion)
        ka, dda = sift.detectAndCompute(small_a, da)
        kb, ddb = sift.detectAndCompute(small_b, db)
        kp_a, kp_b, des_a, des_b = ka, kb, dda, ddb
        used_exclusion = use_exclusion
        # ADAPTIVE: only retry unmasked when masking really did starve us.
        if min(len(ka or []), len(kb or [])) >= MIN_WALL_KEYPOINTS:
            break

    front_end = "sift" if used_exclusion else "sift-nocrackmask"

    def ecc_or_fail(reason: str, n_in=0, n_match=0, ratio=0.0) -> Registration:
        if not ecc_fallback:
            return Registration(None, n_in, n_match, ratio, False, reason, front_end)
        H_ecc, cc = _ecc_homography(gray_a, gray_b)
        if H_ecc is None or cc < ecc_min_corr:
            return Registration(None, n_in, n_match, ratio, False,
                                f"{reason}; ECC corr={cc:.2f}", front_end)
        sane, why = _homography_is_sane(H_ecc, gray_a.shape[:2])
        if not sane:
            return Registration(None, n_in, n_match, ratio, False,
                                f"{reason}; ECC {why}", front_end)
        return Registration(H_ecc, 0, 0, float(cc), True, f"ECC corr={cc:.2f}",
                            "ecc", float("nan"))

    if des_a is None or des_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        return ecc_or_fail("too few keypoints on the wall")

    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    knn = matcher.knnMatch(des_a, des_b, k=2)

    good = [m for pair in knn if len(pair) == 2
            for m, n in [pair] if m.distance < ratio_thresh * n.distance]

    if len(good) < min_inliers:
        return ecc_or_fail(f"only {len(good)} ratio-test matches", 0, len(good))

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    H_small, inliers = cv2.findHomography(pts_a, pts_b, method, ransac_thresh,
                                          maxIters=10000, confidence=0.9999)

    if H_small is None or inliers is None:
        return ecc_or_fail("RANSAC found no model", 0, len(good))

    n_in = int(inliers.sum())
    ratio = n_in / len(good)

    if n_in < min_inliers:
        return ecc_or_fail(f"only {n_in} geometric inliers", n_in, len(good), ratio)

    # Undo the downscaling: H_full = S_b^-1 @ H_small @ S_a
    S_a = np.diag([scale_a, scale_a, 1.0]).astype(np.float64)
    S_b_inv = np.diag([1.0 / scale_b, 1.0 / scale_b, 1.0]).astype(np.float64)
    H = S_b_inv @ H_small @ S_a
    H /= H[2, 2]

    sane, reason = _homography_is_sane(H, gray_a.shape[:2])
    if not sane:
        return ecc_or_fail(reason, n_in, len(good), ratio)

    # Inlier RMS in FULL-resolution px, so downstream shape tolerances can be
    # set from the fit's own accuracy rather than a hard-coded constant.
    sel = inliers.ravel().astype(bool)
    proj = cv2.perspectiveTransform(pts_a[sel], H_small).reshape(-1, 2)
    err = np.linalg.norm(proj - pts_b[sel].reshape(-1, 2), axis=1)
    rms = float(np.sqrt((err ** 2).mean()) / max(scale_b, 1e-9))

    return Registration(H, n_in, len(good), ratio, True, "", front_end, rms)


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

def _pair_window(pix_a: np.ndarray, pix_b: np.ndarray, shape: tuple[int, int],
                 margin: int) -> tuple[int, int, int, int]:
    """Union bounding box of two pixel sets, padded and clipped to the image."""
    h, w = shape
    x0 = max(int(min(pix_a[:, 0].min(), pix_b[:, 0].min())) - margin, 0)
    y0 = max(int(min(pix_a[:, 1].min(), pix_b[:, 1].min())) - margin, 0)
    x1 = min(int(max(pix_a[:, 0].max(), pix_b[:, 0].max())) + margin + 1, w)
    y1 = min(int(max(pix_a[:, 1].max(), pix_b[:, 1].max())) + margin + 1, h)
    return x0, y0, x1, y1


def symmetric_chamfer(inst_a: CrackInstance, inst_b: CrackInstance,
                      shape: tuple[int, int], cap: float = 1e4,
                      reject_gap: float = 150.0) -> float:
    """Mean symmetric distance (px) between two instances' pixel sets.

    Chamfer distance is the primary criterion rather than IoU because
    cracks are only a few pixels wide: a 3px registration error can drive
    IoU to nearly zero for a pair that is in fact correctly aligned.

    Two cost fixes, which together took this from 187 ms/pair to a few ms:

      * The distance transform is built over the UNION BOUNDING BOX of the
        two instances, not the whole photo. It used to allocate and fill a
        4080x3072 float array twice per pair; pixels far outside both
        cracks cannot affect either mean.

      * The cheap bbox rejection compared its gap against `cap` (1e4 px)
        on an image 4080 px tall, so it never once fired. `reject_gap` is
        a distance at which the pair is already hopeless, so the transform
        is skipped entirely.
    """
    h, w = shape
    pa, pb = inst_a.pixels, inst_b.pixels
    if len(pa) == 0 or len(pb) == 0:
        return cap

    # Cheap rejection: if bounding boxes are far apart, skip the transform.
    ax, ay, aw, ah = inst_a.bbox
    bx, by, bw, bh = inst_b.bbox
    gap_x = max(0, max(ax - (bx + bw), bx - (ax + aw)))
    gap_y = max(0, max(ay - (by + bh), by - (ay + ah)))
    gap = float(np.hypot(gap_x, gap_y))
    if gap > reject_gap:
        return min(gap, cap)

    x0, y0, x1, y1 = _pair_window(pa, pb, (h, w), margin=int(reject_gap) + 2)
    lw, lh = x1 - x0, y1 - y0
    if lw <= 0 or lh <= 0:
        return cap
    ax_, ay_ = pa[:, 0] - x0, pa[:, 1] - y0
    bx_, by_ = pb[:, 0] - x0, pb[:, 1] - y0

    def dt_of(xs, ys):
        occ = np.zeros((lh, lw), np.uint8)
        occ[ys, xs] = 1
        return cv2.distanceTransform(1 - occ, cv2.DIST_L2, 3)

    d_ab = dt_of(bx_, by_)[ay_, ax_].mean()
    d_ba = dt_of(ax_, ay_)[by_, bx_].mean()
    return float(min(0.5 * (d_ab + d_ba), cap))


def chamfer_coverage(inst_a: CrackInstance, inst_b: CrackInstance,
                     shape: tuple[int, int], tau: float = 6.0,
                     reject_gap: float = 150.0) -> float:
    """Scale-free agreement in [0, 1]: symmetric fraction of crack pixels
    lying within `tau` px of the other crack.

    Raw Chamfer is a distance in PIXELS, and every photo pair carries its
    own independently solved homography, so its scale differs per pair and
    per crack size -- a 3 px error on a 40 px crack and on a 400 px crack
    are not the same evidence. The ranking metrics never noticed, because
    they only compare within a row; DIR@FAR, pair-F1 and the assignment
    all apply ONE GLOBAL threshold and were measuring calibration rather
    than accuracy. reid_analysis patches that after the fact with row
    normalisation; this fixes it at the source and is bounded, so a
    threshold learned on val transfers to test.

    `tau` should be set from the registration's own inlier RMS for the
    pair when it is available -- see Registration.inlier_rms.
    """
    h, w = shape
    pa, pb = inst_a.pixels, inst_b.pixels
    if len(pa) == 0 or len(pb) == 0:
        return 0.0
    ax, ay, aw, ah = inst_a.bbox
    bx, by, bw, bh = inst_b.bbox
    gap_x = max(0, max(ax - (bx + bw), bx - (ax + aw)))
    gap_y = max(0, max(ay - (by + bh), by - (ay + ah)))
    if np.hypot(gap_x, gap_y) > reject_gap:
        return 0.0

    x0, y0, x1, y1 = _pair_window(pa, pb, (h, w), margin=int(reject_gap) + 2)
    lw, lh = x1 - x0, y1 - y0
    if lw <= 0 or lh <= 0:
        return 0.0
    ax_, ay_ = pa[:, 0] - x0, pa[:, 1] - y0
    bx_, by_ = pb[:, 0] - x0, pb[:, 1] - y0

    def dt_of(xs, ys):
        occ = np.zeros((lh, lw), np.uint8)
        occ[ys, xs] = 1
        return cv2.distanceTransform(1 - occ, cv2.DIST_L2, 3)

    cov_ab = float((dt_of(bx_, by_)[ay_, ax_] <= tau).mean())
    cov_ba = float((dt_of(ax_, ay_)[by_, bx_] <= tau).mean())
    return 0.5 * (cov_ab + cov_ba)


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
