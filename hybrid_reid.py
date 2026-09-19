"""
Hybrid defect re-identification.

Given a query image carrying one or more defects (each with a segmentation
mask) and a set of reference images whose defects carry unique IDs, decide
which query defect corresponds to which reference defect -- or reject the
match when no reference defect is sufficiently consistent.

The pipeline is deliberately modular.  Each stage is an independent,
tuneable unit:

  1. HOMOGRAPHY stage (Approach 1).  Full-image feature detection
     (crack pixels masked out so the wall texture drives the fit),
     ratio-test correspondences and RANSAC homography.  When the fit
     passes the quality gate the query masks are warped into the
     reference frame and compared geometrically: dilated IoU, Dice,
     Chamfer coverage, centroid displacement, area/perimeter and
     bounding-box ratios, contour similarity, orientation, and the
     fraction of the query that is NEW damage.

  2. SKELETON stage (Approach 2).  If the homography gate fails, each
     defect is reduced to a pruned morphological-skeleton graph
     (endpoints, junctions, branches with length/curvature/orientation,
     plus connectivity and width descriptors) and compared by a
     translation/rotation/scale-invariant structural score whose
     containment-style terms tolerate defect growth.

  3. DECISION stage.  A ranked candidate list per query defect, the
     method actually used, the quality indicators, a confidence value
     and a verdict (reliable / ambiguous / rejected) with an explicit
     rejection mechanism instead of forcing every query onto a gallery
     entry.

`HybridReIDScorer` adapts the same machinery to reid_eval's
full-image-scope scorer interface so the real benchmark and the
synthetic sweeps can run it unchanged.

Only cv2 / numpy are required for the whole pipeline.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

from crack_registration_reid import register_images, Registration
from crack_reid_baselines import skeletonize_mask


# ===========================================================================
# 1. Homography-based alignment (Approach 1)
# ===========================================================================

@dataclass
class HomographyConfig:
    """Quality gate and front-end knobs for the registration stage."""
    max_dim: int = 1600            # registration downscale target (long side)
    ratio_thresh: float = 0.75     # Lowe ratio test for SIFT matches
    ransac_thresh: float = 4.0     # RANSAC reprojection threshold (small px)
    min_inliers: int = 25
    ecc_fallback: bool = True      # photometric pyramid alignment as last resort
    ecc_min_corr: float = 0.80
    min_inlier_ratio: float = 0.30
    max_reproj_error_px: float = 8.0   # inlier RMS gate, full-resolution px


def estimate_homography(img_a: np.ndarray, mask_a: np.ndarray | None,
                        img_b: np.ndarray, mask_b: np.ndarray | None,
                        cfg: HomographyConfig | None = None) -> Registration:
    """Best-effort homography mapping image A into image B's frame.

    Wraps crack_registration_reid.register_images (SIFT on wall texture ->
    SIFT with cracks back in -> ECC) and returns the full Registration
    record so downstream quality indicators come from the fit itself.
    """
    cfg = cfg or HomographyConfig()
    return register_images(
        img_a, img_b, mask_a, mask_b,
        exclude_cracks=True,
        max_dim=cfg.max_dim,
        ratio_thresh=cfg.ratio_thresh,
        ransac_thresh=cfg.ransac_thresh,
        min_inliers=cfg.min_inliers,
        ecc_fallback=cfg.ecc_fallback,
        ecc_min_corr=cfg.ecc_min_corr,
    )


def homography_quality_ok(reg: Registration,
                          cfg: HomographyConfig | None = None) -> bool:
    """Whether a Registration meets the alignment quality gate.

    ECC registrations carry no keypoint statistics (inlier ratio is the
    correlation, inlier RMS is NaN), so they are gated on correlation;
    keypoint fits are gated on inlier count, ratio and reprojection RMS.
    """
    cfg = cfg or HomographyConfig()
    if not reg.ok:
        return False
    if reg.method == "ecc":
        return float(reg.inlier_ratio) >= cfg.ecc_min_corr
    if reg.n_inliers < cfg.min_inliers:
        return False
    if float(reg.inlier_ratio) < cfg.min_inlier_ratio:
        return False
    if np.isfinite(reg.inlier_rms) and reg.inlier_rms > cfg.max_reproj_error_px:
        return False
    return True


def warp_mask(mask: np.ndarray, H: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Binary mask warped by H into a frame of `shape` (h, w)."""
    h, w = shape
    return cv2.warpPerspective((mask > 0).astype(np.uint8), H, (w, h),
                               flags=cv2.INTER_NEAREST)


# ===========================================================================
# 2. Mask comparison (Approach 1, after alignment)
# ===========================================================================

def _union_bbox(mask_a: np.ndarray, mask_b: np.ndarray,
                margin: int) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask_a)
    if len(xs) == 0:
        ys, xs = np.nonzero(mask_b)
    if len(xs) == 0:
        return None
    x0 = max(int(xs.min()) - margin, 0)
    y0 = max(int(ys.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, mask_a.shape[1])
    y1 = min(int(ys.max()) + margin + 1, mask_a.shape[0])
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def _chamfer_coverage(ma: np.ndarray, mb: np.ndarray, margin: int = 12) -> float:
    """Symmetric fraction of pixels of each mask within margin px of the other."""
    box = _union_bbox(ma, mb, margin)
    if box is None:
        return 0.0
    x, y, w, h = box
    a = ma[y:y + h, x:x + w] > 0
    b = mb[y:y + h, x:x + w] > 0
    if not a.any() or not b.any():
        return 0.0

    def dt(m: np.ndarray) -> np.ndarray:
        return cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3)

    cab = float((dt(b)[a] <= margin).mean())
    cba = float((dt(a)[b] <= margin).mean())
    return 0.5 * (cab + cba)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.logical_and(a, b).sum())
    return 2.0 * inter / max(float(a.sum()) + float(b.sum()), 1e-9)


def _dilated_iou(a: np.ndarray, b: np.ndarray, dpx: int = 6) -> float:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dpx, dpx))
    ad = cv2.dilate(a.astype(np.uint8), k).astype(bool)
    bd = cv2.dilate(b.astype(np.uint8), k).astype(bool)
    return float(np.logical_and(ad, bd).sum()
                 / max(np.logical_or(ad, bd).sum(), 1))


def _principal_angle(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return 0.0
    pts = np.c_[xs, ys].astype(np.float64)
    c = pts - pts.mean(0)
    _, vals, vecs = np.linalg.svd(c.T @ c)
    return float(np.arctan2(vecs[0, 1], vecs[0, 0])) % np.pi


def _perimeter(mask: np.ndarray) -> float:
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    b = (mask > 0).astype(np.uint8)
    return float(max(cv2.dilate(b, k).sum() - cv2.erode(b, k).sum(), 1))


def _hu_similarity(mask: np.ndarray, other: np.ndarray) -> float:
    """Contour similarity from log-scaled Hu moments (1 = identical)."""
    def h(mask_):
        cnt, _ = cv2.findContours(mask_.astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        if not cnt:
            return np.zeros(7)
        mu = cv2.HuMoments(cv2.moments(max(cnt, key=cv2.contourArea))).flatten()
        with np.errstate(divide="ignore"):
            return np.where(mu > 0, -np.sign(mu) * np.log10(np.abs(mu) + 1e-12), mu)
    hm = h(mask)
    ho = h(other)
    if np.all(np.abs(hm) < 1e-9) or np.all(np.abs(ho) < 1e-9):
        return 0.5
    return float(np.exp(-0.5 * np.sum(np.abs(hm - ho) * [1, 1, 1, 1, 1, 1, 1])))


@dataclass
class MaskComparison:
    """Every geometric measurement of one aligned defect pair."""
    iou: float
    dice: float
    dilated_iou: float
    chamfer_coverage: float
    centroid_displacement: float           # px
    centroid_displacement_norm: float      # / defect RMS radius
    area_ratio: float                      # min/max
    perimeter_ratio: float
    bbox_iou: float
    orientation_diff_deg: float
    contour_similarity: float
    retention: float                       # overlap / reference area
    precision: float                       # overlap / query area
    new_damage_fraction: float             # 1 - precision
    growth_ratio: float                    # query area / reference area
    score: float                           # combined [0, 1]


def mask_alignment_comparison(ma_aligned: np.ndarray, mb_ref: np.ndarray,
                              chamfer_tau: float = 8.0) -> MaskComparison:
    """Compare an aligned (warped) query mask against a reference mask.

    Both masks are full-frame binaries sharing a coordinate system.
    Containment-oriented terms make the score tolerant of defect growth:
    retention (how much of the already-known defect is re-observed) is
    weighted above precision (how much of the new defect lies on the old
    footprint), so extending a crack does not destroy the match.
    """
    a = ma_aligned > 0
    b = mb_ref > 0
    inter = float(np.logical_and(a, b).sum())
    area_a, area_b = float(a.sum()), float(b.sum())
    if area_a == 0 and area_b == 0:
        empty = MaskComparison(0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0.0, 1.0, 1.0)
        empty.dice, empty.dilated_iou = 1.0, 1.0
        return empty

    retention = inter / max(area_b, 1.0)
    precision = inter / max(area_a, 1.0)
    overlap = 0.6 * retention + 0.4 * precision

    cc = _chamfer_coverage(a, b, margin=int(round(chamfer_tau)))

    ca = np.array(np.nonzero(a)).mean(1) if area_a else np.array([0.0, 0.0])
    cb = np.array(np.nonzero(b)).mean(1) if area_b else np.array([0.0, 0.0])
    disp = float(np.hypot(*((ca - cb) if ca.shape == cb.shape else [0, 0])))
    if area_a and area_b:
        pts = np.c_[np.nonzero(np.logical_or(a, b))[1], np.nonzero(np.logical_or(a, b))[0]]
        scale = float(np.sqrt(np.mean((pts - pts.mean(0)) ** 2))) if len(pts) > 1 else 1.0
    else:
        scale = 1.0
    ctr = float(np.exp(-disp / max(scale, 1.0)))

    area_ratio = min(area_a, area_b) / max(area_a, area_b, 1.0)
    perimeter_ratio = min(_perimeter(a), _perimeter(b)) / max(_perimeter(a), _perimeter(b))

    bx0 = np.array([float(a.argmax(1).min()) if a.any() else 0,
                    float(a.argmax(0).min()) if a.any() else 0])
    if a.any() and b.any():
        ny, nx = a.shape
        ax0a, ay0a = float(np.nonzero(a)[1].min()), float(np.nonzero(a)[0].min())
        ax1a, ay1a = float(np.nonzero(a)[1].max()), float(np.nonzero(a)[0].max())
        bx0b, by0b = float(np.nonzero(b)[1].min()), float(np.nonzero(b)[0].min())
        bx1b, by1b = float(np.nonzero(b)[1].max()), float(np.nonzero(b)[0].max())
        ix0, iy0 = max(ax0a, bx0b), max(ay0a, by0b)
        ix1, iy1 = min(ax1a, bx1b), min(ay1a, by1b)
        iw = max(0.0, ix1 - ix0)
        ih = max(0.0, iy1 - iy0)
        iarea = iw * ih
        uarea = ((ax1a - ax0a) * (ay1a - ay0a) + (bx1b - bx0b) * (by1b - by0b) - iarea)
        bbox_iou = iarea / max(uarea, 1.0)
    else:
        bbox_iou = 0.0

    ori_diff = abs(_principal_angle(a) - _principal_angle(b))
    ori_diff = min(ori_diff, np.pi - ori_diff)
    ori_term = float(np.exp(-np.degrees(ori_diff) / 25.0))

    hu = _hu_similarity(a, b)

    score = (0.55 * overlap
             + 0.25 * cc
             + 0.10 * ori_term
             + 0.05 * ctr
             + 0.05 * hu)
    score = float(np.clip(score, 0.0, 1.0))

    return MaskComparison(
        iou=inter / max(float(np.logical_or(a, b).sum()), 1.0),
        dice=_dice(a, b),
        dilated_iou=_dilated_iou(a, b),
        chamfer_coverage=cc,
        centroid_displacement=disp,
        centroid_displacement_norm=disp / max(scale, 1e-6),
        area_ratio=area_ratio,
        perimeter_ratio=perimeter_ratio,
        bbox_iou=bbox_iou,
        orientation_diff_deg=float(np.degrees(ori_diff)),
        contour_similarity=hu,
        retention=float(retention),
        precision=float(precision),
        new_damage_fraction=float(1.0 - precision),
        growth_ratio=area_a / max(area_b, 1.0),
        score=score,
    )


def appearance_similarity(img_a: np.ndarray, mask_a: np.ndarray,
                          img_b: np.ndarray, mask_b: np.ndarray) -> float | None:
    """Illumination-normalised texture agreement inside the defect regions.

    Returns None when either or both masks are empty or one of the images
    is missing.  The patches are zero-mean / unit-std normalised before the
    Pearson correlation so absolute brightness (which inspection photos
    never match across visits) does not dominate.
    """
    if img_a is None or img_b is None:
        return None
    ma, mb = mask_a > 0, mask_b > 0
    if not ma.any() or not mb.any():
        return None
    ga = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY) if img_a.ndim == 3 else img_a
    gb = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY) if img_b.ndim == 3 else img_b

    va = ga[ma].astype(np.float64)
    vb = gb[mb].astype(np.float64)
    if len(va) < 4 or len(vb) < 4:
        return None
    va = (va - va.mean()) / max(va.std(), 1e-6)
    vb = (vb - vb.mean()) / max(vb.std(), 1e-6)
    return float(np.clip(np.corrcoef(va, vb)[0, 1], -1.0, 1.0) * 0.5 + 0.5)


# ===========================================================================
# 3. Skeleton graph (Approach 2)
# ===========================================================================

@dataclass
class SkeletonConfig:
    """Tuning for skeleton extraction, pruning and the structural score."""
    prune_branch_frac: float = 0.08   # terminal branches shorter than this
                                      # fraction of the longest branch are hair
    min_branch_px: float = 3.0        # absolute pruning floor, skeleton px
    junction_tol: float = 4.0         # clustering radius for junction blobs
    shape_tau: float = 0.30           # containment radius, unit-RMS coords
    align_samples: int = 240          # per-set sample for the rotation search
    length_q: tuple = (0.25, 0.5, 0.75, 0.9)
    curvature_bins: int = 10
    angle_bins: int = 8


@dataclass
class SkeletonBranch:
    start: tuple[int, int]
    end: tuple[int, int]
    length_px: float
    detour: float            # path length / euclidean(start, end); 1.0 = straight
    mean_turn_deg: float
    orientation: float       # radians, mod pi


@dataclass
class SkeletonGraph:
    """Pruned morphological-skeleton graph of one defect mask.

    Nodes are endpoints (degree 1) and junction clusters (degree >= 3),
    edges are the skeleton chains between them.  Descriptors are computed
    once and reused across every comparison.
    """
    points: np.ndarray                 # (N, 2) raw skeleton pixels (x, y)
    points_norm: np.ndarray            # centred, divided by RMS radius
    scale: float                       # RMS radius of the skeleton points
    endpoints: np.ndarray              # (E, 2) normalised endpoint positions
    junctions: np.ndarray              # (J, 2) normalised junction positions
    branches: list[SkeletonBranch]
    degree_hist: np.ndarray            # normalised (deg 1, 2, 3, >=4)
    n_components: int
    cyclomatic: int                    # edges - nodes + components
    width_median: float                # distance-transform width / RMS radius
    curvature_hist: np.ndarray         # normalised detour-ratio histogram
    angle_hist: np.ndarray             # normalised pairwise branch-angle bins
    load_seconds: float = 0.0


def segment_defects(mask: np.ndarray, min_area: int = 200,
                    close_px: int = 5) -> list[np.ndarray]:
    """Split a combined binary mask into per-defect masks (full-frame)."""
    binary = (mask > 0).astype(np.uint8)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    return [(labels == i).astype(np.uint8)
            for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def _thin(img: np.ndarray) -> np.ndarray:
    """Skeletonise a binary mask to the one-pixel 8-connected medial line.

    Preferred backend is scikit-image's Zhang-Suen (it keeps straight
    diagonal runs to a single pixel, which a hand-rolled Zhang-Suen
    routinely leaves two pixels thick at ragged slab ends).  Falls back to
    the built-in _thin_zhang_suen when scikit-image is not importable.
    """
    try:
        from skimage.morphology import skeletonize
        return skeletonize((img > 0).astype(np.uint8) > 0).astype(np.uint8)
    except Exception:
        return _thin_zhang_suen(img)


def _thin_zhang_suen(img: np.ndarray) -> np.ndarray:
    """Parallel Zhang-Suen thinning (8-connected medial line).

    The morphological skeleton (a repeated erode / dilate difference) is
    brittle on diagonal crack runs: staircase pixels read as extra
    junctions, which pollutes the degree histogram and fragment count.
    Zhang-Suen sends diagonal runs down to a clean one-pixel 8-connected
    line, which is what the graph stage wants.
    """
    bin_ = (img > 0).astype(np.uint8)
    p = np.zeros((bin_.shape[0] + 2, bin_.shape[1] + 2), np.uint8)
    p[1:-1, 1:-1] = bin_
    changing = True
    while changing:
        changing = False
        for step in (1, 2):
            pts = np.argwhere(p == 1)
            to_delete = []
            for y, x in pts:
                if y == 0 or y == p.shape[0] - 1 or x == 0 or x == p.shape[1] - 1:
                    continue
                seq = [p[y - 1, x], p[y - 1, x + 1], p[y, x + 1], p[y + 1, x + 1],
                       p[y + 1, x], p[y + 1, x - 1], p[y, x - 1], p[y - 1, x - 1]]
                b = sum(seq)
                if not (2 <= b <= 6):
                    continue
                a = sum(1 for i in range(8) if seq[i] == 0 and seq[(i + 1) % 8] == 1)
                if a != 1:
                    continue
                p2, p4, p6, p8 = seq[0], seq[2], seq[4], seq[6]
                if step == 1:
                    ok = (p2 * p4 * p6 == 0) and (p4 * p6 * p8 == 0)
                else:
                    ok = (p2 * p4 * p8 == 0) and (p2 * p6 * p8 == 0)
                if ok:
                    to_delete.append((y, x))
            for (y, x) in to_delete:
                p[y, x] = 0
                changing = True
    return p[1:-1, 1:-1]


def _prune_terminal_hair(skel: np.ndarray, min_len: float) -> np.ndarray:
    """Remove short dead-end chains (spurious branches) from a skeleton."""
    work = skel.copy()
    h, w = work.shape

    def degrees():
        deg = cv2.filter2D(work, cv2.CV_16S, np.ones((3, 3), np.uint8)) - work
        return deg

    max_iter = 200
    for _ in range(max_iter):
        deg = degrees()
        eps = [tuple(pt) for pt in np.c_[np.nonzero((work > 0) & (deg == 1))[1],
                                         np.nonzero((work > 0) & (deg == 1))[0]]
               if deg[pt[1], pt[0]] == 1]
        removed_any = False
        for (ex, ey) in eps:
            chain = [(ex, ey)]
            prev = None
            px, py = ex, ey
            while True:
                nxt = []
                for ny in range(max(0, py - 1), min(h, py + 2)):
                    for nx in range(max(0, px - 1), min(w, px + 2)):
                        if (nx, ny) == (px, py) or (nx, ny) == prev:
                            continue
                        if work[ny, nx]:
                            nxt.append((nx, ny))
                if not nxt:
                    break
                if len(nxt) > 1:
                    break                         # reached a junction
                nx0, ny0 = nxt[0]
                if deg[ny0, nx0] >= 3:
                    chain.append((nx0, ny0))       # include the junction pixel
                    break
                chain.append((nx0, ny0))
                prev, px, py = (px, py), nx0, ny0
                if len(chain) > 100000:
                    break
            # A terminal chain must be removed either as pure hair (no
            # junction) or as a stub shorter than min_len.
            is_stub = chain and deg[chain[-1][1], chain[-1][0]] >= 3
            length = float(len(chain))
            if (is_stub and length < min_len) or (not is_stub and length <= 1):
                for (rx, ry) in chain:
                    if not is_stub or (rx, ry) != chain[-1]:
                        work[ry, rx] = 0
                removed_any = True
        if not removed_any:
            break
    return work


def _cluster_points(pts: np.ndarray, tol: float) -> np.ndarray:
    """Greedy nearest-neighbour clustering -> one centroid per cluster."""
    if not len(pts):
        return np.empty((0, 2))
    kept, taken = [], set()
    for i in range(len(pts)):
        if i in taken:
            continue
        d2 = np.sum((pts - pts[i]) ** 2, axis=1)
        cluster = np.nonzero(d2 <= tol ** 2)[0]
        taken.update(map(int, cluster))
        kept.append(pts[cluster].mean(axis=0))
    return np.array(kept, float).reshape(-1, 2)


def _extract_graph(skel: np.ndarray, cfg: SkeletonConfig):
    """Collapse a thinned skeleton into a defect graph.

    Junction blobs (contiguous degree >= 3 runs -- one physical
    bifurcation is usually a 2-4 pixel blob) become SINGLE junction nodes,
    so a thick rotated junction does not spawn dozens of spurious branches.
    Endpoints are degree-1 pixels (clustered within 2 px).  Branches are the
    skeleton chains between nodes; parallel probe chains through the same
    blob are deduplicated to the longest representative.
    """
    h, w = skel.shape
    ys, xs = np.nonzero(skel)
    pts = np.c_[xs, ys].astype(np.float64)
    index = {(int(x), int(y)): i for i, (x, y) in enumerate(zip(xs, ys))}

    degree = cv2.filter2D(skel, cv2.CV_16S, np.ones((3, 3), np.uint8)).astype(np.int16)
    degree -= skel.astype(np.int16)

    # --- Junction blobs ------------------------------------------------
    blob = ((skel > 0) & (degree >= 3)).astype(np.uint8)
    n_blobs, blob_labels, stats, centers = cv2.connectedComponentsWithStats(blob, 8)
    if n_blobs > 1:
        blob_centers = centers[1:].astype(np.float64)
        kept = _cluster_points(blob_centers, cfg.junction_tol)
        # nearest kept-centroid -> cluster id for every original blob label
        cluster_of_label = {}
        for lab in range(1, n_blobs):
            c = blob_centers[lab - 1]
            d = np.sum((kept - c) ** 2, axis=1)
            cluster_of_label[lab] = int(d.argmin())
        junction_coords = kept
    else:
        junction_coords = np.empty((0, 2))
        cluster_of_label = {}

    # --- Endpoint nodes ------------------------------------------------
    ep_idx = [i for i in range(len(xs)) if degree[ys[i], xs[i]] == 1
              and not blob[ys[i], xs[i]]]
    ep_coords = pts[ep_idx] if ep_idx else np.empty((0, 2))
    ep_coords = _cluster_points(ep_coords, 2.0)

    # --- Node bookkeeping ----------------------------------------------
    node_key_of: dict[int, tuple] = {}
    node_coord: dict[tuple, np.ndarray] = {}
    for (x, y), i in index.items():
        if blob[y, x]:
            k = cluster_of_label.get(int(blob_labels[y, x]), 0)
            node_key_of[i] = ("J", k)
    for i in ep_idx:
        pos = pts[i]
        # map the endpoint pixel to the nearest endpoint cluster
        if len(ep_coords):
            d = np.sum((ep_coords - pos) ** 2, axis=1)
            m = int(d.argmin())
            node_key_of[i] = ("E", m)
    for k, c in enumerate(junction_coords):
        node_coord[("J", k)] = c
    for m, c in enumerate(ep_coords):
        node_coord[("E", m)] = c

    # --- Neighbour lists -----------------------------------------------
    nbrs: dict[int, list[int]] = {}
    for (x, y), i in index.items():
        lst = []
        for ny in range(max(0, y - 1), min(h, y + 2)):
            for nx in range(max(0, x - 1), min(w, x + 2)):
                j = index.get((nx, ny))
                if j is not None and j != i:
                    lst.append(j)
        nbrs[i] = lst

    # --- Chain walking -------------------------------------------------
    best_edge: dict = {}          # canonical edge -> (SkeletonBranch, chain px)
    seen_dir: set = set()
    for i, key in node_key_of.items():
        for j in nbrs[i]:
            if (i, j) in seen_dir:
                continue
            chain_px = [i, j]
            prev, cur = i, j
            while cur not in node_key_of:
                nxt = [n for n in nbrs[cur] if n != prev]
                if len(nxt) != 1:
                    break
                prev, cur = cur, nxt[0]
                chain_px.append(cur)
                if len(chain_px) > len(index):
                    break
            for a, b in zip(chain_px, chain_px[1:]):
                seen_dir.add((a, b))
                seen_dir.add((b, a))

            end_key = node_key_of.get(cur)
            if end_key is None or end_key == key:
                continue                       # self loop around one blob
            start_pos = pts[chain_px[0]]
            end_pos = pts[chain_px[-1]]
            seg_len = float(np.linalg.norm(end_pos - start_pos))
            path = 0.0
            turns: list[float] = []
            for a, b in zip(chain_px, chain_px[1:]):
                path += float(np.linalg.norm(pts[a] - pts[b]))
            pv = None
            for k in range(0, len(chain_px) - 1, 2):
                v = pts[chain_px[k + 1]] - pts[chain_px[k]]
                if pv is not None and np.linalg.norm(pv) > 1e-9 and np.linalg.norm(v) > 1e-9:
                    cang = float(np.clip(np.dot(pv, v) / (np.linalg.norm(pv) * np.linalg.norm(v)), -1, 1))
                    turns.append(np.degrees(np.arccos(cang)))
                pv = v
            detour = path / max(seg_len, 1e-6)
            mean_turn = float(np.mean(turns)) if turns else 0.0
            orient = float(np.arctan2(end_pos[1] - start_pos[1], end_pos[0] - start_pos[0])) % np.pi
            br = SkeletonBranch(start=key, end=end_key, length_px=path,
                                detour=detour, mean_turn_deg=mean_turn,
                                orientation=orient)
            edge = tuple(sorted((key, end_key)))
            if edge not in best_edge or br.length_px > best_edge[edge][0].length_px:
                best_edge[edge] = (br, chain_px)

    # Junction hinge cleanup: a junction blob that collapses onto a pure
    # line (the staircase elbows of a wide diagonal stroke read as
    # degree-3 pixels) sits on exactly TWO branches.  Splice it away so
    # the arm is one straight chain again.
    while True:
        incidence: dict = {}
        for (a, b) in best_edge:
            incidence[a] = incidence.get(a, 0) + 1
            incidence[b] = incidence.get(b, 0) + 1
        hinge = next((n for n in incidence if n[0] == "J" and incidence[n] == 2), None)
        if hinge is None:
            break
        keep = {}
        for edge, (br, chain) in best_edge.items():
            if hinge in edge:
                continue
            keep[edge] = (br, chain)
        # merge the two branches that used to touch the hinge
        others = []
        for edge, (br, chain) in best_edge.items():
            if edge[0] == hinge:
                others.append((edge[1], br, chain))
            elif edge[1] == hinge:
                others.append((edge[0], br, chain))
        if len(others) == 2:
            (n1, b1, c1), (n2, b2, c2) = others
            path = b1.length_px + b2.length_px
            seg = float(np.linalg.norm(node_coord[n2] - node_coord[n1]))
            merged = SkeletonBranch(start=n1, end=n2, length_px=path,
                                    detour=path / max(seg, 1e-6),
                                    mean_turn_deg=float(np.mean([b1.mean_turn_deg, b2.mean_turn_deg])),
                                    orientation=float(
                                        np.arctan2(node_coord[n2][1] - node_coord[n1][1],
                                                   node_coord[n2][0] - node_coord[n1][0])) % np.pi)
            ne = tuple(sorted((n1, n2)))
            if ne not in keep or merged.length_px > keep[ne][0].length_px:
                keep[ne] = (merged, c1 + c2)
        best_edge = keep
    # a junction that lost all its branches is dropped by construction;
    # endpoints and junctions still referenced are kept in node_ids below.
    node_ids = [n for n in node_coord
                if n[0] == "E" or any(n in e for e in best_edge)]
    branches = sorted((e[0] for e in best_edge.values()), key=lambda b: b.length_px, reverse=True)
    return branches, node_ids, node_coord, pts, degree, index, ep_coords, junction_coords, best_edge


def skeleton_graph_from_mask(mask: np.ndarray,
                             cfg: SkeletonConfig | None = None) -> SkeletonGraph:
    """Build the pruned skeleton graph of one defect mask."""
    cfg = cfg or SkeletonConfig()
    t0 = time.time()
    skel0 = _thin((mask > 0).astype(np.uint8))
    if cv2.countNonZero(skel0) == 0:
        return SkeletonGraph(np.empty((0, 2)), np.empty((0, 2)), 1.0,
                             np.empty((0, 2)), np.empty((0, 2)), [],
                             np.zeros(4), 0, 0, 0.0, np.zeros(cfg.curvature_bins),
                             np.zeros(cfg.angle_bins), time.time() - t0)

    # Prune terminal hair so detractors and rasterisation stair-steps do
    # not inflate the degree histogram.
    branches0, _, _, _, _, _, _, _, _ = _extract_graph(skel0, cfg)
    longest = max((b.length_px for b in branches0), default=0.0)
    min_len = max(cfg.min_branch_px, cfg.prune_branch_frac * longest)
    skel1 = _prune_terminal_hair(skel0, min_len)

    branches, node_ids, node_coord, pts, degree, index, ep_coords, jun_coords, best_edge = \
        _extract_graph(skel1, cfg)
    points = pts
    centre = points.mean(axis=0)
    scale = float(np.sqrt(np.mean(np.sum((points - centre) ** 2, axis=1)))) if len(points) else 1.0
    s = max(scale, 1e-6)
    norm = (points - centre) / s

    endpoints = (ep_coords - centre) / s if len(ep_coords) else np.empty((0, 2))
    junctions = (jun_coords - centre) / s if len(jun_coords) else np.empty((0, 2))

    # Degree histogram over the CLEANED graph skeleton: the thick junction
    # blobs are collapsed to single pixels here, so the histogram reflects
    # topology rather than the antialiasing of one bifurcation.
    clean = np.zeros_like(skel1)
    ys1, xs1 = np.nonzero(skel1)
    for _, chain in best_edge.values():
        for p in chain:
            clean[ys1[p], xs1[p]] = 1
    for _, c in node_coord.items():
        px, py = int(round(c[0])), int(round(c[1]))
        if 0 <= py < skel1.shape[0] and 0 <= px < skel1.shape[1]:
            clean[py, px] = 1
    deg_clean = cv2.filter2D(clean, cv2.CV_16S, np.ones((3, 3), np.uint8)).astype(np.int16)
    deg_clean -= clean.astype(np.int16)
    deg_vals = deg_clean[clean > 0]
    hist = np.array([(deg_vals == 1).sum(), (deg_vals == 2).sum(),
                     (deg_vals == 3).sum(), (deg_vals >= 4).sum()],
                    dtype=float) if len(deg_vals) else np.zeros(4)
    hist = hist / max(hist.sum(), 1.0)

    # Connectivity: number of components and the cyclomatic number.
    n_labels, _ = cv2.connectedComponents(skel1, 8)
    n_comp = int(n_labels - 1)
    n_nodes = len(node_ids)
    n_edges = len(branches)
    cyclomatic = n_edges - n_nodes + n_comp if n_nodes else 0

    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    widths = dist[ys1, xs1].astype(float) / s if len(xs1) else np.array([])
    width_median = float(np.median(widths)) if len(widths) else 0.0

    # Curvature: detour-ratio histogram (1.0 straight -> 1.5 serpentine).
    detours = np.array([b.detour for b in branches], float) if branches else np.array([])
    detours = np.clip(detours, 1.0, 1.5)
    c_hist, _ = np.histogram(detours, bins=cfg.curvature_bins, range=(1.0, 1.5))
    c_hist = c_hist.astype(float) / max(len(detours), 1)

    # Angles between branches meeting at the same junction node.
    pair_angles: list[float] = []
    if branches:
        by_node: dict = {}
        for b in branches:
            by_node.setdefault(b.start, []).append(b)
            by_node.setdefault(b.end, []).append(b)
        for node, blist in by_node.items():
            if node[0] != "J" or len(blist) < 2:
                continue
            ors = [b.orientation for b in blist]
            for i in range(len(ors)):
                for j in range(i + 1, len(ors)):
                    d = abs(ors[i] - ors[j]) % np.pi
                    pair_angles.append(min(d, np.pi - d))
    pair_angles = np.array(pair_angles, float) if pair_angles else np.array([])
    if len(pair_angles):
        pair_angles = np.clip(pair_angles, 1e-6, np.pi / 2)
        a_hist, _ = np.histogram(pair_angles, bins=cfg.angle_bins, range=(0, np.pi / 2))
        a_hist = a_hist.astype(float) / max(len(pair_angles), 1)
    else:
        a_hist = np.zeros(cfg.angle_bins)

    return SkeletonGraph(points, norm, scale, endpoints, junctions, branches,
                         hist, n_comp, max(cyclomatic, 0), width_median,
                         c_hist, a_hist, time.time() - t0)


# ===========================================================================
# 4. Structural similarity (Approach 2)
# ===========================================================================

def _best_rigid_alignment(pa: np.ndarray, pb: np.ndarray,
                          n_sample: int = 240) -> tuple[np.ndarray, float]:
    """Best rotation (about the normalised centroids) aligning a to b.

    Returns (R, symmetric chamfer).  Starts on a 6-deg coarse grid over the
    full circle WITH and WITHOUT a reflection (a mirrored crack is a
    different defect, but principal-axis sign flips make both orientations
    reachable), then refines the best candidates on a fine grid.
    """
    if len(pa) == 0 or len(pb) == 0:
        return np.eye(2), np.inf
    step_a = max(1, len(pa) // n_sample)
    a = pa[::step_a][:n_sample]
    b = pb[::max(1, len(pb) // n_sample)][:n_sample]

    def chamfer(R):
        q = a @ R.T
        d2 = np.sum((q[:, None, :] - b[None, :, :]) ** 2, axis=2)
        return float((np.sqrt(d2).min(1).mean() + np.sqrt(d2).min(0).mean()) / 2)

    def R_of(theta, mirror):
        c, s = np.cos(theta), np.sin(theta)
        M = np.array([[c, -s], [s, c]])
        if mirror:
            M = M @ np.array([[1.0, 0.0], [0.0, -1.0]])
        return M

    coarse = []
    for mirror in (False, True):
        for theta in np.linspace(0, 2 * np.pi, 60, endpoint=False):
            coarse.append((chamfer(R_of(theta, mirror)), theta, mirror))
    coarse.sort(key=lambda t: t[0])

    # Seed with the actual coarse winner: np.eye(2) would silently win on
    # flat surfaces (straight lines) where refinement never strictly improves.
    best = (R_of(coarse[0][1], coarse[0][2]), coarse[0][0],
            coarse[0][1], coarse[0][2])
    # A reflected matrix's base angle is extracted through its dihedral form.
    for c0, theta0, mirror0 in coarse[:6]:
        if mirror0:
            # R = [[c,-s],[s,c]] @ diag(1,-1) = [[c,s],[s,-c]] -> base arg
            base = -theta0
        else:
            base = theta0
        for dtheta in np.linspace(-6.0, 6.0, 25):
            th = base + np.radians(dtheta)
            cc = chamfer(R_of(th, mirror0))
            if cc < best[1]:
                best = (R_of(th, mirror0), cc, th, mirror0)
    return best[0], best[1]


def _directed_coverage(a: np.ndarray, b: np.ndarray, tol: float) -> float:
    """Fraction of points of a within tol of some point of b."""
    if len(a) == 0 and len(b) == 0:
        return 1.0                       # both structures lack this feature set
    if len(b) == 0:
        return 0.0
    if len(a) == 0:
        return 1.0
    d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    return float((np.sqrt(d2).min(1) <= tol).mean())


def _hist_intersection(ha: np.ndarray, hb: np.ndarray) -> float:
    if float(ha.sum()) == 0 and float(hb.sum()) == 0:
        return 1.0                       # both histograms empty -> identical
    return float(np.minimum(ha, hb).sum())


@dataclass
class StructuralScore:
    score: float
    shape: float
    endpoints: float
    junctions: float
    topology: float
    branch_lengths: float
    curvature: float
    angles: float
    chamfer: float
    n_components_a: int
    n_components_b: int
    n_branches_a: int
    n_branches_b: int
    n_endpoints_a: int
    n_endpoints_b: int
    n_junctions_a: int
    n_junctions_b: int


def structural_similarity(a: SkeletonGraph, b: SkeletonGraph,
                          cfg: SkeletonConfig | None = None,
                          weights: tuple = (0.30, 0.15, 0.15, 0.12, 0.10, 0.10, 0.08)
                          ) -> StructuralScore:
    """Translation/rotation/scale-invariant structural similarity of two
    skeleton graphs, in [0, 1].

    The (larger, grown) query is placed as the second argument: the shape
    and landmark terms are containment-weighted so that NEW branches and
    extensions in the query cost much less than an old landmark
    disappearing.  Topology terms use histogram intersection, which again
    only mildly penalises extra structure.
    """
    cfg = cfg or SkeletonConfig()
    w = np.asarray(weights, float)
    w = w / w.sum()

    if len(a.points) < 2 or len(b.points) < 2:
        empty = StructuralScore(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                np.inf, a.n_components, b.n_components,
                                len(a.branches), len(b.branches),
                                len(a.endpoints), len(b.endpoints),
                                len(a.junctions), len(b.junctions))
        if len(a.points) == len(b.points) == 0:
            empty.score, empty.shape, empty.topology = 1.0, 1.0, 1.0
        return empty

    R, chamfer = _best_rigid_alignment(a.points_norm, b.points_norm, cfg.align_samples)

    # Containment shape: the reference's pattern must survive in the query;
    # extra query material is tolerated.
    cov_ab = _directed_coverage(a.points_norm @ R.T, b.points_norm, cfg.shape_tau)
    cov_ba = _directed_coverage(b.points_norm, a.points_norm @ R.T, cfg.shape_tau)
    shape = 0.65 * cov_ab + 0.35 * cov_ba

    ep_a = _directed_coverage(a.endpoints @ R.T, b.endpoints, 0.35)
    ep_b = _directed_coverage(b.endpoints, a.endpoints @ R.T, 0.35)
    endpoints = 0.7 * ep_a + 0.3 * ep_b

    jn_a = _directed_coverage(a.junctions @ R.T, b.junctions, 0.40)
    jn_b = _directed_coverage(b.junctions, a.junctions @ R.T, 0.40)
    junctions = 0.6 * jn_a + 0.4 * jn_b

    topo_deg = _hist_intersection(a.degree_hist, b.degree_hist)
    dc = min(abs(a.n_components - b.n_components) + abs(a.cyclomatic - b.cyclomatic), 4)
    conn = float(np.exp(-dc / 3.0))
    topology = 0.7 * topo_deg + 0.3 * conn

    if len(a.branches) and len(b.branches):
        la = np.quantile([br.length_px for br in a.branches], list(cfg.length_q))
        lb = np.quantile([br.length_px for br in b.branches], list(cfg.length_q))
        la = la / max(la.mean(), 1e-6)
        lb = lb / max(lb.mean(), 1e-6)
        branch_lengths = float(np.exp(-np.abs(la - lb).mean() / 0.45))
    else:
        branch_lengths = 1.0 if len(a.branches) == len(b.branches) else 0.0

    curvature = _hist_intersection(a.curvature_hist, b.curvature_hist)
    angles = _hist_intersection(a.angle_hist, b.angle_hist)

    terms = np.clip(np.array([shape, endpoints, junctions, topology,
                              branch_lengths, curvature, angles]), 0.0, 1.0)
    score = float(np.dot(w, terms))

    return StructuralScore(score, *map(float, terms), float(chamfer),
                           a.n_components, b.n_components,
                           len(a.branches), len(b.branches),
                           len(a.endpoints), len(b.endpoints),
                           len(a.junctions), len(b.junctions))


# ===========================================================================
# 5. Decision strategy
# ===========================================================================

@dataclass
class DecisionConfig:
    accept_threshold: float = 0.40     # below this the top candidate is rejected
    margin_threshold: float = 0.05     # top-2 gap needed to call it reliable
    align_weight: float = 0.55         # alignment weight in the HYBRID blend
    blend_align: bool = True           # combine alignment + skeleton when H is good
    appearance: bool = False           # include the local-texture term


@dataclass
class HybridConfig:
    homography: HomographyConfig = field(default_factory=HomographyConfig)
    skeleton: SkeletonConfig = field(default_factory=SkeletonConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    chamfer_tau_px: float = 8.0
    min_defect_area: int = 200
    close_px: int = 5


@dataclass
class Candidate:
    """One ranked per-query-defect matching hypothesis."""
    reference_id: str
    score: float
    method: str                     # "HOMOGRAPHY" | "SKELETON" | "HYBRID"
    confidence: float
    verdict: str                    # "reliable" | "ambiguous" | "rejected"
    quality: dict


@dataclass
class Reference:
    """One reference image and its defects (each with a unique id)."""
    id: str
    image: np.ndarray
    mask: np.ndarray                       # combined full-frame binary mask
    image_id: str = ""                     # optional stable key for caching


class HybridReID:
    """End-to-end hybrid defect re-identification engine.

    `match` implements the sequential decision strategy described in the
    objective:

        * estimate the homography between the query and every reference;
        * if it meets the quality gate, use the alignment-based mask
          comparison (Optionally blended with the structural score ->
          HYBRID);
        * otherwise fall back to the skeleton comparison (SKELETON);
        * rank the candidates for every query defect and attach a
          reliable / ambiguous / rejected verdict.
    """

    def __init__(self, config: HybridConfig | None = None):
        self.config = config or HybridConfig()
        self._h_cache: dict = {}

    def _homography(self, img_a, mask_a, img_b, mask_b) -> Registration:
        cfg = self.config.homography
        return estimate_homography(img_a, mask_a, img_b, mask_b, cfg)

    def _skeleton(self, mask: np.ndarray) -> SkeletonGraph:
        return skeleton_graph_from_mask(mask, self.config.skeleton)

    # -- per-cell scoring --------------------------------------------------
    def _cell(self, align: MaskComparison | None,
              struct: StructuralScore,
              reg: Registration | None) -> Candidate:
        """Assemble one per-cell score and its quality record."""
        cfg = self.config
        dec = cfg.decision
        align_good = reg is not None and homography_quality_ok(reg, cfg.homography)

        if align_good and dec.blend_align:
            s = dec.align_weight * align.score + (1.0 - dec.align_weight) * struct.score
            method = "HYBRID"
        elif align_good:
            s = align.score
            method = "HOMOGRAPHY"
        else:
            s = struct.score
            method = "SKELETON"
        s = float(np.clip(s, 0.0, 1.0))

        if align_good:
            rms_term = float(np.exp(-min(reg.inlier_rms, 30.0) / 12.0)) \
                if np.isfinite(reg.inlier_rms) else 0.0
            confidence = float(np.clip(0.35 * min(reg.inlier_ratio, 1.0)
                                       + 0.35 * rms_term
                                       + 0.30 * align.score, 0.0, 1.0))
        else:
            confidence = float(struct.score)

        quality = {
            "method": method,
            "n_matches": reg.n_matches if reg else 0,
            "n_inliers": reg.n_inliers if reg else 0,
            "inlier_ratio": float(reg.inlier_ratio) if reg else 0.0,
            "inlier_rms_px": float(reg.inlier_rms) if reg and np.isfinite(reg.inlier_rms) else None,
            "registration_reason": reg.reason if reg and not reg.ok else "",
            "homography_ok": bool(align_good),
            "align_score": float(align.score) if align else None,
            "structural_score": float(struct.score),
            "iou": align.iou if align else None,
            "dice": align.dice if align else None,
            "dilated_iou": align.dilated_iou if align else None,
            "chamfer_coverage": align.chamfer_coverage if align else None,
            "centroid_displacement_px": align.centroid_displacement if align else None,
            "area_ratio": align.area_ratio if align else None,
            "perimeter_ratio": align.perimeter_ratio if align else None,
            "bbox_iou": align.bbox_iou if align else None,
            "orientation_diff_deg": align.orientation_diff_deg if align else None,
            "contour_similarity": align.contour_similarity if align else None,
            "new_damage_fraction": align.new_damage_fraction if align else None,
            "growth_ratio": align.growth_ratio if align else None,
            "skeleton_shape": float(struct.shape),
            "skeleton_endpoints": float(struct.endpoints),
            "skeleton_junctions": float(struct.junctions),
            "skeleton_topology": float(struct.topology),
            "skeleton_lengths": float(struct.branch_lengths),
            "skeleton_curvature": float(struct.curvature),
            "skeleton_angles": float(struct.angles),
            "endpoints_ref": struct.n_endpoints_a,
            "endpoints_query": struct.n_endpoints_b,
            "junctions_ref": struct.n_junctions_a,
            "junctions_query": struct.n_junctions_b,
            "branches_ref": struct.n_branches_a,
            "branches_query": struct.n_branches_b,
            "components_ref": struct.n_components_a,
            "components_query": struct.n_components_b,
        }
        return Candidate(reference_id="", score=s, method=method,
                         confidence=confidence, verdict="", quality=quality)

    # -- top-level matching ------------------------------------------------
    def match(self, query_image: np.ndarray,
              query_masks: np.ndarray | Iterable[np.ndarray],
              references: list[Reference]) -> list[list[Candidate]]:
        """Rank the reference defects for every query defect.

        Returns one list (ranked descending) per query defect.  Every
        candidate carries the reference defect id, method, confidence,
        quality indicators and a verdict.
        """
        cfg = self.config
        if isinstance(query_masks, np.ndarray) and query_masks.ndim == 2:
            q_defects = segment_defects(query_masks, cfg.min_defect_area, cfg.close_px)
        else:
            q_defects = [np.asarray(m) for m in query_masks]
            q_defects = [m for m in q_defects if (m > 0).any()]

        # Reference defects, each carrying the caller's id.
        ref_defects: list[tuple[str, Reference, np.ndarray]] = []
        for ref in references:
            if getattr(ref, "defect_masks", None):
                masks = list(ref.defect_masks)
            else:
                masks = segment_defects(ref.mask, cfg.min_defect_area, cfg.close_px)
            if masks:
                for m in masks:
                    ref_defects.append((ref.id, ref, m))
            else:
                ref_defects.append((ref.id, ref, ref.mask))

        # Precompute skeleton graphs once per defect.
        q_graphs = [self._skeleton(m) for m in q_defects]
        r_graphs = [self._skeleton(m) for _, _, m in ref_defects]

        full_q = np.logical_or.reduce([m > 0 for m in q_defects]) if q_defects else \
            np.zeros(query_masks.shape[:2] if hasattr(query_masks, "shape") else (0, 0), bool)
        query_shape = (query_image.shape[0], query_image.shape[1])

        # Register once per reference image.
        regs: list[Registration] = []
        for ref in references:
            reg = self._homography(query_image, full_q.astype(np.uint8),
                                   ref.image, (ref.mask > 0).astype(np.uint8))
            regs.append(reg)

        # Group reference defects by their image so the warp is computed once.
        by_img: dict[int, list[tuple[int, str, np.ndarray]]] = {}
        for ri, (rid, _ref, rm) in enumerate(ref_defects):
            by_img.setdefault(id(_ref), []).append((ri, rid, rm))

        results: list[list[Candidate]] = []
        for qi, qm in enumerate(q_defects):
            cands: list[Candidate] = []
            ri = 0
            for ref in references:
                reg = regs[ri]
                ri += 1
                ref_shape = (ref.image.shape[0], ref.image.shape[1])
                mem = by_img[id(ref)]
                for (_ri, rid, rm) in mem:
                    sel = reg
                    align = None
                    if homography_quality_ok(sel, cfg.homography):
                        wm = warp_mask(qm, sel.H, ref_shape)
                        align = mask_alignment_comparison(
                            wm, rm, chamfer_tau=cfg.chamfer_tau_px)
                    struct = structural_similarity(q_graphs[qi], r_graphs[_ri], cfg.skeleton)
                    c = self._cell(align, struct, sel)
                    c.reference_id = f"{rid}" if len(mem) == 1 else f"{rid}#{_ri + 1}"
                    cands.append(c)
            cands.sort(key=lambda c: c.score, reverse=True)
            results.append(self._verdicts(cands))
        return results

    def _verdicts(self, cands: list[Candidate]) -> list[Candidate]:
        dec = self.config.decision
        if not cands:
            return cands
        top = cands[0].score
        second = cands[1].score if len(cands) > 1 else -np.inf
        for c in cands:
            if top < dec.accept_threshold:
                c.verdict = "rejected"
            elif top - second >= dec.margin_threshold:
                c.verdict = "reliable"
            else:
                c.verdict = "ambiguous"
        return cands


# ===========================================================================
# 6. reid_eval adapter (full-image scope) for the benchmark sweeps
# ===========================================================================

class HybridReIDScorer:
    """reid_eval.ReIDScorer adapter for the hybrid pipeline.

    Scores every (query instance, gallery instance) cell:

      * homography between the two full images -> alignment-based mask
        comparison, optionally blended with the structural score (HYBRID);
      * no usable homography -> skeleton structural score (SKELETON).

    Homographies and skeleton graphs are cached so a Q x G sweep costs one
    registration per image pair plus one structural comparison per cell.
    Diagnostics are accumulated so the evaluation report can state which
    method handled which pairs.
    """

    name = "Hybrid"
    input_scope = "full-image"

    def __init__(self, config: HybridConfig | None = None,
                 name: str | None = None, prune: bool = True):
        self.hybrid = HybridReID(config)
        self.name = name or self.hybrid.__class__.__name__
        self.prune = prune
        self._skel: dict = {}
        self._H: dict = {}
        self._diag: list[dict] = []

    # -- instance-level geometry ------------------------------------------
    def _full_mask(self, ref, data) -> np.ndarray:
        inst = data.instance_of(ref)
        img = data.image(ref.image_id)
        full = np.zeros(img.shape[:2], np.uint8)
        x, y, w, h = inst.bbox
        x, y = int(x), int(y)
        m = inst.mask_crop
        full[y:y + m.shape[0], x:x + m.shape[1]] = m
        return full

    def prepare(self, refs, data) -> None:
        todo = [r for r in refs if r.instance_id not in self._skel]
        if not todo:
            return
        for r in todo:
            m = self._full_mask(r, data)
            self._skel[r.instance_id] = skeleton_graph_from_mask(
                m, self.hybrid.config.skeleton)

    def _registration(self, q_img_id: str, g_img_id: str, data) -> Registration:
        key = (q_img_id, g_img_id)
        if key not in self._H:
            qm = (data.mask(q_img_id) > 0).astype(np.uint8)
            gm = (data.mask(g_img_id) > 0).astype(np.uint8)
            self._H[key] = estimate_homography(
                data.image(q_img_id), qm, data.image(g_img_id), gm,
                self.hybrid.config.homography)
        return self._H[key]

    def score_matrix(self, queries, gallery, data) -> np.ndarray:
        S = np.full((len(queries), len(gallery)), -np.inf, dtype=float)

        if self.prune:
            from benchmark import valid_mask
            valid = valid_mask(queries, gallery)
        else:
            valid = np.ones((len(queries), len(gallery)), bool)

        by_pair: dict = {}
        for i, q in enumerate(queries):
            for j in np.nonzero(valid[i])[0]:
                by_pair.setdefault((q.image_id, gallery[j].image_id),
                                   []).append((i, j))

        qmasks = [self._full_mask(q, data) for q in queries]
        gmasks = [self._full_mask(g, data) for g in gallery]

        for (qid, gid), cells in by_pair.items():
            reg = self._registration(qid, gid, data)
            align_good = homography_quality_ok(reg, self.hybrid.config.homography)
            g_shape = data.image(gid).shape[:2]
            if align_good:
                warped = {qi: warp_mask(qmasks[qi], reg.H, g_shape)
                          for qi in {c[0] for c in cells}}
            for i, j in cells:
                qmask, gmask = qmasks[i], gmasks[j]
                struct = structural_similarity(
                    self._skel[queries[i].instance_id],
                    self._skel[gallery[j].instance_id],
                    self.hybrid.config.skeleton)
                align = None
                wm = warped[i] if align_good else None
                if align_good:
                    align = mask_alignment_comparison(wm, gmask,
                                                      chamfer_tau=self.hybrid.config.chamfer_tau_px)
                c = self.hybrid._cell(align, struct, reg)
                S[i, j] = c.score
                self._diag.append({
                    "query": queries[i].instance_id, "gallery": gallery[j].instance_id,
                    "query_image": qid, "gallery_image": gid,
                    "method": c.method, "score": c.score,
                    "confidence": c.confidence,
                    "homography_ok": c.quality["homography_ok"],
                    "n_inliers": c.quality["n_inliers"],
                    "inlier_ratio": c.quality["inlier_ratio"],
                    "align_score": c.quality["align_score"],
                    "structural_score": c.quality["structural_score"],
                    "registration_reason": c.quality["registration_reason"],
                })
        return S

    def diagnostics(self) -> dict:
        """Usage statistics over every scored cell of the last matrix."""
        n = len(self._diag)
        if not n:
            return {"cells": 0}
        from collections import Counter
        methods = Counter(d["method"] for d in self._diag)
        confs = np.array([d["confidence"] for d in self._diag], float)
        ok = Counter(bool(d["homography_ok"]) for d in self._diag)
        return {
            "cells": n,
            "method_counts": dict(methods),
            "method_fracs": {k: v / n for k, v in methods.items()},
            "homography_ok_cells": ok[True],
            "homography_failed_cells": ok[False],
            "confidence_mean": float(confs.mean()),
            "confidence_hist": np.histogram(confs, bins=5, range=(0, 1))[0].tolist(),
        }