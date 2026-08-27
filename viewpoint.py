"""
A viewpoint covariate that does not depend on the method being evaluated.

WHY THIS EXISTS
---------------
The benchmark claims invariance to viewpoint. To evidence that claim you
have to stratify performance BY viewpoint change, which needs a per-pair
number saying how much the camera moved. Two candidates were considered
and one of them was wrong for two turns of review:

  FRAME GAP -- rejected. Frame index measures elapsed capture order, and
    over the pairs where geometry can be measured it tracks only one of
    the three viewpoint components: in-plane rotation moves with it while
    scale ratio and perspective tilt do not. `report()` below prints the
    three rank correlations measured on the dataset at hand -- run it and
    quote those, rather than any figure carried over from a review.

    So a frame-gap sweep is a CAMERA-ROLL sweep. Roll is the least
    interesting viewpoint component here -- an in-plane rotation that any
    rotation-invariant descriptor handles and that a homography absorbs
    exactly -- so sweeping it and reporting flat performance would look
    like a demonstration of viewpoint invariance while demonstrating
    almost nothing. `reid_analysis.py --frame-gap-sweep` is kept, but as
    a NEAR-DUPLICATE CONTROL over capture separation, not as the
    viewpoint axis.

  THE METHOD'S OWN HOMOGRAPHY -- rejected as circular. Scale and tilt are
    only measurable where the homography succeeds, so stratifying on it
    means stratifying on a covariate that exists precisely where the
    method already works.

This module supplies the third option: estimate the pairwise geometry
with front ends the SCORING pipeline does not use, so the covariate
exists on pairs registration cannot handle, and report honestly how often
it does.

WHAT "INDEPENDENT" MEANS HERE, PRECISELY
----------------------------------------
crack_registration_reid.register_images scores with SIFT keypoints
detected on wall texture with the cracks MASKED OUT, and falls back to
ECC. The covariate here is estimated by:

    akaze : AKAZE keypoints on the full image with NOTHING masked out --
            a different detector, a different descriptor, and a different
            input, so its failures are not the scoring front end's
            failures.
    ecc   : dense photometric alignment, no keypoints at all.

ECC is shared with the scoring pipeline's third stage, so this is
independence from the front end that produces the scores rather than from
every line of code the method contains. `scan()` reports both the
cross-estimator agreement (do AKAZE and ECC agree where both succeed?)
and the coverage gain (how many pairs get a covariate that registration
could not score), which are the two numbers that decide whether the
stratification is usable. If the coverage gain is small, the covariate is
not solving the circularity and the paper needs the controlled re-capture
instead -- re-shooting a subset of walls at MARKED distances and angles,
so the viewpoint covariate is known by construction rather than
estimated.

DECOMPOSITION
-------------
A homography is decomposed at the image centre into the components a
reviewer will ask about separately:

    scale       isotropic scale change, sqrt(det J), reported unordered
                as max(s, 1/s) so a pair has one value
    rotation    in-plane roll in degrees (the component frame gap tracks)
    tilt        out-of-plane tilt implied by foreshortening,
                arccos(s2/s1) in degrees -- intrinsics-free
    projectivity how much the local scale varies across the frame, a
                unitless measure of how projective (vs affine) the
                transform is

    python viewpoint.py dataset --out benchmark_out          # full scan
    python viewpoint.py dataset --out benchmark_out --split test
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import time

import cv2
import numpy as np


# ===========================================================================
# Decomposition
# ===========================================================================

def _jacobian(H: np.ndarray, x: float, y: float) -> np.ndarray | None:
    """2x2 local linearisation of the projective map at (x, y)."""
    p = np.array([x, y, 1.0])
    den = float(H[2] @ p)
    if abs(den) < 1e-9:
        return None
    num = H[:2] @ p
    J = (H[:2, :2] * den - np.outer(num, H[2, :2])) / (den ** 2)
    return J if np.all(np.isfinite(J)) else None


def decompose(H: np.ndarray, shape: tuple[int, int]) -> dict | None:
    """Viewpoint components of H at the centre of an image of `shape`.

    Everything here is intrinsics-free. A physical tilt angle would need
    the focal length, and these are two handsets whose EXIF this dataset
    does not carry; foreshortening (the ratio of the two singular values
    of the local Jacobian) measures the same thing up to that unknown and
    cannot be inflated by guessing a focal length wrong.
    """
    h, w = shape[:2]
    J = _jacobian(H, w / 2.0, h / 2.0)
    if J is None:
        return None
    s = np.linalg.svd(J, compute_uv=False)
    s1, s2 = float(max(s)), float(min(s))
    if s2 <= 1e-9:
        return None
    det = float(np.linalg.det(J))
    if det <= 0:
        return None                                  # mirrored: not a viewpoint change
    scale = float(np.sqrt(det))

    U, _, Vt = np.linalg.svd(J)
    R = U @ Vt
    rot = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

    aniso = s1 / s2
    tilt = float(np.degrees(np.arccos(min(1.0, 1.0 / aniso))))

    # How much the local scale varies across the frame: 1.0 for an affine
    # map, larger as the transform becomes genuinely projective.
    corners = [(0.15 * w, 0.15 * h), (0.85 * w, 0.15 * h),
               (0.15 * w, 0.85 * h), (0.85 * w, 0.85 * h)]
    scales = []
    for cx, cy in corners:
        Jc = _jacobian(H, cx, cy)
        if Jc is None:
            continue
        d = float(np.linalg.det(Jc))
        if d > 0:
            scales.append(np.sqrt(d))
    projectivity = float(max(scales) / min(scales)) if len(scales) >= 2 else float("nan")

    return {
        "scale": scale,                      # directional a -> b
        "scale_change": float(max(scale, 1.0 / scale)),   # unordered
        "rotation_deg": rot,
        "abs_rotation_deg": abs(rot),
        "tilt_deg": tilt,
        "anisotropy": float(aniso),
        "projectivity": projectivity,
    }


# ===========================================================================
# Independent front ends
# ===========================================================================

def _keypoint_detector():
    """AKAZE where the build has it, ORB otherwise.

    Both are non-SIFT and binary-descriptored, so either is a genuinely
    different front end from the scoring pipeline's masked SIFT. AKAZE is
    preferred because its nonlinear scale space holds up better on the
    low-contrast plaster; opencv-python 5.x minimal builds ship without
    it, so the covariate must not depend on its presence. The name is
    recorded per pair so a reviewer can see which build measured what.
    """
    if hasattr(cv2, "AKAZE_create"):
        return cv2.AKAZE_create(), "akaze"
    return cv2.ORB_create(4000), "orb"


def _akaze_homography(gray_a: np.ndarray, gray_b: np.ndarray,
                      max_dim: int = 1600, min_inliers: int = 20
                      ) -> tuple[np.ndarray | None, float]:
    """AKAZE (or ORB) on the FULL image, nothing masked out.

    Deliberately not the scoring pipeline's detector, descriptor or input:
    it keys on the crack as readily as on the wall, which is the right
    thing for MEASURING geometry and the wrong thing for scoring identity
    (a crack that dominates the correspondence set biases the homography
    towards aligning cracks, which would beg the question the benchmark
    asks). Returns (H, confidence) with confidence = inlier ratio.
    """
    from crack_registration_reid import _homography_is_sane

    sa = min(1.0, max_dim / max(gray_a.shape[:2]))
    sb = min(1.0, max_dim / max(gray_b.shape[:2]))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    A = clahe.apply(cv2.resize(gray_a, None, fx=sa, fy=sa, interpolation=cv2.INTER_AREA))
    B = clahe.apply(cv2.resize(gray_b, None, fx=sb, fy=sb, interpolation=cv2.INTER_AREA))

    ak, _ = _keypoint_detector()
    ka, da = ak.detectAndCompute(A, None)
    kb, db = ak.detectAndCompute(B, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None, 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = [m for pr in bf.knnMatch(da, db, k=2) if len(pr) == 2
            for m, n in [pr] if m.distance < 0.75 * n.distance]
    if len(good) < min_inliers:
        return None, 0.0
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    Hs, inl = cv2.findHomography(src, dst, method, 4.0, maxIters=10000, confidence=0.9999)
    if Hs is None or inl is None or int(inl.sum()) < min_inliers:
        return None, 0.0
    H = np.diag([1.0 / sb, 1.0 / sb, 1.0]) @ Hs @ np.diag([sa, sa, 1.0])
    H /= H[2, 2]
    sane, _ = _homography_is_sane(H, gray_a.shape[:2])
    if not sane:
        return None, 0.0
    return H, float(inl.sum()) / len(good)


#: Minimum RANSAC inlier ratio to accept a keypoint estimate, per detector.
#: ORB's binary descriptors match far less cleanly than AKAZE's, so holding
#: both to one number either rejects most good ORB fits or accepts bad AKAZE
#: ones. Both are backed by min_inliers=20 and the corner sanity check, so
#: this gate only has to exclude fits that are mostly outliers.
KP_MIN_CONF = {"akaze": 0.35, "orb": 0.15}


def estimate(gray_a: np.ndarray, gray_b: np.ndarray,
             ecc_min_corr: float = 0.55, kp_min_conf: float | None = None) -> dict:
    """Both independent estimates plus the chosen one.

    ecc_min_corr is looser than the scoring pipeline's 0.80 on purpose: a
    coarse alignment that is good enough to say "this pair is a 1.6x scale
    change" is not good enough to Chamfer-match hairlines, and holding a
    measurement to a matching tolerance is what shrinks the covariate back
    onto the pairs the method already handles.
    """
    from crack_registration_reid import _ecc_homography, _homography_is_sane

    kp_name = _keypoint_detector()[1]
    thr = KP_MIN_CONF.get(kp_name, 0.35) if kp_min_conf is None else kp_min_conf
    out = {"estimators": {}}
    Ha, ca = _akaze_homography(gray_a, gray_b)
    if Ha is not None and ca >= thr:
        d = decompose(Ha, gray_a.shape[:2])
        if d:
            out["estimators"][kp_name] = {**d, "confidence": ca}

    He, ce = _ecc_homography(gray_a, gray_b)
    # The keypoint path sanity-checks its homography inside
    # _akaze_homography; the ECC path had no such check, and a diverged
    # alignment decomposes into a plausible-looking 55 deg tilt rather than
    # into an obvious failure. Same corner test, same threshold.
    if He is not None and ce >= ecc_min_corr and _homography_is_sane(He, gray_a.shape[:2])[0]:
        d = decompose(He, gray_a.shape[:2])
        if d:
            out["estimators"]["ecc"] = {**d, "confidence": ce}

    # AKAZE first when it clears its gate: a RANSAC fit on real
    # correspondences is metrically tighter than a photometric alignment
    # driven by large-scale shading.
    for name in (kp_name, "ecc"):
        if name in out["estimators"]:
            out.update({k: v for k, v in out["estimators"][name].items()})
            out["estimator"] = name
            out["ok"] = True
            return out
    out["ok"] = False
    out["estimator"] = None
    return out


# ===========================================================================
# Pair scan
# ===========================================================================

def _manifest(root: str) -> list[dict]:
    with open(os.path.join(root, "walls.csv")) as f:
        return list(csv.DictReader(f))


def _frame(image_id: str) -> int:
    tail = str(image_id).rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else -1


def scan(root: str, out_dir: str, split: str | None = "test",
         with_registration: bool = True, limit: int | None = None,
         resume: bool = True) -> dict:
    """Every same-wall photo pair, measured three ways.

    Writes <out_dir>/viewpoint.json holding, per pair:
      * the independent viewpoint covariate and which front end produced it
      * whether the SCORING pipeline registered that pair, and with which
        of its three stages -- this is the pair-level outcome that
        image_quality.py's gate sweep needs and that splits `scored` into
        its real components
      * both images' sharpness and the frame gap

    One pass because all three want the same decoded image pairs, and
    decoding 12 MP JPEGs dominates the cost.
    """
    from image_quality import sharpness_table

    rows = _manifest(root)
    splits_path = os.path.join(root, "splits.json")
    splits = json.load(open(splits_path)) if os.path.exists(splits_path) else {}
    keep = set(splits.get(split, [])) if split else None
    rows = [r for r in rows if keep is None or r["wall_id"] in keep]

    sharp = sharpness_table(root, quiet=True)
    by_wall: dict[str, list[dict]] = {}
    for r in rows:
        by_wall.setdefault(r["wall_id"], []).append(r)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "viewpoint.json")
    have = {}
    if resume and os.path.exists(path):
        have = json.load(open(path)).get("pairs", {})

    pairs = [(a, b) for w in sorted(by_wall)
             for a, b in itertools.combinations(sorted(by_wall[w], key=lambda r: r["image_id"]), 2)]
    if limit:
        pairs = pairs[:limit]

    # Two entries, evicting the oldest: pairs are enumerated per wall in
    # image order, so consecutive pairs share an image and a two-slot cache
    # halves the JPEG decoding. A one-slot cache (clear-then-insert) evicts
    # the image the very next call wants and decodes every 12 MP frame twice.
    gray_cache: dict[str, np.ndarray] = {}

    def gray(r):
        iid = r["image_id"]
        if iid not in gray_cache:
            while len(gray_cache) >= 2:
                gray_cache.pop(next(iter(gray_cache)))
            gray_cache[iid] = cv2.imread(os.path.join(root, r["path"]),
                                         cv2.IMREAD_GRAYSCALE)
        return gray_cache[iid]

    todo = [(a, b) for a, b in pairs if f'{a["image_id"]}|{b["image_id"]}' not in have]
    print(f"{len(pairs)} same-wall pairs in split={split!r}; {len(todo)} to measure")
    t0 = time.time()
    for n, (ra, rb) in enumerate(todo, 1):
        key = f'{ra["image_id"]}|{rb["image_id"]}'
        ga, gb = gray(ra), gray(rb)
        rec = {
            "wall": ra["wall_id"],
            "frame_gap": abs(_frame(ra["image_id"]) - _frame(rb["image_id"])),
            "sharp_a": sharp.get(ra["image_id"], float("nan")),
            "sharp_b": sharp.get(rb["image_id"], float("nan")),
        }
        rec["sharp_min"] = min(rec["sharp_a"], rec["sharp_b"])

        vp = estimate(ga, gb)
        rec["viewpoint"] = {k: v for k, v in vp.items() if k != "estimators"}
        rec["viewpoint_by_estimator"] = vp["estimators"]

        if with_registration:
            from crack_registration_reid import register_images
            ma = cv2.imread(os.path.join(root, "masks", f'{ra["image_id"]}.png'), 0)
            mb = cv2.imread(os.path.join(root, "masks", f'{rb["image_id"]}.png'), 0)
            # register_images greyscales a 3-channel input as its first act,
            # so handing it the grey image is identical and skips a
            # 12 MP grey->BGR->grey round trip per pair.
            reg = register_images(ga, gb, ma, mb)
            rec["registration"] = {
                "ok": bool(reg.ok),
                "front_end": reg.method,
                "n_inliers": int(reg.n_inliers),
                "inlier_ratio": float(reg.inlier_ratio),
                "reason": reg.reason[:120],
            }
        have[key] = rec

        if n % 5 == 0 or n == len(todo):
            el = time.time() - t0
            print(f"\r  {n}/{len(todo)} pairs  {el/60:.1f}m elapsed, "
                  f"{el/n*(len(todo)-n)/60:.1f}m left   ", end="", flush=True)
            with open(path, "w") as f:
                json.dump({"split": split, "pairs": have}, f)
    print()
    with open(path, "w") as f:
        json.dump({"split": split, "pairs": have}, f)

    # pair_outcomes.json: the flat form image_quality.py's sweep reads
    outcomes = {k: v["registration"]["ok"] for k, v in have.items() if "registration" in v}
    if outcomes:
        with open(os.path.join(out_dir, "pair_outcomes.json"), "w") as f:
            json.dump({"split": split, "registered": outcomes}, f, indent=1)
    print(f"wrote {path}" + (f" and {out_dir}/pair_outcomes.json" if outcomes else ""))
    return have


# ===========================================================================
# Reporting
# ===========================================================================

def load(out_dir: str) -> dict:
    path = os.path.join(out_dir, "viewpoint.json")
    if not os.path.exists(path):
        raise SystemExit(f"no {path}; run `python viewpoint.py <root> --out {out_dir}` first")
    return json.load(open(path))["pairs"]


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4 or x[m].std() == 0 or y[m].std() == 0:
        return float("nan"), float("nan")
    from scipy import stats
    r = stats.spearmanr(x[m], y[m])
    return float(r.statistic), float(r.pvalue)


def _logit_fit(X: np.ndarray, y: np.ndarray, names: list[str]) -> list[dict]:
    """Joint logistic fit with Wald z from the observed information.

    A joint fit, not two separate correlations: the whole question in S1 is
    whether sharpness predicts registration failure ONCE ELAPSED BASELINE IS
    CONTROLLED FOR, and a pair of marginal correlations cannot answer that.
    Predictors are standardised so the coefficients are comparable to each
    other; log1p is applied to sharpness first because it spans two orders
    of magnitude and a linear term would be driven entirely by wall06.

    Hand-rolled because statsmodels is not a dependency of this repo and
    this is thirty lines: Newton-free BFGS on the log-likelihood, standard
    errors from the inverse Hessian.
    """
    from scipy import optimize, stats

    n, k = X.shape
    Xs = np.column_stack([np.ones(n), (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)])

    def nll(b):
        z = np.clip(Xs @ b, -30, 30)
        return float(np.sum(np.log1p(np.exp(z)) - y * z))

    res = optimize.minimize(nll, np.zeros(k + 1), method="BFGS")
    b = res.x
    p = 1 / (1 + np.exp(-np.clip(Xs @ b, -30, 30)))
    W = p * (1 - p)
    info = Xs.T @ (Xs * W[:, None])
    try:
        se = np.sqrt(np.diag(np.linalg.inv(info)))
    except np.linalg.LinAlgError:
        se = np.full(k + 1, np.nan)
    out = []
    for i, name in enumerate(names, start=1):
        z = b[i] / se[i] if np.isfinite(se[i]) and se[i] > 0 else float("nan")
        out.append({"name": name, "coef": float(b[i]), "z": float(z),
                    "p": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan")})
    return out


def report(pairs: dict, gate: float | None = None) -> str:
    ok = [p for p in pairs.values() if p["viewpoint"].get("ok")]
    reg = [p for p in pairs.values() if p.get("registration", {}).get("ok")]
    L = ["", "=" * 78, "VIEWPOINT COVARIATE", "=" * 78,
         f"pairs scanned                     {len(pairs)}",
         f"covariate estimated               {len(ok)} ({len(ok)/max(len(pairs),1):.0%})",
         f"registered by the scoring pipeline{len(reg):>4} ({len(reg)/max(len(pairs),1):.0%})"]

    gain = [p for p in pairs.values()
            if p["viewpoint"].get("ok") and not p.get("registration", {}).get("ok")]
    L.append(f"covariate WHERE REGISTRATION FAILED {len(gain)} "
             f"-- the whole point: {len(gain)} pairs that a method-dependent")
    L.append("    stratifier could not have covered")

    from collections import Counter
    L.append(f"  front end used: {dict(Counter(p['viewpoint'].get('estimator') for p in ok))}")

    # cross-estimator agreement
    kp = next((k for p in pairs.values()
               for k in p.get("viewpoint_by_estimator", {}) if k != "ecc"), "akaze")
    both = [p for p in pairs.values()
            if {kp, "ecc"} <= set(p.get("viewpoint_by_estimator", {}))]
    if both:
        ds = [abs(np.log(p["viewpoint_by_estimator"][kp]["scale"]) -
                  np.log(p["viewpoint_by_estimator"]["ecc"]["scale"])) for p in both]
        dr = [abs(p["viewpoint_by_estimator"][kp]["rotation_deg"] -
                  p["viewpoint_by_estimator"]["ecc"]["rotation_deg"]) for p in both]
        L += ["", f"CROSS-ESTIMATOR AGREEMENT ({kp} vs ecc) on the {len(both)} pairs both solved",
              f"  |log scale| difference   median {np.median(ds):.3f}  "
              f"(= {100*(np.exp(np.median(ds))-1):.1f}% of scale)",
              f"  rotation difference      median {np.median(dr):.2f} deg"]
    else:
        L += ["", "CROSS-ESTIMATOR AGREEMENT: no pair solved by both front ends"]

    if ok:
        L += ["", "VIEWPOINT ENVELOPE over pairs with a covariate "
                  "(the numbers the claim should be confined to)",
              f"{'component':<22}{'median':>9}{'p90':>9}{'max':>9}"]
        for key, name in (("scale_change", "scale change (x)"),
                          ("abs_rotation_deg", "in-plane rotation (deg)"),
                          ("tilt_deg", "out-of-plane tilt (deg)"),
                          ("projectivity", "projectivity")):
            v = np.array([p["viewpoint"][key] for p in ok if np.isfinite(p["viewpoint"].get(key, np.nan))])
            if len(v):
                L.append(f"{name:<22}{np.median(v):>9.2f}{np.percentile(v,90):>9.2f}{v.max():>9.2f}")

    # the S2 refutation, recomputed from this scan
    if ok:
        L += ["", "IS FRAME GAP A VIEWPOINT PROXY?  (the claim this module retracts)",
              f"{'frame gap vs':<26}{'rho':>8}{'p':>10}   usable as a viewpoint stratifier"]
        for key, name in (("abs_rotation_deg", "in-plane rotation"),
                          ("scale_change", "scale change"),
                          ("tilt_deg", "out-of-plane tilt")):
            r, p = _spearman([q["frame_gap"] for q in ok],
                             [q["viewpoint"][key] for q in ok])
            verdict = "yes" if (np.isfinite(p) and p < 0.05) else "NO"
            L.append(f"{name:<26}{r:>8.3f}{p:>10.3g}   {verdict}")

    # the S1 evidence, recomputed from this scan
    if reg or True:
        L += ["", "WHAT PREDICTS A REGISTRATION FAILURE",
              f"{'predictor':<26}{'median(reg)':>13}{'median(fail)':>14}{'z':>8}{'p':>10}"]
        good = [p for p in pairs.values() if p.get("registration", {}).get("ok")]
        bad = [p for p in pairs.values() if "registration" in p and not p["registration"]["ok"]]
        have = [p for p in pairs.values() if "registration" in p]
        fits = []
        if len(have) > 10 and good and bad:
            X = np.column_stack([np.log1p([p["sharp_min"] for p in have]),
                                 [p["frame_gap"] for p in have]])
            y = np.array([float(p["registration"]["ok"]) for p in have])
            m = np.all(np.isfinite(X), 1)
            if m.sum() > 10 and 0 < y[m].sum() < m.sum():
                fits = _logit_fit(X[m], y[m], ["sharp_min", "frame_gap"])
        by_name = {f["name"]: f for f in fits}
        for key, name in (("sharp_min", "min sharpness of pair"),
                          ("frame_gap", "frame gap")):
            gv = [p[key] for p in good]
            bv = [p[key] for p in bad]
            f = by_name.get(key, {})
            z, pv = f.get("z", float("nan")), f.get("p", float("nan"))
            L.append(f"{name:<26}{np.median(gv) if gv else float('nan'):>13.1f}"
                     f"{np.median(bv) if bv else float('nan'):>14.1f}{z:>8.2f}{pv:>10.3g}")
        L.append("z and p are Wald statistics from ONE joint logistic fit of "
                 "registration success on")
        L.append("both predictors, standardised (sharpness as log1p). Joint, because the "
                 "question is")
        L.append("whether sharpness still predicts failure once elapsed baseline is "
                 "controlled for.")

    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="method-independent viewpoint covariate")
    ap.add_argument("root")
    ap.add_argument("--out", default="benchmark_out")
    ap.add_argument("--split", default="test", help="'test', 'val' or 'all'")
    ap.add_argument("--limit", type=int, default=None, help="first N pairs only (smoke test)")
    ap.add_argument("--no-registration", action="store_true",
                    help="skip the scoring pipeline's outcome per pair (much faster)")
    ap.add_argument("--report-only", action="store_true",
                    help="re-print the report from a saved viewpoint.json")
    args = ap.parse_args()

    split = None if args.split == "all" else args.split
    if args.report_only:
        pairs = load(args.out)
    else:
        pairs = scan(args.root, args.out, split=split,
                     with_registration=not args.no_registration, limit=args.limit)
    print(report(pairs))
