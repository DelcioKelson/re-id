"""
Unified baseline harness for wall-crack re-identification.

Every method — classical keypoints (SIFT/ORB), learned matchers
(SuperGlue, LoFTR), and global embeddings (DeiT, ViT, CLIP, OSNet, YOLO
backbone) — is wrapped behind one interface:

    matcher.match(instances_a, instances_b) -> Assignment

so the whole results table comes from a single loop over a registry,
not seven hand-run scripts. Two things the old per-script code got
wrong are fixed structurally here:

  * Descriptors / embeddings are computed ONCE per instance in
    `prepare()` and reused across the N*M pairwise comparisons, instead
    of re-running detectAndCompute (or a forward pass) inside every
    pair. For a gallery of any size this is the difference between
    O(N+M) and O(N*M) feature extractions.

  * The accept/reject decision lives in ONE place — the shared
    Hungarian assignment with a per-method threshold — rather than in a
    per-method `is_crack_match` with the unreachable `if n < 4` branch
    after an early return. There is no dead code to keep in sync.

Heavy backbones are lazy-imported inside each matcher, so this module
imports and runs the classical methods even in an environment with no
torch / kornia / torchreid / transformers / ultralytics installed. A
method whose dependency is missing is skipped with a note, not a crash.

    pip install opencv-contrib-python numpy scipy        # always
    pip install torch timm kornia torchreid transformers ultralytics  # as needed
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# ===========================================================================
# Shared instance extraction  (was copy-pasted into all seven scripts)
# ===========================================================================

@dataclass
class CrackInstance:
    crop: np.ndarray          # cropped BGR image
    mask_crop: np.ndarray     # cropped binary mask, same HxW as crop
    bbox: tuple               # (x, y, w, h) in original-image coords


def extract_crack_instances(image: np.ndarray, mask: np.ndarray,
                            min_area: int = 200, pad: int = 10,
                            close_px: int = 5,
                            apply_mask: bool = False) -> list[CrackInstance]:
    """Connected-component instances from a binary crack mask.

    close_px:   morphological closing before labelling, so a crack that
                breaks into several components in one capture stays a
                single instance (instance counts must be comparable
                between the two images being matched).
    apply_mask: zero out background outside the crack pixels. Keypoint
                methods want this True (stops a big crack's crop visually
                containing neighbouring cracks); embedding backbones
                trained on natural images often prefer False. The harness
                sets this per-method, not globally.
    """
    binary = (mask > 0).astype(np.uint8)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    h_img, w_img = binary.shape[:2]

    instances = []
    for lid in range(1, n_labels):
        if stats[lid, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y, w, h = stats[lid, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT + 4]
        x0, y0 = max(int(x) - pad, 0), max(int(y) - pad, 0)
        x1, y1 = min(int(x) + int(w) + pad, w_img), min(int(y) + int(h) + pad, h_img)

        crop_raw = image[y0:y1, x0:x1].copy()
        inst_mask = (labels[y0:y1, x0:x1] == lid).astype(np.uint8) * 255
        crop = cv2.bitwise_and(crop_raw, crop_raw, mask=inst_mask) if apply_mask else crop_raw

        instances.append(CrackInstance(crop=crop, mask_crop=inst_mask,
                                       bbox=(x0, y0, x1 - x0, y1 - y0)))
    return instances


# ===========================================================================
# Assignment + shared Hungarian solver
# ===========================================================================

@dataclass
class Assignment:
    method: str
    pairs: list                       # (idx_a, idx_b, score)
    unmatched_a: list                 # queries with no accepted match
    unmatched_b: list                 # gallery entries not claimed
    score_matrix: np.ndarray          # (N_a, N_b), higher = more similar
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {p[0]: p[1] for p in self.pairs}


def solve_assignment(score_matrix: np.ndarray, min_score: float,
                     method: str = "") -> Assignment:
    """Global one-to-one matching from a similarity matrix.

    Solving globally (not argmax-per-query) stops two queries claiming
    the same gallery crack. Scores below `min_score` are rejected AFTER
    the assignment, which yields the open-set outputs the paper needs:
    unmatched_a = 'not re-observed', unmatched_b = 'newly appeared'.
    """
    n_a, n_b = score_matrix.shape
    if n_a == 0 or n_b == 0:
        return Assignment(method, [], list(range(n_a)), list(range(n_b)), score_matrix)

    # Hungarian minimises cost; we maximise similarity -> negate.
    rows, cols = linear_sum_assignment(-score_matrix)

    pairs, matched_a, matched_b = [], set(), set()
    for i, j in zip(rows, cols):
        s = float(score_matrix[i, j])
        if s >= min_score:
            pairs.append((int(i), int(j), s))
            matched_a.add(int(i))
            matched_b.add(int(j))

    return Assignment(
        method=method,
        pairs=pairs,
        unmatched_a=[i for i in range(n_a) if i not in matched_a],
        unmatched_b=[j for j in range(n_b) if j not in matched_b],
        score_matrix=score_matrix,
    )


# ===========================================================================
# Base matcher: prepare-once, score-matrix, assign
# ===========================================================================

class BaseMatcher(ABC):
    """Interface every baseline implements.

    Subclasses provide `prepare` (per-instance features, computed once)
    and either `score_pair` (default O(N*M) loop) or an overridden
    `build_score_matrix` (embedding methods vectorise this to a matmul).
    `min_score` is the single accept/reject knob.
    """

    name: str = "base"
    min_score: float = 0.0
    apply_mask: bool = False          # passed through to extract_crack_instances

    @abstractmethod
    def prepare(self, instances: list[CrackInstance]) -> list[Any]:
        """Return one cached representation per instance. Called once per
        image, not once per pair — this is where the O(N*M) recompute is
        eliminated."""

    def score_pair(self, prep_a: Any, prep_b: Any) -> float:
        raise NotImplementedError

    def build_score_matrix(self, preps_a: list[Any], preps_b: list[Any]) -> np.ndarray:
        S = np.zeros((len(preps_a), len(preps_b)), dtype=np.float64)
        for i, a in enumerate(preps_a):
            for j, b in enumerate(preps_b):
                S[i, j] = self.score_pair(a, b)
        return S

    def match(self, instances_a: list[CrackInstance],
              instances_b: list[CrackInstance]) -> Assignment:
        t0 = time.time()
        preps_a = self.prepare(instances_a)
        preps_b = self.prepare(instances_b)
        S = self.build_score_matrix(preps_a, preps_b)
        assignment = solve_assignment(S, self.min_score, method=self.name)
        assignment.seconds = time.time() - t0
        return assignment


# ---------------------------------------------------------------------------
# Family 1: classical keypoint matching (SIFT / ORB)
# ---------------------------------------------------------------------------

def _tiebreak(*qualities: float) -> float:
    """Fold continuous quality signals into a fraction strictly below 1.

    Every keypoint and learned-pairwise matcher here scores in RANSAC
    inlier COUNTS -- small integers, so a query's row is mostly exact
    ties. Measured on walls 07/08/11/12: 94% of ORB's valid pairs score
    exactly 0.0, and the median query has 30 gallery entries tied for top
    place. reid_eval now charges ties their expected rank rather than
    their list position, which stops the ordering being an artefact, but
    a tie the matcher could have broken is still information thrown away.

    Adding a sub-integer quality term keeps the inlier count as the
    primary criterion (the published, interpretable number) while letting
    genuinely better-supported matches win among equals. Capped just
    under 1.0 so it can never promote a pair past one with a higher
    inlier count.
    """
    q = [float(x) for x in qualities if x is not None and np.isfinite(x)]
    if not q:
        return 0.0
    return float(min(sum(q) / len(q), 1.0)) * 0.999


def _match_margin(pairs, ratio: float) -> float:
    """Mean Lowe-ratio margin over knn pairs, in [0, 1).

    How decisively the best match beat the runner-up, averaged. Near 0
    means every match was ambiguous; near 1 means they were distinctive.
    Defined even when no match survives the ratio test, which is exactly
    the case (score 0) that needs separating.
    """
    if not pairs:
        return 0.0
    m = np.array([p[0].distance for p in pairs], dtype=float)
    n = np.array([p[1].distance for p in pairs], dtype=float)
    return float(np.clip(1.0 - m / np.maximum(n, 1e-6), 0.0, 1.0).mean())


class KeypointMatcher(BaseMatcher):
    """SIFT or ORB keypoints + ratio test + RANSAC inlier count as score.

    Faithful reproduction of the original baseline (kept as-is for the
    comparison table), with two fixes: descriptors are detected once per
    instance in `prepare`, and the score is simply the inlier count — the
    shared solver does the thresholding, so there is no separate
    is_crack_match with an unreachable branch.

    (The homography fit on near-collinear crack points is statistically
    fragile; that weakness is exactly what the registration-based method
    is meant to beat, so it's left intact here rather than patched.)
    """

    apply_mask = True

    def __init__(self, method: str = "sift", ratio: float = 0.75,
                 ransac_thresh: float = 5.0, min_inliers: int = 8):
        self.method = method
        self.ratio = ratio
        self.ransac_thresh = ransac_thresh
        self.min_inliers = min_inliers
        self.min_score = float(min_inliers)
        self.name = method.upper()

        if method == "sift":
            self.detector = cv2.SIFT_create()
            self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
            self._norm_float = True
        elif method == "orb":
            self.detector = cv2.ORB_create(nfeatures=2000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            self._norm_float = False
        else:
            raise ValueError("method must be 'sift' or 'orb'")

    def prepare(self, instances):
        out = []
        for inst in instances:
            gray = cv2.cvtColor(inst.crop, cv2.COLOR_BGR2GRAY) if inst.crop.ndim == 3 else inst.crop
            kp, des = self.detector.detectAndCompute(gray, None)
            if des is not None and not self._norm_float:
                des = des.astype(np.uint8)
            out.append((kp, des))
        return out

    def score_pair(self, prep_a, prep_b) -> float:
        kp_a, des_a = prep_a
        kp_b, des_b = prep_b
        if des_a is None or des_b is None or len(kp_a) < 2 or len(kp_b) < 2:
            return 0.0

        knn = self.matcher.knnMatch(des_a, des_b, k=2)
        pairs = [pair for pair in knn if len(pair) == 2]
        good = [m for m, n in pairs if m.distance < self.ratio * n.distance]
        margin = _match_margin(pairs, self.ratio)
        if len(good) < 4:
            return len(good) + margin

        pts_a = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_b = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, inliers = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, self.ransac_thresh)
        if inliers is None:
            return margin
        n_in = int(inliers.sum())
        return n_in + _tiebreak(n_in / len(good), margin)


# ---------------------------------------------------------------------------
# Family 2: global embeddings + cosine similarity
# ---------------------------------------------------------------------------

def letterbox(crop: np.ndarray, size: int, pad_value: int = 114) -> np.ndarray:
    """Resize to a square keeping aspect ratio, padding the short side.

    The crops here are long and thin -- median aspect ratio 1.93, 90th
    percentile 4.14, median size 85x129 px cut from a 3072x4080 photo.
    A plain Resize((224, 224)) squashes them anisotropically, which
    destroys the crack's proportions: the single most identifying thing
    about a crack is its shape, and a 4:1 crack and a 1:1 crack become
    the same picture. Padding preserves it.
    """
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        return np.full((size, size, 3), pad_value, np.uint8)
    s = size / max(h, w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    r = cv2.resize(crop, (nw, nh), interpolation=interp)
    if r.ndim == 2:
        r = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
    out = np.full((size, size, 3), pad_value, np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return out


class EmbeddingMatcher(BaseMatcher):
    """Base for any 'one vector per crop, cosine similarity' method.

    prepare() returns an (N, D) L2-normalised array, so the whole score
    matrix is a single matmul instead of an N*M Python loop — the main
    speedup for the embedding family.

    Subclasses implement `embed_batch(crops) -> (N, D)`.
    """

    min_score = 0.7                    # cosine; tune per backbone on val data

    @abstractmethod
    def embed_batch(self, crops: list[np.ndarray]) -> np.ndarray: ...

    def prepare(self, instances):
        if not instances:
            return np.zeros((0, 1), dtype=np.float32)
        embs = self.embed_batch([inst.crop for inst in instances])
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.clip(norms, 1e-8, None)

    def build_score_matrix(self, preps_a, preps_b) -> np.ndarray:
        if len(preps_a) == 0 or len(preps_b) == 0:
            return np.zeros((len(preps_a), len(preps_b)))
        return preps_a @ preps_b.T


class TimmEmbeddingMatcher(EmbeddingMatcher):
    """DeiT / ViT via timm (num_classes=0 strips the head)."""

    def __init__(self, model_name: str = "deit_small_patch16_224",
                 image_size: int = 224, min_score: float = 0.7, name: str | None = None):
        self.model_name = model_name
        self.image_size = image_size
        self.min_score = min_score
        self.name = name or model_name
        self._model = None
        self._tf = None

    def _lazy(self):
        if self._model is not None:
            return
        import torch, timm
        from torchvision import transforms
        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        self._model.eval().to(self._device)
        # No Resize here: crops are letterboxed to a square in embed_batch,
        # so the aspect ratio survives. Resize((S, S)) would squash it.
        self._tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def embed_batch(self, crops):
        self._lazy()
        torch = self._torch
        batch = torch.stack([
            self._tf(cv2.cvtColor(letterbox(c, self.image_size), cv2.COLOR_BGR2RGB))
            for c in crops])
        with torch.no_grad():
            feats = self._model(batch.to(self._device))
        return feats.cpu().numpy()


class CLIPEmbeddingMatcher(EmbeddingMatcher):
    """CLIP image tower via transformers (light multimodal baseline)."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 min_score: float = 0.7):
        self.model_name = model_name
        self.min_score = min_score
        self.name = "CLIP"
        self._model = None

    def _lazy(self):
        if self._model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = CLIPModel.from_pretrained(self.model_name).to(self._device).eval()
        self._proc = CLIPProcessor.from_pretrained(self.model_name)

    def embed_batch(self, crops):
        self._lazy()
        torch = self._torch
        from PIL import Image
        # CLIPProcessor resizes the SHORT edge to 224 and then centre-crops,
        # which on an 85x400 crop keeps roughly the middle half of the crack
        # and discards the rest. Letterboxing to a square first makes the
        # resize a no-op and the centre crop harmless; do_center_crop=False
        # is belt and braces for processor versions that ignore that.
        imgs = [Image.fromarray(cv2.cvtColor(letterbox(c, 224), cv2.COLOR_BGR2RGB))
                for c in crops]
        inputs = self._proc(images=imgs, return_tensors="pt",
                            do_center_crop=False).to(self._device)
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        if not torch.is_tensor(feats):
            # transformers >=5 wraps the result in BaseModelOutputWithPooling;
            # its .pooler_output is the already-projected image embedding and
            # is numerically identical to the bare tensor <5 returned here.
            feats = feats.pooler_output
        return feats.cpu().numpy()


class OSNetEmbeddingMatcher(EmbeddingMatcher):
    """OSNet person-ReID backbone via torchreid — the only baseline with a
    metric-learning objective, hence usually the strongest off-the-shelf
    embedding here (still trained on people, not cracks)."""

    def __init__(self, model_name: str = "osnet_x1_0", model_path: str = "",
                 min_score: float = 0.6):
        self.model_name = model_name
        self.model_path = model_path
        self.min_score = min_score
        self.name = "OSNet"
        self._ex = None

    def _lazy(self):
        if self._ex is not None:
            return
        import torch
        try:
            # deep-person-reid, installed from source
            from torchreid.utils import FeatureExtractor
        except ImportError:
            # the `torchreid` PyPI repackage nests everything under .reid
            from torchreid.reid.utils import FeatureExtractor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._ex = FeatureExtractor(model_name=self.model_name,
                                    model_path=self.model_path, device=device)

    def embed_batch(self, crops):
        self._lazy()
        feats = self._ex(crops)          # torchreid handles resize/normalise
        return feats.cpu().numpy()


class YOLOEmbeddingMatcher(EmbeddingMatcher):
    """YOLOv8 backbone as a region-embedding extractor (forward hook on an
    intermediate layer + global average pool)."""

    def __init__(self, weights: str = "yolov8n.pt", layer_index: int = 9,
                 image_size: int = 224, min_score: float = 0.6):
        self.weights = weights
        self.layer_index = layer_index
        self.image_size = image_size
        self.min_score = min_score
        self.name = "YOLOv8-backbone"
        self._model = None

    def _lazy(self):
        if self._model is not None:
            return
        import torch
        from ultralytics import YOLO
        self._torch = torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = YOLO(self.weights).model.to(self._device).eval()
        self._feat = None
        self._model.model[self.layer_index].register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._feat = out

    def embed_batch(self, crops):
        self._lazy()
        torch = self._torch
        import torch.nn.functional as F
        out = []
        with torch.no_grad():
            for c in crops:
                img = cv2.cvtColor(letterbox(c, self.image_size),
                                   cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self._device)
                _ = self._model(t)
                v = F.adaptive_avg_pool2d(self._feat, 1).flatten(1)
                out.append(v.squeeze(0).cpu().numpy())
        return np.stack(out)


# ---------------------------------------------------------------------------
# Family 3: learned pairwise matchers (SuperGlue, LoFTR)
# ---------------------------------------------------------------------------

def _add_superglue_to_path():
    """Put the Magic Leap SuperGlue checkout on sys.path, lazily.

    Its package is called `models`, which is far too generic to add to the
    interpreter path globally (a .pth in site-packages would shadow any
    other top-level `models` for every program in the env). So the path
    goes on only when the SuperGlue baseline is actually instantiated, and
    only if `models.matching` is not already importable.

    Override the location with $SUPERGLUE_PATH; the default is the
    third_party/ checkout beside this file."""
    import importlib.util
    try:
        if importlib.util.find_spec("models.matching") is not None:
            return
    except (ImportError, ValueError):
        pass          # parent package absent: find_spec raises, not returns None
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "third_party", "SuperGluePretrainedNetwork")
    root = os.environ.get("SUPERGLUE_PATH", default)
    if not os.path.isdir(root):
        raise ImportError(
            f"SuperGlue checkout not found at {root}. Clone it with\n"
            f"  git clone https://github.com/magicleap/SuperGluePretrainedNetwork "
            f"third_party/SuperGluePretrainedNetwork\n"
            f"or point $SUPERGLUE_PATH at an existing copy.")
    if root not in sys.path:
        sys.path.insert(0, root)


class SuperGlueMatcher(BaseMatcher):
    """SuperPoint + SuperGlue. Genuinely pairwise (the matcher reasons
    jointly over both images), so prepare() only caches the preprocessed
    grayscale tensor; the network runs per pair. Score = geometric-inlier
    count, thresholded by the shared solver."""

    apply_mask = True

    def __init__(self, weights: str = "indoor", resize: int = 512,
                 min_inliers: int = 6):
        # resize was 256, on crops whose median long side is 140 px -- the
        # network was being fed upsampled blur. weights was "outdoor";
        # these are flat, low-texture, close-range interior surfaces, which
        # is what the indoor model was trained on.
        self.weights = weights
        self.resize = resize
        self.min_inliers = min_inliers
        self.min_score = float(min_inliers)
        self.name = "SuperGlue"
        self._matching = None

    def _lazy(self):
        if self._matching is not None:
            return
        import torch
        _add_superglue_to_path()
        from models.matching import Matching       # from the Magic Leap repo
        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = {"superpoint": {"nms_radius": 4, "keypoint_threshold": 0.005, "max_keypoints": 1024},
               "superglue": {"weights": self.weights, "sinkhorn_iterations": 20, "match_threshold": 0.2}}
        self._matching = Matching(cfg).eval().to(self._device)

    def _prep_gray(self, crop):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        if self.resize > 0:
            scale = self.resize / max(gray.shape[:2])
            nw, nh = max(int(gray.shape[1] * scale), 8), max(int(gray.shape[0] * scale), 8)
            gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_CUBIC)
        return gray

    def prepare(self, instances):
        return [self._prep_gray(inst.crop) for inst in instances]

    def score_pair(self, gray_a, gray_b) -> float:
        self._lazy()
        torch = self._torch
        from models.utils import frame2tensor
        ta = frame2tensor(gray_a, self._device)
        tb = frame2tensor(gray_b, self._device)
        with torch.no_grad():
            pred = self._matching({"image0": ta, "image1": tb})
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        matches = pred["matches0"]
        valid = matches > -1
        n = int(valid.sum())
        conf = pred.get("matching_scores0")
        q = float(conf[valid].mean()) if conf is not None and n else 0.0
        if n < 4:                        # too few points to verify geometrically
            return n + _tiebreak(q)
        mk0 = pred["keypoints0"][valid]
        mk1 = pred["keypoints1"][matches[valid]]
        _, inl = cv2.findHomography(mk0, mk1, cv2.RANSAC, 5.0)
        if inl is None:
            return n + _tiebreak(q)
        n_in = int(inl.sum())
        return n_in + _tiebreak(n_in / n, q)


class LoFTRMatcher(BaseMatcher):
    """Detector-free dense matching via kornia LoFTR. Also inherently
    pairwise; prepare() caches the preprocessed tensor, network runs per
    pair. Score = geometric-inlier count."""

    apply_mask = True

    def __init__(self, pretrained: str = "indoor", resize: int = 512,
                 conf_thresh: float = 0.5, min_inliers: int = 10):
        # At resize=256 LoFTR's coarse grid is ~32 cells on the long side,
        # so only a handful of matches are geometrically possible before
        # RANSAC ever sees them. "indoor" (ScanNet) is far closer to smooth
        # painted plaster than the outdoor landmark model.
        self.pretrained = pretrained
        self.resize = resize
        self.conf_thresh = conf_thresh
        self.min_inliers = min_inliers
        self.min_score = float(min_inliers)
        self.name = "LoFTR"
        self._matcher = None

    def _lazy(self):
        if self._matcher is not None:
            return
        import torch
        import kornia as K
        import kornia.feature as KF
        self._torch = torch
        self._K = K
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._matcher = KF.LoFTR(pretrained=self.pretrained).to(self._device).eval()

    def _prep(self, crop):
        self._lazy()
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        if self.resize > 0:
            scale = self.resize / max(gray.shape[:2])
            nw, nh = max(int(gray.shape[1] * scale), 8), max(int(gray.shape[0] * scale), 8)
        else:
            nh, nw = gray.shape[:2]
        nh, nw = max((nh // 8) * 8, 8), max((nw // 8) * 8, 8)   # LoFTR needs /8
        gray = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_CUBIC)
        return self._K.image_to_tensor(gray, False).float() / 255.0

    def prepare(self, instances):
        return [self._prep(inst.crop).to(self._device) for inst in instances]

    def score_pair(self, ta, tb) -> float:
        torch = self._torch
        with torch.no_grad():
            corr = self._matcher({"image0": ta, "image1": tb})
        mk0 = corr["keypoints0"].cpu().numpy()
        mk1 = corr["keypoints1"].cpu().numpy()
        conf = corr["confidence"].cpu().numpy()
        keep = conf > self.conf_thresh
        mk0, mk1 = mk0[keep], mk1[keep]
        n = len(mk0)
        q = float(conf[keep].mean()) if n else 0.0
        if n < 4:
            return n + _tiebreak(q)
        _, inl = cv2.findHomography(mk0, mk1, cv2.RANSAC, 3.0)
        if inl is None:
            return n + _tiebreak(q)
        n_in = int(inl.sum())
        return n_in + _tiebreak(n_in / n, q)




# ---------------------------------------------------------------------------
# Family 4: crack morphology  (the shape of the crack itself)
# ---------------------------------------------------------------------------

def crack_shape_descriptor(mask_crop: np.ndarray, n_bins: int = 48,
                           n_freq: int = 16) -> np.ndarray:
    """A viewpoint-tolerant description of a crack's own geometry.

    Every other baseline here is either generic appearance (ImageNet,
    CLIP, person re-ID) or geometric registration of the surrounding
    wall. None of them describes the thing that actually identifies a
    crack: its shape. On a smooth painted wall there is no texture to
    fall back on, so shape is all there is.

    Construction, chosen so the descriptor survives the transformations a
    second photo applies:

      1. PCA-align the crack pixels, so the long axis is horizontal.
         Removes in-plane rotation.
      2. Resample the centreline into `n_bins` bins along that axis and
         take the mean perpendicular offset in each. This is the crack's
         waviness signature.
      3. Divide offsets by the crack's length. Removes scale.
      4. Take |FFT| of the offset profile. A second photo may traverse
         the crack in the opposite direction; the magnitude spectrum is
         invariant to that reversal, where the raw profile is not.

    Plus scale-free scalars that carry what the spectrum drops:
    tortuosity, elongation, relative thickness, and the spread of local
    turning angles.

    Returns an L2-normalised vector, so cosine similarity is the score.
    """
    ys, xs = np.nonzero(mask_crop > 0)
    n = len(xs)
    out_dim = n_freq + 8
    if n < 12:
        return np.zeros(out_dim, dtype=np.float32)

    P = np.stack([xs, ys], 1).astype(np.float64)
    P -= P.mean(0)
    # principal axis
    w, V = np.linalg.eigh(np.cov(P.T) + 1e-9 * np.eye(2))
    axis, perp = V[:, 1], V[:, 0]
    u = P @ axis                     # along the crack
    v = P @ perp                     # across it

    length = float(u.max() - u.min())
    if length < 1e-6:
        return np.zeros(out_dim, dtype=np.float32)

    # centreline profile: mean offset per bin along the axis
    idx = np.clip(((u - u.min()) / length * n_bins).astype(int), 0, n_bins - 1)
    prof = np.zeros(n_bins)
    thick = np.zeros(n_bins)
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            prof[b] = v[sel].mean()
            thick[b] = v[sel].max() - v[sel].min()
    filled = prof != 0
    if filled.sum() >= 2:            # interpolate across empty bins
        prof = np.interp(np.arange(n_bins), np.flatnonzero(filled), prof[filled])
    prof /= length                   # scale-free

    spec = np.abs(np.fft.rfft(prof - prof.mean()))[1:n_freq + 1]
    if len(spec) < n_freq:
        spec = np.pad(spec, (0, n_freq - len(spec)))

    d = np.diff(prof)
    ang = np.arctan2(d, 1.0 / max(n_bins, 1))
    arc = float(np.sum(np.hypot(np.diff(prof) * length, length / n_bins)))
    end_to_end = float(np.hypot(length, (prof[-1] - prof[0]) * length))

    scalars = np.array([
        arc / max(end_to_end, 1e-6) - 1.0,        # tortuosity above straight
        np.log1p(np.sqrt(w[1] / max(w[0], 1e-9))),  # elongation
        thick.mean() / length,                    # relative thickness
        thick.std() / max(thick.mean(), 1e-6),
        np.abs(d).mean() * n_bins,                # mean |slope|
        ang.std(),                                # turning-angle spread
        n / max(length ** 2, 1e-6),               # fill / raggedness
        np.abs(prof).max(),                       # peak excursion (scale-free)
    ], dtype=np.float64)
    scalars = np.nan_to_num(scalars, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalise the two blocks SEPARATELY before joining them. Concatenating
    # raw values and L2-normalising once let whichever block happened to have
    # the larger magnitude dominate the cosine -- with the scalars winning,
    # the waviness spectrum contributed almost nothing and two cracks of
    # completely different frequency scored 0.999 alike. Each block is now a
    # unit vector, and `shape_weight` sets how much the spectrum is worth.
    shape_weight = 0.75
    spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
    ns, nc = np.linalg.norm(spec), np.linalg.norm(scalars)
    spec = spec / ns if ns > 1e-8 else spec
    scalars = scalars / nc if nc > 1e-8 else scalars

    vec = np.concatenate([shape_weight * spec,
                          (1.0 - shape_weight) * scalars]).astype(np.float32)
    nrm = np.linalg.norm(vec)
    return vec / nrm if nrm > 1e-8 else vec


class CrackShapeMatcher(EmbeddingMatcher):
    """Cosine similarity over crack_shape_descriptor. No weights, no GPU."""

    def __init__(self, min_score: float = 0.9):
        self.min_score = min_score
        self.name = "CrackShape"

    def embed_batch(self, crops):        # unused; prepare() needs the MASK
        raise NotImplementedError

    def prepare(self, instances):
        if not instances:
            return np.zeros((0, 24), dtype=np.float32)
        return np.stack([crack_shape_descriptor(i.mask_crop) for i in instances])


# ===========================================================================
# Registry + benchmark loop
# ===========================================================================

# Factories (not instances) so heavy weights/imports happen only when a
# method is actually selected to run.
REGISTRY: dict[str, Callable[[], BaseMatcher]] = {
    "sift":      lambda: KeypointMatcher("sift"),
    "orb":       lambda: KeypointMatcher("orb"),
    "superglue": lambda: SuperGlueMatcher(),
    "loftr":     lambda: LoFTRMatcher(),
    "deit":      lambda: TimmEmbeddingMatcher("deit_small_patch16_224", name="DeiT"),
    "vit":       lambda: TimmEmbeddingMatcher("vit_base_patch16_224", name="ViT"),
    "clip":      lambda: CLIPEmbeddingMatcher(),
    "osnet":     lambda: OSNetEmbeddingMatcher(),
    "yolo":      lambda: YOLOEmbeddingMatcher(),
    "shape":     lambda: CrackShapeMatcher(),
}


def evaluate(assignment: Assignment, gt_pairs: set[tuple[int, int]]) -> dict:
    """Precision / recall / F1 of predicted correspondences against a
    ground-truth set of (idx_a, idx_b) pairs. This is the per-pair
    matching metric; plug the same assignments into your CMC/mAP harness
    for the retrieval view."""
    pred = {(i, j) for i, j, _ in assignment.pairs}
    tp = len(pred & gt_pairs)
    fp = len(pred - gt_pairs)
    fn = len(gt_pairs - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def run_benchmark(image_a, mask_a, image_b, mask_b,
                  methods: list[str] | None = None,
                  gt_pairs: set[tuple[int, int]] | None = None,
                  min_area: int = 200) -> dict[str, Assignment]:
    """Run every selected method through the same interface and print one
    table. Methods whose dependencies are missing are skipped with a note.

    Instances are extracted per-method because keypoint and embedding
    families disagree on `apply_mask`; the (cheap) connected-component
    step is redone, the (expensive) feature extraction is cached inside
    each matcher.
    """
    methods = methods or list(REGISTRY)
    results: dict[str, Assignment] = {}

    print(f"{'method':<18} {'pairs':>6} {'newA':>6} {'newB':>6} {'sec':>7}", end="")
    print(f" {'P':>6} {'R':>6} {'F1':>6}" if gt_pairs is not None else "")
    print("-" * (60 if gt_pairs is not None else 45))

    for key in methods:
        if key not in REGISTRY:
            print(f"{key:<18} (unknown method, skipped)")
            continue
        try:
            matcher = REGISTRY[key]()
            ia = extract_crack_instances(image_a, mask_a, min_area=min_area, apply_mask=matcher.apply_mask)
            ib = extract_crack_instances(image_b, mask_b, min_area=min_area, apply_mask=matcher.apply_mask)
            assignment = matcher.match(ia, ib)
            results[key] = assignment

            line = (f"{matcher.name:<18} {len(assignment.pairs):>6} "
                    f"{len(assignment.unmatched_a):>6} {len(assignment.unmatched_b):>6} "
                    f"{assignment.seconds:>7.2f}")
            if gt_pairs is not None:
                m = evaluate(assignment, gt_pairs)
                line += f" {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}"
            print(line)
        except ImportError as e:
            print(f"{key:<18} (missing dependency: {e.name}, skipped)")
        except Exception as e:
            print(f"{key:<18} (error: {type(e).__name__}: {e}, skipped)")
            traceback.print_exc()

    return results


if __name__ == "__main__":
    img1 = cv2.imread("/content/IMG25-6.jpg", cv2.IMREAD_COLOR)
    img2 = cv2.imread("/content/IMG25-6_rotated.jpg", cv2.IMREAD_COLOR)
    mask1 = cv2.imread("/content/wall_crack_mask_1.png", cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread("/content/wall_crack_mask_2.png", cv2.IMREAD_GRAYSCALE)

    for nm, arr in [("img1", img1), ("img2", img2), ("mask1", mask1), ("mask2", mask2)]:
        if arr is None:
            raise FileNotFoundError(f"Failed to load {nm}")

    for m in (mask1, mask2):
        pass  # (resize-to-image handled inside your pipeline if needed)

    # Run only the methods you have installed, e.g. ["sift", "orb", "deit", "osnet"].
    # gt_pairs = {(0, 0), (1, 2), ...}  # supply to get P/R/F1 columns
    run_benchmark(img1, mask1, img2, mask2, methods=None, gt_pairs=None)
