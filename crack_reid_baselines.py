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
        good = [m for pair in knn if len(pair) == 2
                for m, n in [pair] if m.distance < self.ratio * n.distance]
        if len(good) < 4:
            return float(len(good))

        pts_a = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts_b = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        _, inliers = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, self.ransac_thresh)
        if inliers is None:
            return 0.0
        return float(int(inliers.sum()))


# ---------------------------------------------------------------------------
# Family 2: global embeddings + cosine similarity
# ---------------------------------------------------------------------------

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
        self._tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def embed_batch(self, crops):
        self._lazy()
        torch = self._torch
        batch = torch.stack([self._tf(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops])
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
        imgs = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
        inputs = self._proc(images=imgs, return_tensors="pt").to(self._device)
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
                img = cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB),
                                 (self.image_size, self.image_size)).astype(np.float32) / 255.0
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

    def __init__(self, weights: str = "outdoor", resize: int = 256,
                 min_inliers: int = 6):
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
        if n < 4:                        # too few points to verify geometrically
            return float(n)
        mk0 = pred["keypoints0"][valid]
        mk1 = pred["keypoints1"][matches[valid]]
        _, inl = cv2.findHomography(mk0, mk1, cv2.RANSAC, 5.0)
        return float(int(inl.sum())) if inl is not None else float(n)


class LoFTRMatcher(BaseMatcher):
    """Detector-free dense matching via kornia LoFTR. Also inherently
    pairwise; prepare() caches the preprocessed tensor, network runs per
    pair. Score = geometric-inlier count."""

    apply_mask = True

    def __init__(self, pretrained: str = "outdoor", resize: int = 256,
                 conf_thresh: float = 0.5, min_inliers: int = 10):
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
        if n < 4:
            return float(n)
        _, inl = cv2.findHomography(mk0, mk1, cv2.RANSAC, 3.0)
        return float(int(inl.sum())) if inl is not None else float(n)


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
