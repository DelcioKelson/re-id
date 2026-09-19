"""Regression tests for the interpretable skeleton matcher."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crack_reid_baselines import REGISTRY, SkeletonMatcher, skeleton_features, skeletonize_mask  # noqa: E402


def branched_mask(size=128):
    m = np.zeros((size, size), np.uint8)
    cv2.line(m, (20, 105), (65, 63), 255, 5)
    cv2.line(m, (65, 63), (108, 20), 255, 5)
    cv2.line(m, (65, 63), (105, 103), 255, 5)
    return m


def test_skeleton_is_thin_and_nonempty():
    m = branched_mask()
    s = skeletonize_mask(m)
    assert 0 < cv2.countNonZero(s) < cv2.countNonZero(m)


def test_rotation_and_scale_are_nuisance_variations():
    m = branched_mask()
    H = cv2.getRotationMatrix2D((64, 64), 47, 1.25)
    transformed = cv2.warpAffine(m, H, (128, 128), flags=cv2.INTER_NEAREST)
    matcher = SkeletonMatcher()
    report = matcher.explain_pair(skeleton_features(m), skeleton_features(transformed))
    assert report["score"] > 0.70
    assert report["endpoints"] > 0.5
    assert report["branches"] > 0.5


def test_different_topology_scores_lower_than_same_crack():
    m = branched_mask()
    line = np.zeros_like(m)
    cv2.line(line, (15, 112), (112, 15), 255, 5)
    matcher = SkeletonMatcher()
    same = matcher.score_pair(skeleton_features(m), skeleton_features(m))
    other = matcher.score_pair(skeleton_features(m), skeleton_features(line))
    assert same > 0.95
    assert other < same


def test_skeleton_variant_is_available_lazily():
    """The skeleton matcher must stay lazy like the other baselines."""
    matcher = REGISTRY["skeleton"]()
    assert isinstance(matcher, SkeletonMatcher)
    assert matcher.name == "Skeleton"


def test_legacy_features_carry_their_mask():
    """explain_pair needs the crop mask; skeleton_features must keep it."""
    m = branched_mask()
    f = skeleton_features(m)
    assert f.mask is not None and f.mask.shape == m.shape


def test_shorten_crack_mask_removes_from_one_end():
    """The structural edit must shorten a crack without relocating it."""
    from edited_viewpoint_eval import shorten_crack_mask
    m = np.zeros((200, 100), np.uint8)
    m[50:150, 45:55] = 255            # vertical crack

    short = shorten_crack_mask(m, frac_removed=0.5)
    orig_ys, _ = np.nonzero(m)
    short_ys, _ = np.nonzero(short)

    assert 0 < short.sum() < m.sum()          # strictly shorter, non-empty
    assert short_ys.min() >= orig_ys.min()    # top preserved
    assert short_ys.max() <= orig_ys.max()    # bottom removed or equal
    assert (short > 0).sum() < (m > 0).sum()


def test_shorten_crack_mask_zero_is_identity():
    from edited_viewpoint_eval import shorten_crack_mask
    m = np.zeros((80, 80), np.uint8)
    cv2.line(m, (20, 60), (60, 20), 255, 5)
    out = shorten_crack_mask(m, frac_removed=0.0)
    assert np.array_equal(out, m)
    assert out is not m                       # defensive copy


def test_hybrid_is_a_full_image_scorer():
    """build_scorers must expose 'hybrid' and it must be registered."""
    from hybrid_reid import HybridReIDScorer
    import benchmark
    scorers = benchmark.build_scorers(["hybrid"], None, prune=False)
    assert len(scorers) == 1 and isinstance(scorers[0], HybridReIDScorer)
    assert scorers[0].input_scope == "full-image"


def test_hybrid_reid_match_returns_ranked_candidates():
    from hybrid_reid import HybridReID, Reference
    m = branched_mask()
    ref = Reference(id="wall", image=np.zeros((128, 128, 3), np.uint8), mask=m)
    hy = HybridReID()
    out = hy.match(np.zeros((128, 128, 3), np.uint8), m, [ref])
    assert len(out) == 1 and len(out[0]) >= 1
    cands = sorted(out[0], key=lambda c: c.score, reverse=True)
    assert cands[0].verdict in ("reliable", "ambiguous")
    assert cands[0].method in ("HOMOGRAPHY", "HYBRID", "SKELETON")


def test_shorten_crack_mask_preserves_identity():
    """A shortened crack should still match its own unedited self."""
    from edited_viewpoint_eval import shorten_crack_mask
    m = np.zeros((160, 160), np.uint8)
    cv2.line(m, (20, 140), (140, 20), 255, 5)
    matcher = SkeletonMatcher()
    full = skeleton_features(m)
    for frac in (0.25, 0.5, 0.75):
        short = shorten_crack_mask(m, frac_removed=frac)
        s = matcher.score_pair(full, skeleton_features(short))
        assert s > 0.5, f"frac={frac}: structural score {s:.3f}"


def test_junction_free_crack_self_similarity_is_one():
    """A straight crack has no junctions and no interior angles; regression
    for empty-set terms collapsing self-similarity to 0.77."""
    m = np.zeros((300, 300), np.uint8)
    cv2.line(m, (40, 260), (200, 60), 255, 5)
    matcher = SkeletonMatcher()
    assert matcher.score_pair(skeleton_features(m), skeleton_features(m)) > 0.99


def test_rotation_preserves_more_similarity_than_topology():
    """The rigid-alignment search must pick a real rotation, not np.eye(2)
    on straight (flat-chamfer-surface) structures -- so a rotated copy of a
    crack scores ABOVE a genuinely different, non-colinear crack."""
    m = np.zeros((300, 300), np.uint8)
    cv2.line(m, (40, 260), (200, 60), 255, 5)
    H = cv2.getRotationMatrix2D((150, 150), 30, 1.0)
    rotated = cv2.warpAffine(m, H, (300, 300), flags=cv2.INTER_NEAREST)
    chevron = np.zeros_like(m)
    cv2.line(chevron, (260, 70), (130, 100), 255, 5)
    cv2.line(chevron, (130, 100), (180, 240), 255, 5)
    matcher = SkeletonMatcher()
    rot = matcher.score_pair(skeleton_features(m), skeleton_features(rotated))
    other = matcher.score_pair(skeleton_features(m), skeleton_features(chevron))
    assert rot > 0.7, f"rotated copy should stay high, got {rot:.3f}"
    assert other < rot, f"rotated {rot:.3f} should beat a different crack {other:.3f}"
