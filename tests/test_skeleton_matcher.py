"""Regression tests for the interpretable skeleton matcher."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crack_reid_baselines import REGISTRY, SkeletonLoFTRMatcher, SkeletonMatcher, skeleton_features, skeletonize_mask  # noqa: E402


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


def test_loftr_skeleton_variant_is_available_without_loading_weights():
    """The optional learned matcher must remain lazy like the other baselines."""
    matcher = REGISTRY["skeleton-loftr"]()
    assert isinstance(matcher, SkeletonLoFTRMatcher)
    assert matcher.name == "Skeleton+LoFTR"


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
