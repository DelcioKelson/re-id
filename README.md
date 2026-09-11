# CrackID

A benchmark for **crack re-identification**: given a new photograph of a wall
carrying several cracks, decide which recorded crack each observation belongs
to, and which are new.

140 photographs, 17 exterior walls, 212 annotated crack identities, eleven
matchers, and a geometric reference baseline — released with the annotation
provenance and the chance level of every metric stated, rather than as
unaudited labels and bare numbers.

```
pip install -r requirements.txt
python -m pytest tests/ -q                       # metric layer
python chance_baseline.py dataset --out banchmark_out
python benchmark.py dataset --out banchmark_out --ablations
python paper/verify_claims.py                    # re-derive every quoted number
```

## What this measures, and what it does not

Each wall was captured in a **single continuous walk-around**. Consecutive
frames are a median of 3 s apart. So the benchmark measures **viewpoint and
scale correspondence** — find this crack from a different position, distance
and angle — and **not** correspondence across separate visits, which is the
deployment case. Treat every figure as an upper bound on multi-visit
performance, and see `dataset/README.md` for wording that stays inside what
the data supports.

Two consequences are load-bearing and are reported rather than buried:

* **For 100% of answerable test queries the nearest correct answer is the
  immediately adjacent frame.** At `--min-frame-gap 0` this is near-duplicate
  retrieval. Sweep the gap (`--min-frame-gap 1 2 3`) and read the numbers as a
  curve, not a point.
* **Every metric has a chance level, and two of them are high.** Pairwise F1
  cannot fall below 0.305 on this pool and Rank-5 chance is 0.767. Run
  `chance_baseline.py` and quote the floor beside the number.

## Read the numbers against a baseline, not against zero

`chance_baseline.py` and the `coverage-only` scorer exist because several
published-looking numbers here are at or below chance:

| | R@1 | R@5 | mAP | DIR@FAR=.1 | pair F1 | assign F1 |
|---|---|---|---|---|---|---|
| chance (200 random matrices) | 0.336 | 0.767 | 0.385 | 0.038 | **0.305** | 0.311 |
| `coverage-only` (never sees a crack) | 0.515 | 0.874 | 0.533 | **0.000** | 0.544 | 0.591 |

`coverage-only` answers YES to every pair whose two photographs registered.
It beats the best crop-scope method's mAP — and scores exactly zero on the
open-set metric. So the geometric method's ranking and pairwise-F1 margins are
substantially **alignment coverage**, while its DIR@FAR margin is entirely
**Chamfer matching**. Any new method should be reported against both rows.

## Layout

| Path | What it is |
|---|---|
| `dataset/` | images, masks, click-point labels, splits, quality, contact sheets |
| `benchmark.py` | dataset loading, scorer construction, the run driver |
| `reid_eval.py` | the protocol: validity masks, CMC/mAP, open-set curve, pairwise and assignment F1, chance levels |
| `crack_registration_reid.py` | the geometric method: masked registration, Chamfer agreement, Hungarian matching |
| `crack_reid_baselines.py` | SIFT, ORB, SuperGlue, LoFTR, ViT, DeiT, CLIP, YOLO, OSNet, CrackShape, and the interpretable Skeleton matcher |
| `chance_baseline.py` | chance levels, the frame-gap structure, the coverage confound |
| `viewpoint.py` | per-pair registration outcomes and the independent viewpoint covariate |
| `image_quality.py` | sharpness measurement and the admission gate sweep |
| `label_points.py`, `prefill_labels.py`, `label_audit.py` | annotation, merge provenance, the audit instrument |
| `tests/` | the metric layer's regression tests |
| `paper/` | LaTeX source, figures, and `verify_claims.py` |

## Reproducing the paper

Every number quoted in the paper is re-derived from committed artefacts by
`paper/verify_claims.py`, which exits non-zero if any of them stops matching.
See `paper/README.md` for the artefact-to-claim map and for the two claims that
need the score matrices regenerated.

## Skeleton versus OSNet

`skeleton-loftr` turns each predicted crack mask into a centreline and compares
endpoints, junctions, curvature/shape, topology/segment lengths, and relative
width. LoFTR supplies pairwise learned keypoints; only correspondences in a
dilated skeleton neighbourhood in both crops are retained, and their RANSAC
consensus adds up to 15% corroborating evidence. Missing LoFTR points do not
penalise a structural match. The structural terms remain separately reported.
It requires optional `torch` and `kornia`.

To compare synthetic queries against the untouched original-photo gallery:

    python3 synthetic_viewpoint.py dataset --methods skeleton-loftr osnet@ctx1

To include the requested context baseline in the real-data benchmark:

    python3 benchmark.py dataset --methods skeleton-loftr osnet@ctx1 --out skeleton_vs_osnet

For the separate GIMP illustrative images, use the qualitative-only runner;
its JSON is deliberately not a benchmark result:

    python3 illustrative_comparison.py --out illustrative_comparison_out

## Provenance of the crack masks

`dataset/masks/` is **committed**, so no reported number requires re-running
the segmenter. The masks come from a UNet++ crack segmenter trained on
**public crack-segmentation data only** — no CrackID wall, and no photograph
from this site or capture session, appears in its training set, so there is no
path from the evaluation walls into the mask model. Masks are treated as given
and their error is not quantified; `segmentation_audit.py` contains the
sensitivity sweep, which becomes checkable once hand-drawn reference masks
exist.

## Licence and citation

The SuperGlue checkout under `third_party/` is research / non-commercial —
check its licence before use. Everything else here is this project's.
