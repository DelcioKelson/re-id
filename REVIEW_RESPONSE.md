# Response to the pre-submission review

Worked against the ten-finding defect schedule. This file records what changed in
the code, three places where the review's own numbers do not reproduce against the
data, the consolidated run that produces the paper's table, and paste-ready wording.

Everything below was measured on this repository with the scripts named. Numbers
marked **[re-run]** need the consolidated benchmark run, which needs a GPU box
(kornia / transformers / torchreid are not installed here and there is no CUDA
device), so they are the one class of thing that could not be produced locally.

---

## 1. Corrections to the review

Three of the schedule's figures do not survive contact with the files. Two of them
change what the fix should be, so they are first.

### 1.1 The sharpness gate table does not reproduce, and gate 25 does not keep all 11 walls

S1 justified a gate at 25 on two properties: *"it keeps all 11 walls, doubles the
registration rate, and discards zero successful registrations."* The first is false
under variance-of-Laplacian at any scale.

Measured (`python image_quality.py dataset --sweep`), quarter-scale grey, test split:

| gate | images kept | walls with ≥2 images | same-wall pairs |
|-----:|------------:|---------------------:|----------------:|
| none |          74 |                   11 |             301 |
| ≥ 10 |          53 |                   10 |             157 |
| ≥ 25 |          40 |                    7 |             126 |
| ≥ 50 |          32 |                    6 |             105 |

The review's table reads 74/11/301, 50/11/144, 41/10/126, 31/6/105, 26/5/89 for
none/25/50/100/200. Its *pair* counts are this table shifted by one row, so its
sharpness measure was roughly twice this one; half-scale and full-scale were both
tried and neither reproduces it either. The measure was never pinned down in the
schedule, which is why the threshold could not be checked.

**What this changes.** A gate at 25 empties walls 13, 14, 16 and 17 — exactly the
four the review names as registering on 0% of their pairs. So the gate does the job
it was chosen for (it discards nothing that registered) but it is *not* free of
walls, and the paper cannot claim it is. The honest framing is the one criterion
the data supports:

> The gate is set at the loosest value that discards no successful registration.
> It removes N of 74 test captures and, with them, four walls whose median sharpness
> is below 25 — walls on which no method in the comparison registers a single pair.
> That is a scope condition, stated here rather than buried.

`image_quality.py --sweep --pairs banchmark_out/pair_outcomes.json` fills in the
registration columns and settles the final threshold from measurement rather than
from the review.

### 1.2 The effective sample is 19 identities, not 44

S5 reports 44 test identities appearing in two or more photos (71 across the
dataset). Counted from `dataset/labels/*.json` through `benchmark.Dataset` itself:

| split | walls | identities | in ≥2 photos |
|-------|------:|-----------:|-------------:|
| val   |     6 |         66 |           18 |
| test  |    11 |        146 |           19 |
| whole |    17 |        212 |           37 |

The identity totals match the review; the multi-photo counts do not. `nQ = 280` is
280 *query instances* (538 labelled test instances, 280 of which have a valid
relevant cell), and those come from **19 identities over 11 walls**.

This makes S5 considerably stronger, not weaker. Leave-one-wall-out on the cheap
descriptor already shows what that does to a point estimate:

```
CrackShape   R@1 0.495 +/- 0.329   mAP 0.464 +/- 0.294   (14 folds)
```

A wall-to-wall standard deviation of 0.33 on a mean of 0.50 is the real
uncertainty. Quote the cluster bootstrap and the LOWO spread; a fixed-split
point estimate from 19 identities cannot separate two methods.

### 1.3 The ECC fallback was dead, so every coverage number was measured with a broken third stage

`register_images` documents three front ends: SIFT on masked texture, SIFT
unmasked, then ECC "which needs no keypoints whatsoever" — the stage the two
smooth-plaster walls depend on. It never ran.

`_ecc_homography` began its pyramid at 160 px on the long side and returned
`(None, 0.0)` the first time any level failed to converge. At 160 px
(160×120 after a σ-1.2 blur) `cv2.findTransformECC` raises on every pair in this
dataset, so the function returned `None` unconditionally and `ecc_or_fail`
reported `ECC corr=0.00` everywhere.

Fixed in `crack_registration_reid.py`: the pyramid starts at 320 px, opens with
`MOTION_TRANSLATION`, and a level that fails keeps the best warp so far instead of
aborting. Verified against the numbers in the function's own docstring, which were
evidently taken when it worked:

| pair | docstring | after the fix |
|------|----------:|--------------:|
| wall13 0001→0002 | 0.955 | **0.958** |
| wall13 0006→0007 | 0.901 | **0.918** |

**What this changes.** Registration coverage, the `scored` column, and everything
S4 says about it were all measured with the fallback disabled. They must be
re-measured. Expect coverage to rise and the registration timing to rise with it
(the fallback costs 5–20 s on a pair SIFT rejects, where it previously cost
nothing).

**And it may not be a pure win — check before claiming it as one.** ECC is driven
by large-scale shading, so its homographies are coarser than a RANSAC fit; the
function's own docstring says so. A coarse homography feeding `symmetric_chamfer`
returns a *finite* distance, which converts a clean non-answer into a possibly
wrong answer. Coverage will rise; accuracy on the newly covered pairs may fall.
That trade is exactly what the coverage-matched table and the three-way `scored`
split are for — read them together after the re-run, and if the ECC-covered pairs
are mostly wrong, raise `ecc_min_corr` above 0.80 rather than keeping coverage you
cannot trust. `viewpoint.json` records the front end per pair, so the two
populations can be separated.

---

## 2. What changed, by finding

| # | Finding | Status |
|---|---------|--------|
| S1 | Blur confounds the viewpoint claim | **code + wording.** `image_quality.py`; `benchmark.py --min-sharpness`; threshold to be fixed from the measured sweep |
| S2 | Frame gap is not a viewpoint proxy | **code + wording.** `viewpoint.py`; `reid_analysis.py --viewpoint-sweep`; frame-gap sweep relabelled a near-duplicate control everywhere |
| S3 | Full-image vs crop is uncontrolled | **code.** `benchmark.py --controls` runs crop+ctx and crack-erased; **[re-run]** for the numbers |
| S4 | `scored = 0.50` will be misread | **code.** three-way split of `scored` measured, saved and reported; per-wall coverage table |
| S5 | n is 11 walls, not 280 queries | **code.** cluster bootstrap already existed and is now printed by default; `--lowo` verified; count corrected (§1.2) |
| S6 | 2,200× slower | **code.** seconds-per-image-pair reported beside seconds-per-query, with the amortisation stated |
| S7 | pairF1 is an oracle number | **done.** column renamed `pairF1*`, val-calibrated `pairF1@v` added beside it |
| S8 | Ground truth merged automatically, never audited | **code.** `label_audit.py`: merge report, sampled audit with contact sheets, error rate with a Wilson interval, inter-annotator ARI |
| S9 | Segmenter accuracy never reported | **code.** `segmentation_audit.py`: mask quality vs hand-drawn GT, min_area/close_px sensitivity sweep |
| S10 | Scope statement | **wording.** §4 below |

### New files

- **`image_quality.py`** — variance-of-Laplacian sharpness, cached to
  `dataset/quality.json`; the gate sweep that justifies a threshold; the admission
  report the paper has to quote.
- **`viewpoint.py`** — per-pair viewpoint covariate from front ends the scoring
  pipeline does not use (AKAZE/ORB unmasked, dense ECC), decomposed into scale,
  in-plane rotation, out-of-plane tilt and projectivity; recomputes the S1 logistic
  evidence and the S2 correlations from one pass; writes `pair_outcomes.json`.
- **`label_audit.py`** — merge provenance, a sampled audit a human can actually
  perform, the audited error rate, and inter-annotator agreement as ARI.
- **`segmentation_audit.py`** — mask quality against hand-drawn masks, and whether
  the ranking survives a different `min_area` / `close_px`.

### Changed files

- **`crack_registration_reid.py`** — `_ecc_homography` fixed (§1.3).
- **`benchmark.py`** — `--min-sharpness` admission gate applied once, before
  extraction, so every method sees the same photographs; `--controls`; out-of-frame
  cells counted; the coverage breakdown saved into each `.npz`; per-image-pair cost.
- **`reid_eval.py`** — `pair_f1_at()`; `pairF1*` / `pairF1@v` columns; a table
  footer that says what `scored` means instead of inviting the misreading.
- **`reid_analysis.py`** — `viewpoint_sweep()` / `--viewpoint-sweep`;
  `per_wall_coverage()` / `--per-wall-coverage`; the three-way `scored` breakdown
  and the amortised cost table in the report; frame-gap sweep relabelled.

---

## 3. The consolidated run

One run, in this order. Steps 1–2 are local and cheap; step 3 needs the GPU box.

```bash
# 1. sharpness, and the gate sweep that fixes the threshold
python image_quality.py dataset --sweep

# 2. viewpoint covariates + per-pair registration outcomes (~1 h, CPU)
python viewpoint.py dataset --out banchmark_out --split test

#    now re-read the sweep with the registration columns filled in and
#    pick the loosest gate that discards no successful registration
python image_quality.py dataset --sweep --pairs banchmark_out/pair_outcomes.json

# 3. the benchmark: persistent matrices, the gate, the controls
python benchmark.py dataset --out banchmark_out --min-sharpness <GATE> --controls

#    and leave-one-wall-out, which is the interval the paper should quote
python benchmark.py dataset --out banchmark_out_lowo --min-sharpness <GATE> --lowo

# 4. everything else reads the saved matrices and costs seconds
python reid_analysis.py banchmark_out --split test \
    --bootstrap 2000 --viewpoint-sweep --per-wall-coverage \
    --frame-gap-sweep --surfaces dataset/walls.csv

# 5. ground truth and segmentation
python label_audit.py dataset --report
python label_audit.py dataset --sample 30 --seed 0      # then fill in the verdicts
python label_audit.py dataset --score dataset/labels/_audit_0.json
python segmentation_audit.py dataset --gt               # needs dataset/masks_gt/
python segmentation_audit.py dataset --sweep --methods sift orb shape
```

`--surfaces dataset/walls.csv` needs a `surface` column adding to that file
(`wall09,textured` / `wall13,smooth`); the stratified table is skipped without it.

---

## 4. Replacement wording

Numbers in `[brackets]` come from the consolidated run. Everything else is measured.

### Abstract, opening move

> We ask whether the visual appearance of a crack crop carries enough information
> to establish its identity. Across ten matchers spanning handcrafted keypoints,
> learned correspondence, general vision embeddings, a re-identification
> architecture and a purpose-built shape descriptor, pairwise F1 falls in the
> narrow band 0.305–0.359 — a spread of 0.054 across methods separated by a decade
> of research and three orders of magnitude in compute. We read this as a ceiling
> in the data rather than a limitation of any encoder.
>
> We therefore locate identity in wall geometry instead of crack appearance: a
> homography estimated from wall texture with the cracks masked out, followed by
> Chamfer matching of projected crack masks. On 17 walls, 212 annotated crack
> identities and 140 photographs, this reaches a detection-and-identification rate
> of [DIR] at 10% false alarm against [best baseline] for the best crop-based
> method, under a protocol that keeps unanswerable queries and calibrates all
> thresholds on a held-out validation split.
>
> Captures are admitted by a sharpness threshold applied identically to every
> method, rejecting [N] of 74 test images and with them [K] walls on which no
> method registers a single pair, so that viewpoint is evaluated on inputs a
> deployed system would accept. On admissible captures, matching holds across
> scale changes to [S]× and camera rotations to [R]° at the 90th percentile.
> All captures of a wall share one lighting condition, and image quality is
> controlled by admission rather than varied, so neither is evaluated as a factor
> here.

### Protocol section — the disclosure that pre-empts S1 and S2

> Each wall was captured in a single continuous pass, so the evaluation measures
> viewpoint and scale invariance rather than temporal re-identification. Query and
> gallery images of a wall share illumination and were captured seconds apart. We
> use the term *correspondence* rather than *re-identification* throughout to keep
> this distinction explicit, and we expect these figures to be an upper bound on
> multi-session performance.
>
> Viewpoint change is measured per image pair by a front end the scoring pipeline
> does not use — AKAZE correspondences on the unmasked image, falling back to dense
> ECC alignment — and decomposed into isotropic scale change, in-plane rotation and
> the out-of-plane tilt implied by foreshortening. Estimating it independently
> matters: a covariate derived from the method's own homography exists only where
> the method already succeeds, and stratifying on it would be circular.
>
> We do not use the gap in frame index as a viewpoint variable. Over the registered
> test pairs it correlates with in-plane rotation ([rho], [p]) but not with scale
> ([rho], n.s.) or tilt ([rho], n.s.), so a frame-gap sweep is a camera-roll sweep —
> and roll is the one component a homography absorbs exactly. We report the
> frame-gap sweep as a near-duplicate control over capture separation, and build no
> invariance claim on it.

### Coverage — the paragraph that defuses S4

> The registration method returns a finite score for [X]% of valid query–gallery
> pairs; every baseline returns one for all of them. This is a penalty rather than
> an advantage. Pairs that fail to register are assigned the failure distance and
> rank last, and the query is still counted in every metric. We verified this
> directly by censoring a complete score matrix: at 34% coverage Rank-1 fell from
> 0.422 to 0.27, and censoring whole image-pair blocks — registration's actual
> failure structure — was no more forgiving than censoring cells uniformly.
>
> The unscored fraction is made of two different events, which we report
> separately. [A]% of valid cells are pairs whose images could not be registered:
> a genuine non-answer. A further [B]% are pairs that registered and in which the
> query crack projects outside the gallery photograph — a confident, geometrically
> grounded "this crack is not in that image", which no crop-scope method can
> produce at all.
>
> Registration failures are concentrated by wall rather than distributed at random.
> We therefore report coverage per wall, resample walls rather than queries when
> computing confidence intervals, and give a coverage-matched comparison
> separately, so that the quality of the method's answers and the frequency with
> which it produces one remain two distinct claims.

### Ground truth — the paragraph S8 needs

> Identities were seeded automatically: a click point is propagated between
> consecutive photographs through an ORB homography and a one-to-one assignment,
> and identities whose components appear to be fragments of one physical crack were
> merged. **95 of the 212 identities (45%) were formed by such a merge**, with a
> median maximum merge gap of 228 px on the test split and 286 px on validation.
> Because a merge that joins two distinct cracks converts a hard negative into a
> free positive and inflates mAP for every method at once, we audited a random
> sample of [N] identities, stratified over merged and unmerged, and found an error
> rate of [E]% (95% CI [lo, hi]). [A second annotator labelled walls X, Y and Z
> independently; adjusted Rand index against the primary annotator was [ARI].]

Worth knowing before that paragraph is written: the merge pass restructured the
partition almost completely. Adjusted Rand index between the final labels and the
pre-merge prefill output (`dataset/labels_old/`) is **0.080** over 1,051 matched
points across 140 photos — near-zero agreement. That is the same fact as "45% of
identities absorbed more than one prefill identity", seen from the other side. It
is not an argument against merging; it is why an audited error rate has to exist
before any mAP computed on these labels is quoted.

```
python label_audit.py dataset --agree dataset/labels dataset/labels_old
```

### Scope — S10, in the abstract rather than the limitations

> All captures come from a single site over four half-days, using two handset
> models, on two broad surface families: stippled render and smooth painted
> plaster. We claim a benchmark and a method for these surfaces under these
> conditions, not a general solution to crack re-identification.

---

## 5. Ordering the claims

Lead with the ceiling, not with R@1.

1. **Lead:** crop appearance is at a ceiling for cracks — ten methods spanning four
   families land in a 0.054-wide band of pairwise F1 — so identity must come from
   geometry.
2. **Then:** registration converts that geometry into a usable open-set decision.
   DIR@FAR=0.1 is the metric that carries the argument, because a registered image
   pair *licenses a negative* ("project the crack, look, it is not there") and a
   crop never can.
3. **Last, and hedged:** R@1. It is the most contaminated by S1 and S2 and the
   least surprising given S3.

---

## 6. Still open

- **[re-run]** every number that needs the benchmark: the controls (S3), the
  post-fix coverage split (S4), the viewpoint sweep cells (S2), CIs on the real
  matrices (S5).
- **Label audit verdicts** (S8) — the sampling and the scoring are built; a human
  has to fill in 30 verdicts.
- **Hand-drawn masks** (S9) — `dataset/masks_gt/` does not exist; ~10 photos.
- **`surface` column** in `dataset/walls.csv` (S4 stratification).
- **Controlled-viewpoint capture** — one afternoon re-shooting a subset at marked
  distances and angles. This is the only thing that converts the central claim from
  inferred to designed, and no amount of code substitutes for it.
