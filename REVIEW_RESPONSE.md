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

Measured (`python image_quality.py dataset --sweep --pairs banchmark_out/pair_outcomes.json`),
quarter-scale grey, test split, full 301-pair registration scan:

| gate  | images kept | walls with ≥2 images | same-wall pairs | registered | rate |
|------:|------------:|----------------------:|-----------------:|-----------:|-----:|
| none  |          74 |                     11 |               301 |         97 |  32% |
| ≥ 10  |          53 |                     10 |               157 |         88 |  56% |
| ≥ 25  |          40 |                      7 |               126 |         86 |  68% |
| ≥ 50  |          32 |                      6 |               105 |         72 |  69% |
| ≥ 100 |          26 |                      5 |                89 |         56 |  63% |
| ≥ 200 |          20 |                      4 |                74 |         41 |  55% |

(`sweep()` itself had a bug worth naming: its `walls` column counted walls with
*any* surviving image, not walls that can still form a same-wall pair. A wall down
to one image reads as "kept" but contributes zero pairs. Fixed — the table above is
walls with ≥2 images, which is what "keeps every wall" has to mean for a pairing
benchmark.)

The review's table reads 74/11/301, 50/11/144, 41/10/126, 31/6/105, 26/5/89 for
none/25/50/100/200 — a different sharpness measure that was never pinned down in
the schedule, which is why its threshold could not be checked directly.

**What this changes.** No gate keeps every wall pairable; the honest question is
which gate loses the least. At **25**, the review's proposed threshold, four walls
lose the ability to form a same-wall pair at all — 13 and 16 emptied outright, 14
and 17 reduced to a single surviving image. That is exactly the set the review
names as registering on 0% of their pairs, so the gate does remove the walls it was
chosen to remove — but it is not free, and costs 11 of the 97 successful
registrations along with it.

**Gate is set at 10**, not 25. It costs exactly one wall (17, already flagged
`unusable` on its own median sharpness), loses 9 of 97 successful registrations
(9%), and still nearly doubles the registration success rate among what remains
(32%→56%) by stripping the captures that were never going to register. That is the
loosest gate that does not gut a second wall on top of the one the review itself
already wrote off.

> The gate is set at the loosest sharpness threshold that costs no more than one
> wall its ability to form a same-wall pair. At variance-of-Laplacian ≥ 10
> (quarter-scale grey), that removes 21 of 74 test captures, drops wall 17 to a
> single surviving photo, and nearly doubles the registration success rate on what
> remains (32%→56%). A stricter gate (25, as in an earlier draft) buys a higher
> rate but costs three more walls their pairability — a scope trade this paper
> does not need to make.

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

**What this changes — [re-run, on Colab, opencv-contrib].** Re-measured against
the fixed pipeline, full test split, `nQ=280`:

| metric | pre-fix (dead ECC) | post-fix |
|---|---:|---:|
| R@1 | 0.920 | **0.952** |
| R@5 | 0.956 | **0.992** |
| mAP | 0.862 | **0.886** |
| DIR@FAR.1 | 0.768 | 0.768 |
| assF1 | 0.769 | **0.784** |
| pairF1@v (calibrated) | 0.645† | 0.582 |
| scored | 0.50 | 0.48 |
| total_s | 3939.6 | 7817.3 |

† the pre-fix run predates the oracle/calibrated split (§S7); its single `pairF1`
column is closer in spirit to `pairF1*` (oracle) than to `pairF1@v`.

**The coverage risk did not materialise.** `scored` moved 0.50→0.48 — essentially
flat, not up — while R@1, R@5, mAP and assF1 all rose. That is the opposite of
"coverage rises and drags accuracy down": the fix is not admitting a flood of
coarse ECC homographies past `ecc_min_corr`, it is fixing SIFT/ORB-registered pairs
that the dead fallback could never reach and correctly declining the rest. No
change to `ecc_min_corr` is warranted on this evidence. `viewpoint.json` still
records the front end per pair if that needs re-checking after any future change
to the ECC threshold.

---

## 2. What changed, by finding

| # | Finding | Status |
|---|---------|--------|
| S1 | Blur confounds the viewpoint claim | **done.** `image_quality.py`; `benchmark.py --min-sharpness 10`; gate settled at 10 by measurement (§1.1); sharpness predicts registration failure at z=8.23, p=1.8e-16, joint with frame gap |
| S2 | Frame gap is not a viewpoint proxy | **done.** `viewpoint.py`; only a partial proxy — ρ=0.31 (rotation) and 0.35 (tilt) but ρ=0.10, p=0.24 (scale change, not usable); frame-gap sweep relabelled a near-duplicate control everywhere |
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
#    -> settles at 10 (§1.1): costs one wall (17), 9 of 97 registrations, and
#       nearly doubles the registration rate on what remains (32%->56%)

# 3. the benchmark: persistent matrices, the gate, the controls
python benchmark.py dataset --out banchmark_out --min-sharpness 10 --controls

#    and leave-one-wall-out, which is the interval the paper should quote
python benchmark.py dataset --out banchmark_out_lowo --min-sharpness 10 --lowo

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

- **[re-run]** the controls (S3) and CIs on the real matrices with the gate applied
  (S5, `--lowo` at `--min-sharpness 10`). Registration+chamfer (§1.3) and the
  viewpoint/gate evidence (§1.1, below) are now measured; the other nine methods'
  score matrices are unaffected by any of these fixes and do not need re-running
  (nothing in their scoring path touches `_ecc_homography` or the sharpness gate
  logic) — only their table row needs reprinting through the updated `reid_eval.py`
  for the `pairF1*`/`pairF1@v` columns.
- **Label audit verdicts** (S8) — the sampling and the scoring are built; a human
  has to fill in 30 verdicts.
- **Hand-drawn masks** (S9) — `dataset/masks_gt/` does not exist; ~10 photos.
- **`surface` column** in `dataset/walls.csv` (S4 stratification).
- **Controlled-viewpoint capture** — one afternoon re-shooting a subset at marked
  distances and angles. This is the only thing that converts the central claim from
  inferred to designed, and no amount of code substitutes for it.

### Measured: the viewpoint covariate, full 301-pair scan (`viewpoint.py --report-only`)

The circularity worry from an earlier partial scan (covariate coverage trailing
registration coverage) does not hold up on the complete pass: the covariate front
ends (AKAZE/ORB + dense ECC, independent of the scoring pipeline) solve 139/301
pairs (46%) against the scoring pipeline's 97/301 (32%) — and 51 of those 139 are
pairs where the scoring pipeline's own registration *failed*. That is 51 of 301
pairs (17%, or 25% of the 204 registration failures) where the paper can now report
a viewpoint covariate the scorer itself never used to produce it — the thing S2
asked for.

| component | median | p90 | max |
|---|---:|---:|---:|
| scale change (×) | 1.42 | 2.26 | 5.20 |
| in-plane rotation (deg) | 6.92 | 39.58 | 130.42 |
| out-of-plane tilt (deg) | 25.57 | 75.32 | 89.57 |
| projectivity | 1.39 | 6.75 | 5744.46 |

(the projectivity max is a degenerate single-pair estimate, not a claim — filter or
cap before it goes in a table.)

Cross-estimator check on the 32 pairs both AKAZE/ORB and ECC solve: median
|log-scale| difference 0.106 (11.2%), median rotation difference 2.91°. That is
close enough to treat either front end's estimate as usable evidence, which is the
basis for pooling them into one covariate column.

What predicts a registration failure — one joint logistic fit, so each effect is
controlled for the other:

| predictor | median (registered) | median (failed) | Wald z | p |
|---|---:|---:|---:|---:|
| sharpness (log1p) | 133.8 | 7.2 | 8.23 | 1.8e-16 |
| frame gap | 2.0 | 4.0 | −5.03 | 4.8e-07 |

Sharpness is the dominant predictor; frame gap remains significant once sharpness
is controlled for, but far smaller. This is the number that justifies gating on
sharpness rather than on frame gap.
