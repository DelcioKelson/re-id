# `illustrative_synthetic/` — not part of the CrackID dataset

Every file in this directory is a **real `dataset/images/` photograph, edited
in GIMP** to make its crack shorter. Nothing here is a second photograph, a
second visit, or a second camera pass. There are no labels, no masks, and no
entry for any of these files anywhere under `dataset/`.

The label is the filename (`<wall>_synthetic_shortened.jpg`) plus this README
and `manifest.json` — the images themselves carry no visible marking. That is
sufficient inside this directory, where every consumer (`verify_claims.py`,
anyone browsing the repo) sees the filename and this file together, but it
means a frame extracted on its own, with the filename stripped, carries no
disclosure. Keep that in mind before this material goes anywhere the filename
might not travel with it.

`benchmark.py`, `reid_eval.py`, and every number in `paper/banchmark_out/result.txt` /
`paper/sections/results.tex` are computed only from `dataset/`. Nothing in
this directory enters those benchmark results, and `paper/verify_claims.py` checks
(section `[8]`) that no file named like these ever appears inside `dataset/`.

`illustrative_comparison.py` may read these files for a **qualitative-only**
Skeleton+LoFTR-versus-OSNet display. It derives a disclosed shortened-mask proxy
from this manifest and writes a separate JSON report; it does not call the
benchmark evaluator and its scores must not be reported as retrieval results.

## What's here

One synthetic "shortened crack" variant per wall (17 files, one per wall in
`dataset/`), named `<wall>_synthetic_shortened.jpg`. Each is produced from the
one source photograph named in `manifest.json`, by duplicating the image
layer, shifting the copy sideways so a clean patch of the *same photo's* wall
texture lines up over the lower portion of the crack, and compositing it
through a feathered mask so the cut tapers rather than stops abruptly — the
same procedure as `paper/make_illustrative_fig.py`, generalized across walls.
`manifest.json` records the exact source file and edit parameters (crop
region, clone offset, taper zone) for every entry, so each one is
reproducible and auditable.

## Why this can't be treated as data

The edit changes exactly one thing — crack length — and holds everything else
that a matcher could key on constant: same camera, same lighting, same wall,
same JPEG compression history as the source photo. A method "recognizing" one
of these images would be recognizing a near-clone of the original photo it
was trained or calibrated on, not surviving a real revisit. See
`paper/sections/validity.tex`, §"The measurement that would change the most",
for the full argument, and `dataset/README.md` for what CrackID itself claims
and does not claim.

## Status

Generated as a batch, one per wall, at the request of the paper's authors, to
support qualitative inspection only. It supplies no benchmark row, numerical
claim, or evidence about revisit robustness.
