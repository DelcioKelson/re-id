# `illustrative_synthetic/` — not part of the CrackID dataset

Every file in this directory is a **real `dataset/images/` photograph, edited
in GIMP** to make its crack shorter. Nothing here is a second photograph, a
second visit, or a second camera pass. There are no labels, no masks, and no
entry for any of these files anywhere under `dataset/`.

**These images are not read by any part of the pipeline.** `benchmark.py`,
`reid_eval.py`, and every number in `paper/banchmark_out/result.txt` /
`paper/sections/results.tex` are computed only from `dataset/`. Nothing in
this directory has ever been scored, and `paper/verify_claims.py` checks
(section `[8]`) that no file named like these ever appears inside `dataset/`.

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
be reviewed before any decision is made about whether or how any of it enters
the paper. As of this commit, none of it does: no figure, no claim, no table
row, and no file under `paper/` other than this README's mention references
this directory.
