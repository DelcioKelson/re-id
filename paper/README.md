# Paper: *CrackID: A Benchmark for Crack Re-Identification*

LaTeX source for the IEEE conference submission built from this repository.

    pdflatex main && bibtex main && pdflatex main && pdflatex main

## Where every number comes from

Every figure and table is regenerated from committed artefacts; nothing is
transcribed by hand except Table I, which is a direct copy of
`../banchmark_out/result.txt` (checked by `verify_claims.py`).

| Artefact | Produced by | Used for |
|---|---|---|
| `../banchmark_out/result.txt` | `benchmark.py` | Table I, Fig. 1 |
| `../banchmark_out/pair_outcomes.json` | `viewpoint.py` | Fig. 2, coverage, gate sweep |
| `../banchmark_out/viewpoint.json` | `viewpoint.py` | Fig. 3, viewpoint stats, logistic fit |
| `../dataset/quality.json` | `image_quality.py` | gate sweep, Fig. 2 ordering |
| `../dataset/labels/*.json` | `label_points.py` | Table II (composition) |
| `../dataset/labels/_changelog.json` | `prefill_labels.py` | merge provenance, Sec. IV-C |
| `../dataset/labels/_triage.json` | `prefill_labels.py` | the 57 widest-gap merges |
| `../dataset/labels_old/` | `prefill_labels.py` | pre-merge labelling, for the ARI |
| `synth/rows_{rot,tilt,scale}.json` | `synthetic_viewpoint.py` | Table II (designed viewpoint ablation) |

    python make_figs.py        # regenerates figs/*.pdf
    python verify_claims.py    # re-derives all 78 quoted statistics

`verify_claims.py` exits non-zero if any number in the paper stops matching the
artefacts. It currently passes on every quoted statistic except the two below.

## Regenerating the designed viewpoint ablation (Table II)

    python synthetic_viewpoint.py dataset --out synth/rot \
        --scales 1.0 --rotations 0,15,30,45 --tilts 0 \
        --methods sift orb --min-sharpness 10 --max-sources-per-wall 3
    python synthetic_viewpoint.py dataset --out synth/tilt \
        --scales 1.0 --rotations 0 --tilts 0,30,60 \
        --methods sift orb --min-sharpness 10 --max-sources-per-wall 3

Adding `registration` to `--methods` extends the ablation to the proposed
method; it was omitted here purely on cost (one full evaluation per bin per
source photograph, and registration is ~26 s per image pair).

## The one claim the paper deliberately does NOT make

Contribution 1 reports the annotation **provenance** (merge rate, bridged-gap
distribution, ARI against the pre-merge labelling) but **not** a sampled error
rate, a confidence interval, or inter-annotator agreement — because the human
audit has not been done. `verify_claims.py` asserts that no filled audit file
exists, so the paper and the repo cannot silently drift apart on this.

To complete it and upgrade the claim:

    python label_audit.py dataset --sample 30 --seed 0   # writes _audit_0.json + contact sheets
    #   ... a human fills in the `verdict` field for each sampled identity ...
    python label_audit.py dataset --score dataset/labels/_audit_0.json

Inter-annotator agreement additionally needs a second annotator to label a
subset into its own directory, then:

    python label_audit.py dataset --agree dataset/labels dataset/labels_b

Note `dataset/labels_old/` is the *pre-merge automatic output*, not a second
annotator; the 0.080 ARI the paper quotes measures how much the merge
restructured the partition, and is labelled as such.

## Two numbers that cannot be re-derived here

These were measured by this project but need the saved `.npz` score matrices,
which are not committed (they are large and regenerated per run):

* leave-one-wall-out spread, `CrackShape` R@1 `0.495 +/- 0.329` over 14 folds
* the censoring control, R@1 `0.422 -> 0.27` at 34% coverage

Reproduce with:

    python benchmark.py dataset --out banchmark_out --min-sharpness 10 --lowo
    python reid_analysis.py banchmark_out --split test --bootstrap 2000
