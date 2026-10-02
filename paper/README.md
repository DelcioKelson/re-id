# Paper: *CrackID: A Within-Visit Benchmark for Crack Re-Identification*

LaTeX source for the IEEE conference submission built from this repository.

    pdflatex main && bibtex main && pdflatex main && pdflatex main

## Where every number comes from

Every quantitative figure and table is regenerated from committed artefacts.
The dataset example figure uses committed photographs and labels; its identity
examples inherit homography-based annotation and are not an independent audit.
Table I is a
direct copy of `../banchmark_out/result.txt`, and `verify_claims.py` now
*parses that file* and asserts each quoted row against it -- previously the
ten rows were hardcoded in the verifier, so the one link never checked was
the one from benchmark output to published table. The counting-units table
(`tab:units`) and the positive-pair/merge counts have no artefact of their own
-- they are properties of the validity mask -- so the verifier re-derives them
from `../dataset` through `benchmark.valid_mask` rather than trusting a CSV.

| Artefact | Produced by | Used for |
|---|---|---|
| `../banchmark_out/result.txt` | `benchmark.py` | Table I, Fig. 1, amortised cost |
| `../banchmark_out/pair_outcomes.json` | `viewpoint.py` | Fig. 2, coverage, gate sweep, clustered logistic |
| `../banchmark_out/viewpoint.json` | `viewpoint.py` | Fig. 3, viewpoint stats, logistic fit |
| `../banchmark_out/chance.json` | `chance_baseline.py` | chance levels, the frame-gap sweep, the coverage confound |
| `../hybrid_eval_out/comparasion_result.txt` | `hybrid_eval.py` | Table (Sec. VI-H, synthetic revisit benchmark) |
| `../dataset/quality.json` | `image_quality.py` | gate sweep, Fig. 2 ordering |
| `../dataset/labels/*.json` | `label_points.py` | dataset composition (Sec. IV-A) |
| `../dataset/images/*.jpg` | `make_figs.py` | representative image crops (Sec. III-A) |
| `../dataset/labels/_changelog.json` | `prefill_labels.py` | merge provenance, Sec. IV-C |
| `../dataset/labels/_triage.json` | `prefill_labels.py` | the 57 widest-gap merges |
| `../dataset/labels_old/` | `prefill_labels.py` | pre-merge labelling, for the ARI |

    python ../chance_baseline.py ../dataset --out ../banchmark_out
    python make_figs.py        # regenerates figs/*.pdf
    python verify_claims.py    # re-derives every quoted statistic

`verify_claims.py` exits non-zero if any number in the paper stops matching the
artefacts. Benchmark-output checks ([0] Table I copy, [1] chance floors, [3]-[5]
coverage/viewpoint/logistic, [7]-[9] capture/synthetic/hybrid) currently pass.
Dataset-composition checks ([2] identities/points, [6] changelog merge/gap counts)
currently FAIL: `dataset/labels/` has been re-edited since the benchmarked
snapshot (walls 18--19 added, changelog rebuilt with `max_merge_gap=None`,
audit template created), while `banchmark_out/` still holds the older run
(`nQ=280`). The paper text now pins its counts to that snapshot (abstract,
Sec. IV-A, Table II caption). Re-run `benchmark.py` + `prefill/rebuild` with gap
computation to make the tree green again; until then the failures are the
version pin working. Mask-dependent units ([2b]/[2c] density) SKIP cleanly when
`cv2` is unavailable. Audit check counts filled `verdict`s, not files, so the
empty `_audit_0.json` template passes. The amortised cost in
Sec. VI-H (`7817.3`\,s total over `301` test image pairs = `26.0`\,s/pair) is
re-derived from `result.txt` and asserted by the verifier.

The wall-level bootstrap interval on `DIR@FAR` was deliberately *not* quoted:
it requires the saved score matrices, which are not committed, so the abstract
and Sec. VIII state the open-set numbers as point estimates on a 19-identity
sample rather than inventing an unreproduceable interval. If the GPU rerun is
ever done, `extract_paper_numbers.py` computes and commits that interval to
`banchmark_out/dir_ci.json`; until then the paper's honesty comes from the
hedge, not from a number.

## The one claim the paper deliberately does NOT make

Contribution 1 reports the annotation **provenance** (merge rate, bridged-gap
distribution, ARI against the pre-merge labelling) but **not** a sampled error
rate, a confidence interval, or inter-annotator agreement — because the human
audit has not been done. `verify_claims.py` asserts that no filled audit verdict
exists (empty `_audit_0.json` template is allowed), so the paper and the repo cannot silently drift apart on this.

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

* leave-one-wall-out spread in Rank-1
* the censoring control, R@1 `0.422 -> 0.27` at 34% coverage

Reproduce with:

    python benchmark.py dataset --out banchmark_out --min-sharpness 10 --lowo
    python reid_analysis.py banchmark_out --split test --bootstrap 2000
