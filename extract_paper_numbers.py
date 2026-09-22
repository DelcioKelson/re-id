"""Extract the numbers the paper needs from a rerun's score matrices.

Two of the current review round's requests depend on values reid_analysis.py
already computes but does not persist: the WALL-level bootstrap interval on
DIR@FAR=0.1 (the abstract now quotes the point estimate only), and the
per-image-pair / per-wall amortised cost of registration+chamfer. This script
re-derives both from the saved matrices in <out>/scores/*.npz, sanity-checks
the point estimates against the committed result.txt (so no interval is
ever drawn from a matrix that no longer reproduces the published table), and
writes banchmark_out/dir_ci.json for verify_claims.py to check.

Usage (run where the rerun's matrices live):

    python extract_paper_numbers.py banchmark_out \\
        --hybrid hybrid_eval_out               # committed seed-0 table
        --hybrid hybrid_eval_out_s1            # extra seed(s), if run

Prints a copy-paste block for the paper and writes dir_ci.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reid_analysis import load_runs, masks_for, bootstrap_ci


def parse_results_table(path):
    rows, header = {}, None
    for line in open(path):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "method":
            header = parts
            continue
        if header is None or parts[0].startswith("-"):
            continue
        try:
            nums = [float(x) for x in parts[2:]]
        except ValueError:
            continue
        if len(nums) != len(header) - 2:
            continue
        rows[parts[0]] = dict(zip(header[2:], nums))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="dir holding scores/*.npz (e.g. banchmark_out)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--result-txt",
                    default="banchmark_out/result.txt")
    ap.add_argument("--write", default="banchmark_out/dir_ci.json")
    ap.add_argument("--hybrid", action="append", default=[],
                    help="hybrid_eval_out dir (table file) to print; give one per seed")
    args = ap.parse_args()

    runs = load_runs(args.out_dir, split=args.split)
    committed = parse_results_table(args.result_txt)
    reg_name = "registration+chamfer"
    reg = None
    for r in runs:
        if r.method == reg_name:
            reg = r
    if reg is None:
        print(f"FATAL: {reg_name} matrix not under {args.out_dir}/scores/")
        sys.exit(1)

    valid, relevant = masks_for(reg)
    thr_calib = None  # DIR@FAR is curve-based (FAR operating point), no val threshold
    from reid_analysis import _metrics
    m = _metrics(reg.scores, relevant, valid, threshold=None,
                 q_image=reg.query["image_id"], g_image=reg.gallery["image_id"])
    ciw = bootstrap_ci(reg.scores, relevant, valid, n_boot=args.n_boot,
                       seed=args.seed, cluster=reg.query["wall_id"])

    # --- sanity: this matrix must reproduce the committed table row ----------
    want = committed.get(reg_name, {})
    for col, field in (("DIR@FAR.1", "dir_at_far10"),
                       ("R@1", "rank1"), ("mAP", "mAP"),
                       ("scored", "scoreable_pair_rate")):
        if col in want and abs(want[col] - m[field]) > 5e-3:
            print(f"WARNING: matrix {field}={m[field]:.4f} vs committed {want[col]:.4f} "
                  f"-- interval below is NOT from the paper's numbers; investigate.")
        elif col in want:
            print(f"  sanity {field:<18} matrix {m[field]:.4f}  committed {want[col]:.4f}  OK")

    total = reg.prepare_seconds + reg.score_seconds
    cov = reg.coverage or {}
    n_img_pairs = cov.get("image_pairs")
    secs_per_pair = (total / n_img_pairs) if n_img_pairs else None

    print("\n" + "=" * 70)
    print("PAPER COPY-PASTE BLOCK (registration+chamfer, %s split)" % args.split)
    print("=" * 70)
    lo, hi = ciw.get("dir_at_far10", (float("nan"), float("nan")))
    rlo, rhi = ciw.get("rank1", (float("nan"), float("nan")))
    alo, ahi = ciw.get("mAP", (float("nan"), float("nan")))
    print(f"DIR@FAR=.1        {m['dir_at_far10']:.3f}   wall 95% CI [{lo:.3f},{hi:.3f}]")
    print(f"R@1               {m['rank1']:.3f}   wall 95% CI [{rlo:.3f},{rhi:.3f}]")
    print(f"mAP               {m['mAP']:.3f}   wall 95% CI [{alo:.3f},{ahi:.3f}]")
    print(f"n_queries         {m['n_queries']}")
    print(f"total_seconds     {total:.1f}")
    print(f"image_pairs       {n_img_pairs}")
    print(f"seconds_per_pair  {secs_per_pair:.1f}  (== total / image_pairs)")
    print(f"seconds_per_query {total / max(m['n_queries'], 1):.2f}")

    out = {
        "split": args.split,
        "n_queries": m["n_queries"],
        "dir_at_far10": m["dir_at_far10"],
        "ci_wall": {"dir_at_far10": [round(lo, 3), round(hi, 3)],
                    "rank1": [round(rlo, 3), round(rhi, 3)],
                    "mAP": [round(alo, 3), round(ahi, 3)]},
        "seconds_per_image_pair": round(secs_per_pair, 1) if secs_per_pair else None,
        "image_pairs": n_img_pairs,
        "total_seconds": round(total, 1),
    }

    # All methods: wall-level DIR CI, so the paper can say how much of the
    # open-set gap is a point estimate rather than a separable difference.
    print("\n--- wall 95% CI on DIR@FAR=.1 for every method ---")
    print(f"{'method':<34}{'DIR@.1':>8}{'wall CI':>16}{'scored':>8}")
    for r in sorted(runs, key=lambda x: x.method):
        v, rel = masks_for(r)
        mm = _metrics(r.scores, rel, v, threshold=None,
                      q_image=r.query["image_id"], g_image=r.gallery["image_id"])
        cw = bootstrap_ci(r.scores, rel, v, n_boot=args.n_boot,
                          seed=args.seed, cluster=r.query["wall_id"])
        if "dir_at_far10" in cw:
            dl, dh = cw["dir_at_far10"]
            print(f"{r.method:<34}{mm['dir_at_far10']:>8.3f}"
                  f"[{dl:.3f},{dh:.3f}]".rjust(16) +
                  f"{mm['scoreable_pair_rate']:>8.2f}")

    if args.write:
        os.makedirs(os.path.dirname(args.write) or ".", exist_ok=True)
        with open(args.write, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {args.write}")

    for d in args.hybrid:
        table = os.path.join(d, "comparasion_result.txt")
        if not os.path.isfile(table):
            table = os.path.join(d, "hybrid_eval_table.txt")
        print(f"\n--- hybrid table from {table} ---")
        print(open(table).read().strip() if os.path.isfile(table) else f"(missing {table})")


if __name__ == "__main__":
    main()