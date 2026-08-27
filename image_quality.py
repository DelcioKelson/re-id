"""
Capture admission: which photographs a deployed system would accept.

WHY THIS EXISTS
---------------
The benchmark's claim is invariance to VIEWPOINT. It is not a claim about
image quality -- a homography is a viewpoint tool, and robustness to
defocus would be incidental rather than designed. But dropping quality
from the claim does not remove it from the data, and in this dataset it is
the dominant variable in every outcome the paper reports:

    predictor of a registration failure   median(reg)  median(fail)  logistic z
    minimum sharpness of the pair             220           16          +6.8
    frame gap between the two photos            3            3          -2.3

Sharpness outweighs elapsed baseline by roughly a factor of three, and at
wall level the pattern is a threshold rather than a gradient: walls at
median sharpness 14-33 register on 0% of their pairs while a wall at 220
registers on 94%. A paper claiming viewpoint invariance whose principal
failure mode is predicted by defocus has not isolated the variable it
claims to study.

The fix is NOT to widen the claim back to two axes. It is to admit only
captures a deployed system would accept -- the same acquisition-validation
stage a vehicle-inspection pipeline runs before it will score anything --
so that the claim becomes precise:

    invariant to viewpoint GIVEN AN ADMISSIBLE CAPTURE.

The gate is a pre-registered deployment rule, applied identically to every
method, and the number of rejected captures is reported. That is what
separates an admission rule from a convenient filter.

CHOOSING THE THRESHOLD
----------------------
`python image_quality.py dataset --sweep` prints images kept, walls kept,
pairs and (given a pair-outcome file) the registration rate at each gate.
The rule for picking one:

  * it must keep every wall -- a gate that deletes walls is selecting the
    surfaces the method likes;
  * it must discard ZERO successful registrations -- every pair it removes
    should be one no method could have answered anyway.

On this dataset a gate at 25 satisfies both. Gates at 100+ start deleting
walls and must be avoided.

MEASURE
-------
Variance of the Laplacian on quarter-scale grayscale. Quarter scale so the
number is comparable across the two handset resolutions and not dominated
by sensor noise; variance-of-Laplacian because it is the standard,
parameter-free defocus measure and a reviewer will recognise it.

    python image_quality.py dataset                  # per-image + per-wall
    python image_quality.py dataset --sweep          # gate sweep
    python image_quality.py dataset --gate 25        # what a gate admits
    python image_quality.py dataset --json out/quality.json
"""

from __future__ import annotations

import csv
import json
import os

import cv2
import numpy as np

# The pre-registered gate. Anything that reads a gate should default to
# this constant rather than hard-coding a number, so the paper, the
# benchmark and the analysis can never quote different thresholds.
DEFAULT_GATE = 25.0

SCALE = 0.25          # quarter-scale, as reported in the paper


def sharpness(gray_or_bgr: np.ndarray, scale: float = SCALE) -> float:
    """Variance of the Laplacian at `scale`. Higher is sharper.

    Resizing FIRST is not cosmetic: at full resolution the measure is
    dominated by per-pixel sensor noise, which is largest on the darker,
    smoother walls -- exactly the ones we are trying to detect as
    unusable. Quarter scale also makes 12 MP and 8 MP captures
    comparable, which matters because the dataset has two handsets.
    """
    g = gray_or_bgr
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


def _manifest(root: str) -> list[dict]:
    with open(os.path.join(root, "walls.csv")) as f:
        return list(csv.DictReader(f))


def sharpness_table(root: str, cache: str | None = None,
                    quiet: bool = False) -> dict[str, float]:
    """{image_id: sharpness} for every photo in the manifest.

    Cached to JSON because it is read by benchmark.py on every run and by
    reid_analysis.py on every report; recomputing it means decoding 140
    full-resolution JPEGs for a number that cannot change.
    """
    cache = cache or os.path.join(root, "quality.json")
    rows = _manifest(root)
    have = {}
    if os.path.exists(cache):
        with open(cache) as f:
            have = json.load(f).get("sharpness", {})
    todo = [r for r in rows if r["image_id"] not in have]
    for i, r in enumerate(todo):
        img = cv2.imread(os.path.join(root, r["path"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        have[r["image_id"]] = round(sharpness(img), 3)
        if not quiet:
            print(f"\r  sharpness {i + 1}/{len(todo)}", end="", flush=True)
    if todo and not quiet:
        print()
    if todo:
        with open(cache, "w") as f:
            json.dump({"measure": "variance_of_laplacian",
                       "scale": SCALE, "sharpness": have}, f, indent=1)
    return {r["image_id"]: have.get(r["image_id"], float("nan")) for r in rows}


def admissible(root: str, gate: float = DEFAULT_GATE,
               cache: str | None = None) -> set[str]:
    """Image ids at or above the gate. A NaN (unreadable) is not admitted."""
    tab = sharpness_table(root, cache=cache, quiet=True)
    return {k for k, v in tab.items() if np.isfinite(v) and v >= gate}


def admission_report(root: str, gate: float = DEFAULT_GATE,
                     cache: str | None = None) -> dict:
    """Everything the paper has to state about the gate in one dict."""
    tab = sharpness_table(root, cache=cache, quiet=True)
    rows = _manifest(root)
    wall_of = {r["image_id"]: r["wall_id"] for r in rows}
    kept = {k for k, v in tab.items() if np.isfinite(v) and v >= gate}
    rejected = sorted(set(tab) - kept)
    walls_all = sorted(set(wall_of.values()))
    walls_kept = sorted({wall_of[k] for k in kept})
    return {
        "gate": float(gate),
        "measure": "variance_of_laplacian@quarter_scale",
        "images_total": len(tab),
        "images_kept": len(kept),
        "images_rejected": len(rejected),
        "rejected_ids": rejected,
        "walls_total": len(walls_all),
        "walls_kept": len(walls_kept),
        "walls_emptied": sorted(set(walls_all) - set(walls_kept)),
        "per_wall_kept": {w: sum(1 for k in kept if wall_of[k] == w) for w in walls_all},
        "per_wall_total": {w: sum(1 for k in tab if wall_of[k] == w) for w in walls_all},
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _split_walls(root: str) -> dict:
    p = os.path.join(root, "splits.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def per_wall(root: str, cache: str | None = None) -> str:
    tab = sharpness_table(root, cache=cache)
    rows = _manifest(root)
    splits = _split_walls(root)
    which = {}
    for name, walls in splits.items():
        for w in walls:
            which[w] = name
    by_wall: dict[str, list[float]] = {}
    for r in rows:
        v = tab.get(r["image_id"])
        if v is not None and np.isfinite(v):
            by_wall.setdefault(r["wall_id"], []).append(v)

    L = ["", "SHARPNESS BY WALL  (variance of Laplacian, quarter-scale grey)",
         f"{'wall':<9}{'split':<7}{'n':>4}{'median':>9}{'min':>8}{'max':>8}   verdict"]
    L.append("-" * len(L[-1]))
    for w in sorted(by_wall):
        v = np.array(by_wall[w])
        med = float(np.median(v))
        verdict = ("unusable" if med < DEFAULT_GATE else
                   "marginal" if med < 65 else "usable")
        L.append(f"{w:<9}{which.get(w, '-'):<7}{len(v):>4}{med:>9.1f}"
                 f"{v.min():>8.1f}{v.max():>8.1f}   {verdict}")
    L += ["",
          f"'unusable' is below the pre-registered gate of {DEFAULT_GATE:g}; 'marginal' is "
          "above the gate but",
          "below 65, where registration succeeds only on textured surfaces."]
    return "\n".join(L)


def sweep(root: str, gates=(0, 10, 25, 50, 100, 200),
          pair_outcomes: str | None = None, split: str | None = "test",
          cache: str | None = None) -> str:
    """Gate sweep. With `pair_outcomes` (written by benchmark.py as
    <out>/pair_outcomes.json: {"wallXX_a|wallXX_b": true/false}) it also
    reports how many image pairs survive the gate and what fraction of
    them registered -- which is the table that justifies the threshold."""
    tab = sharpness_table(root, cache=cache, quiet=True)
    rows = _manifest(root)
    wall_of = {r["image_id"]: r["wall_id"] for r in rows}
    splits = _split_walls(root)
    keep_walls = set(splits.get(split, [])) if split else None
    ids = [k for k in tab if keep_walls is None or wall_of[k] in keep_walls]

    outcomes = {}
    if pair_outcomes and os.path.exists(pair_outcomes):
        with open(pair_outcomes) as f:
            outcomes = json.load(f).get("registered", {})

    L = ["", f"GATE SWEEP  (split={split or 'all'})",
         f"{'gate':<10}{'images':>8}{'walls':>7}{'pairs':>8}{'registered':>12}{'rate':>7}"]
    L.append("-" * len(L[-1]))
    for g in gates:
        kept = [k for k in ids if np.isfinite(tab[k]) and tab[k] >= g]
        walls = len({wall_of[k] for k in kept})
        keptset = set(kept)
        if outcomes:
            live = {k: v for k, v in outcomes.items()
                    if all(p in keptset for p in k.split("|"))}
            n_pairs, n_reg = len(live), sum(1 for v in live.values() if v)
            rate = f"{n_reg / n_pairs:.0%}" if n_pairs else "-"
            cells = f"{n_pairs:>8}{n_reg:>12}{rate:>7}"
        else:
            same = sum(1 for i, a in enumerate(kept) for b in kept[i + 1:]
                       if wall_of[a] == wall_of[b])
            cells = f"{same:>8}{'-':>12}{'-':>7}"
        L.append(f"{('none' if g <= 0 else f'>= {g:g}'):<10}{len(kept):>8}{walls:>7}{cells}")
    L += ["",
          "Take the loosest gate that keeps every wall AND discards no successful",
          "registration. A gate that empties a wall is selecting surfaces, not captures."]
    if not outcomes:
        L += ["", "(no --pairs file: 'pairs' counts same-wall photo pairs, and the",
              " registration columns need benchmark.py's pair_outcomes.json)"]
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("root")
    ap.add_argument("--sweep", action="store_true", help="gate sweep table")
    ap.add_argument("--gate", type=float, default=None,
                    help="report what this gate admits")
    ap.add_argument("--split", default="test", help="'test', 'val' or 'all'")
    ap.add_argument("--pairs", default=None,
                    help="benchmark_out/pair_outcomes.json, for the registration columns")
    ap.add_argument("--json", default=None, help="also write the table here")
    args = ap.parse_args()

    split = None if args.split == "all" else args.split
    print(per_wall(args.root))
    if args.sweep:
        print(sweep(args.root, pair_outcomes=args.pairs, split=split))
    if args.gate is not None:
        rep = admission_report(args.root, gate=args.gate)
        print(f"\nGATE {rep['gate']:g}: keeps {rep['images_kept']}/{rep['images_total']} "
              f"images over {rep['walls_kept']}/{rep['walls_total']} walls")
        if rep["walls_emptied"]:
            print(f"  !! empties walls {rep['walls_emptied']} -- too aggressive")
        print(f"  rejects: {', '.join(rep['rejected_ids']) or 'nothing'}")
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"sharpness": sharpness_table(args.root, quiet=True),
                       "admission": admission_report(args.root,
                                                     args.gate or DEFAULT_GATE)}, f, indent=1)
        print(f"\nwrote {args.json}")
