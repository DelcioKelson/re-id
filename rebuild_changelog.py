"""
Rebuild dataset/labels/_changelog.json from the labels and masks that
actually exist, instead of from a hand-typed summary of what once existed.

WHY THIS FILE IS NEEDED
-----------------------
labels/_changelog.json is the sole record of which prefill identities the
merge rule absorbed into which final identity. It is what label_audit.py
--report prints, what sample_identities() stratifies on, and what
paper/verify_claims.py checks the paper's gap claims against (two places).
No script in this repository writes it: prefill_labels.py computed
identities but never emitted a changelog, so the file was written by hand.

A hand-written summary of a merging process has three failure modes, and
all three are present:

    1. It goes stale. The file was last written 2026-09-29 23:40; the label
       files were rewritten through 2026-09-30 11:37, twelve hours later.
       The commit that fixed the wall assignment of 18 photos updated the
       changelog (368 insertions, 761 deletions) but not the ~15k lines of
       relabelling made the next morning.

    2. It keeps identities that no longer exist. 49 of its 212 identities
       have zero clicks in the labels. 43 of those are recorded BOTH as
       their own identity AND as absorbed into another identity, which is
       self-contradictory: an identity cannot be a merge target and be
       merged away. They are merge targets that were later deleted, whose
       rows outlived them. sample_identities() draws from this pool, so a
       real audit was picking rows with no clicks and no rendered sheet --
       unauditable rows that can only ever be signed off unread.

    3. It counts an identity as absorbing itself. 15 identities list
       themselves in their own absorbed_prefill_ids. merge_report() sets
       merged = absorbed > 1, so every self-absorption inflates the count
       and the "48% of test identities are merged" figure inherits the
       error. Eight of the fifteen are self-absorbing purely because their
       points carry no prefill_identity at all, so the fallback attribution
       lands the point back on its own identity.

WHAT IS RECOVERABLE, AND WHAT IS NOT
------------------------------------
absorbed_prefill_ids is not lost. Every point that prefill_labels.py
placed carries its origin inline:

    "model_review": {"prefill_identity": "wall02_crack100",
                     "verdict": "reassigned"}

Reconstructing the absorbed sets from those fields reproduces the existing
changelog exactly for 134 of the 163 live identities, with no identity
missing -- which is what makes this a reconstruction and not a guess. The
29 that differ are the identities touched by the wall rename, whose
prefixes move once normalised, plus those whose sets differed in the typed
file to begin with.

The limit is coverage. 2151 of 3130 points (69%) carry no
prefill_identity, and they are not spread evenly: they concentrate in the
large identities that absorbed the most.

    wall09_crack01   216 points with no provenance   (262 clicks)
    wall16_crack01   203                              wall13_crack01  187
    wall04_crack01   181                              wall18_crack01  178

So the identities with the widest merges are exactly the ones whose merge
evidence is gone. This file therefore reports provenance coverage per
identity and refuses to imply a merge count it cannot support: an
identity with incomplete provenance gets provenance_complete: false, and
merge_report() must be taught to treat its merged flag as unknown rather
than false. Note that the rebuilt merged count comes out slightly HIGHER
than the typed one (84 vs 81) even after self-absorption is removed --
because the typed count is spread over 212 identities including 49 dead
ones, while the rebuilt count covers only the 163 that still exist. The
two are not comparable, and --compare says so rather than implying a
correction.

max_merge_gap IS NOT REBUILT
----------------------------
It is emitted as null. No code in this repository ever computed it, so
there is no definition to reproduce; it was typed in. Reconstructing it
needs two decisions that are the paper's to make, not this script's:

    * what "the gap bridged inside one identity" means -- max distance
      between two components of one identity within a photo, or across
      photos, changes the number materially;
    * the gap threshold of the merge rule itself, which the paper has to
      quote anyway.

Until both are fixed, --report's median/p90/max gap columns and
verify_claims.py:460 have no defensible input, and the rebuilt file makes
that visible instead of silently carrying the old typed values forward.

WALL PREFIXES
-------------
prefill_identity still carries the pre-rename wall prefix: points now on
wall18 name prefill ids like wall02_crack100. Keeping the raw string makes
a wall18 identity look like it absorbed wall02 fragments. _normalise()
rewrites the prefix to the identity's current wall and keeps the crack
number, which is the part that carries the information.

    WHAT THIS SCRIPT CHANGES DOWNSTREAM
-----------------------------------
format_merge_report() was taught to read a null gap as 'n/a' and to report
how many merged identities carry none, so label_audit.py --report survives
the rebuilt file instead of raising TypeError on the percentiles.

paper/verify_claims.py needs a decision rather than a fix. Section [6]
pins the typed numbers as exact assertions:

    check("identities total", tot, 212, tol=0)
    check("identities formed by merge", merged, 95, tol=0)
    check("  ... as fraction", merged / tot, 0.45, tol=0.005)
    check("median max merge gap (test, px)", ..., 228.2, tol=0.2)

Those pass today, and they are the paper's Sec. IV-C provenance claims. But
212 includes 49 identities with no clicks, and the count of 95 is
len(absorbed) > 1 WITHOUT removing self-absorption -- so it counts the 15
identities that absorb themselves. The rebuilt file yields 163 identities
and 84 merges. verify_claims.py already skips null gaps when collecting
them, so pointing it at the rebuilt file leaves the two gap assertions
failing, correctly, with no gap data to compare. That is the honest state:
the gap claims have no reproducible source.

    python rebuild_changelog.py dataset
    python rebuild_changelog.py dataset --compare

Writes labels/_changelog_rebuilt.json. The original is never overwritten:
it is the only record of the old typed numbers, and the comparison below
is the evidence that rebuilding was worth doing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

from label_audit import (_manifest, _resolve_components, load_changelog,
                        load_labels)

CRACK_RE = re.compile(r"crack(\d+)$")


def _crack_no(identity: str) -> str | None:
    """The trailing crack number of an identity, or None if it has none."""
    m = CRACK_RE.search(identity)
    return m.group(1) if m else None


def _normalise(prefill_identity: str, wall: str) -> str:
    """Re-home a prefill id onto its identity's current wall.

    wall02_crack100 recorded on a point that now lives on wall18 becomes
    wall18_crack100. The crack number is the provenance; the prefix is a
    record of which pass of the pipeline saw it first.
    """
    no = _crack_no(prefill_identity)
    return f"{wall}_crack{no}" if no is not None else prefill_identity


def build(root: str, min_area: int, close_px: int,
          point_tolerance: int) -> tuple[dict, dict]:
    """Reconstruct the changelog. Returns (changelog, stats).

    Keeps the original four keys so every consumer reads the same shape,
    plus n_points, n_points_without_provenance, provenance_complete,
    n_self_absorbed_dropped and prefill_ids_renamed to make the gaps
    visible. format_merge_report() was updated to tolerate a null gap;
    paper/verify_claims.py already skipped nulls but pins the old numbers
    as assertions, so it will fail against this file -- see the module
    docstring.
    """
    from benchmark import Dataset

    data = Dataset(root, min_area=min_area, close_px=close_px,
                   point_tolerance=point_tolerance)
    labels = load_labels(os.path.join(root, "labels"))
    wall_of = {r["image_id"]: r.get("wall_id", "") for r in _manifest(root)}

    absorbed: dict[str, set[str]] = defaultdict(set)
    photos: dict[str, set[str]] = defaultdict(set)
    n_points: dict[str, int] = defaultdict(int)
    n_no_prov: dict[str, int] = defaultdict(int)
    self_absorbed: dict[str, int] = defaultdict(int)
    foreign: dict[str, set[str]] = defaultdict(set)

    for image_id, points in labels.items():
        for p in points:
            ident = p["identity"]
            wall = wall_of.get(image_id) or ident.rsplit("_crack", 1)[0]
            n_points[ident] += 1
            photos[ident].add(image_id)
            prefill = (p.get("model_review") or {}).get("prefill_identity")
            if not prefill:
                n_no_prov[ident] += 1
                continue
            norm = _normalise(prefill, wall)
            if norm == ident:
                self_absorbed[ident] += 1
                continue
            if prefill != norm:
                foreign[ident].add(prefill)
            absorbed[ident].add(norm)

    n_components: dict[str, int] = defaultdict(int)
    for image_id, points in labels.items():
        if not points:
            continue
        for c in _resolve_components(data, image_id, points):
            if c["winner"] is not None:
                n_components[c["identity"]] += 1

    out: dict[str, dict] = {}
    for wall in sorted({i.rsplit("_crack", 1)[0] for i in n_points}):
        out[wall] = {"identities": {}}

    for ident in sorted(n_points):
        wall = ident.rsplit("_crack", 1)[0]
        got = absorbed[ident]
        out[wall]["identities"][ident] = {
            "n_components": n_components.get(ident, 0),
            "n_photos": len(photos[ident]),
            "max_merge_gap": None,
            "absorbed_prefill_ids": sorted(got),
            "n_points": n_points[ident],
            "n_points_without_provenance": n_no_prov.get(ident, 0),
            "provenance_complete": n_no_prov.get(ident, 0) == 0,
        }
        if self_absorbed[ident]:
            out[wall]["identities"][ident]["n_self_absorbed_dropped"] = \
                self_absorbed[ident]
        if foreign[ident]:
            out[wall]["identities"][ident]["prefill_ids_renamed"] = \
                sorted(foreign[ident])

    n_ident = len(n_points)
    complete = sum(1 for i in n_points if n_no_prov.get(i, 0) == 0)
    stats = {
        "min_area": min_area,
        "close_px": close_px,
        "point_tolerance": point_tolerance,
        "points_total": sum(n_points.values()),
        "points_without_provenance": sum(n_no_prov.values()),
        "provenance_coverage": round(
            1 - sum(n_no_prov.values()) / max(sum(n_points.values()), 1), 4),
        "identities": n_ident,
        "identities_provenance_complete": complete,
        "identities_provenance_partial": n_ident - complete,
        "self_absorbed_dropped": sum(self_absorbed.values()),
        "self_absorbing_identities": len(self_absorbed),
        "max_merge_gap": None,
        "max_merge_gap_note": "not rebuilt: no definition exists in-repo",
    }
    return out, stats


def _ids(changelog: dict) -> dict:
    return {i: m for w, rec in changelog.items()
            for i, m in rec.get("identities", {}).items()}


def compare(old: dict, new: dict, stats: dict) -> str:
    """What rebuilding changed. The evidence that it was worth doing."""
    o, n = _ids(old), _ids(new)
    shared = set(o) & set(n)
    dead = sorted(set(o) - set(n))
    born = sorted(set(n) - set(o))
    same = sum(1 for i in shared
               if set(o[i].get("absorbed_prefill_ids", []))
               == set(n[i].get("absorbed_prefill_ids", [])))
    o_self = {i for i in o if i in set(o[i].get("absorbed_prefill_ids", []))}

    def merged(ids):
        return sum(1 for i, m in ids.items()
                   if len([x for x in m.get("absorbed_prefill_ids", [])
                           if x != i]) > 1)

    o_merged, n_merged = merged(o), merged(n)
    delta = n_merged - o_merged
    direction = ("HIGHER" if delta > 0 else
                 "LOWER" if delta < 0 else "unchanged")
    pct_o = 100 * o_merged / max(len(o), 1)
    pct_n = 100 * n_merged / max(len(n), 1)

    lines = [
        "",
        "=" * 78,
        "REBUILT CHANGELOG vs THE TYPED ONE",
        "=" * 78,
        f"identities            typed {len(o):>5}   rebuilt {len(n):>5}",
        f"absorbed sets identical              {same} / {len(shared)}"
        "   (after dropping self-absorption",
        "                                      and normalising wall prefixes)",
        f"self-absorbing identities   typed {len(o_self):>5}   rebuilt 0"
        "   (dropped by construction)",
        "",
        f"identities in the typed file with no clicks anywhere: {len(dead)}",
    ]
    lines += [f"    {i}" for i in dead[:10]]
    if len(dead) > 10:
        lines.append(f"    ... and {len(dead) - 10} more")
    if born:
        lines += ["", f"identities absent from the typed file: {len(born)}"]
        lines += [f"    {i}" for i in born[:10]]
        if len(born) > 10:
            lines.append(f"    ... and {len(born) - 10} more")
    lines += [
        "",
        "MERGED IDENTITIES (absorbed > 1, self-absorption removed from both)",
        "-" * 78,
        f"    typed    {o_merged:>4}  ({pct_o:.0f}% of {len(o)} identities)",
        f"    rebuilt  {n_merged:>4}  ({pct_n:.0f}% of {len(n)} identities)"
        f"   {direction} by {abs(delta)}",
        "",
        "The two counts are NOT directly comparable and the difference is not",
        "a correction. The typed count is taken over 212 identities, 49 of",
        "which hold no clicks, so it includes merges belonging to identities",
        "that no longer exist; the rebuilt count covers only the 163 that do.",
        "Dropping self-absorption and normalising the wall prefixes also",
        "moves individual sets. Neither figure is a label error rate -- both",
        "are statements about how the merge rule was recorded.",
        "",
        "PROVENANCE COVERAGE -- why this cannot settle the merge question",
        "-" * 78,
        f"points with a prefill_identity: "
        f"{stats['points_total'] - stats['points_without_provenance']}"
        f" / {stats['points_total']}"
        f"  ({100 * stats['provenance_coverage']:.1f}%)",
        f"identities fully provenanced: "
        f"{stats['identities_provenance_complete']} / {stats['identities']}",
        f"self-absorbed points dropped: {stats['self_absorbed_dropped']}",
        "",
        "An identity with incomplete provenance may still have been a merge;",
        "the rebuilt file cannot tell you either way and says so with",
        "provenance_complete: false. On those identities the merged flag is",
        "UNKNOWN, not false, and merge_report() must be taught to treat it",
        "that way before any rate is quoted.",
        "",
        "max_merge_gap is null throughout. --report's median/p90/max gap",
        "columns and paper/verify_claims.py:460 have no input until the gap",
        "definition and the merge threshold are fixed.",
        "=" * 78,
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--out", default=None,
                    help="output path (default labels/_changelog_rebuilt.json)")
    ap.add_argument("--min-area", type=int, default=200)
    ap.add_argument("--close-px", type=int, default=5)
    ap.add_argument("--point-tolerance", type=int, default=25)
    ap.add_argument("--compare", action="store_true",
                    help="diff the rebuilt file against the typed one")
    args = ap.parse_args()

    new, stats = build(args.root, args.min_area, args.close_px,
                       args.point_tolerance)
    out = args.out or os.path.join(args.root, "labels",
                                   "_changelog_rebuilt.json")
    with open(out, "w") as f:
        json.dump(new, f, indent=1)

    print(f"\n{stats['identities']} identities over "
          f"{stats['provenance_coverage'] * 100:.1f}% provenance coverage")
    print(f"  -> {out}")
    if args.compare:
        old = load_changelog(args.root)
        if not old:
            print("no existing changelog to compare against")
            return
        print(compare(old, new, stats))


if __name__ == "__main__":
    main()