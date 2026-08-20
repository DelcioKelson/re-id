"""
Organize a folder of raw phone photos into the re-ID benchmark layout.

This is the step that comes BEFORE batch_segment.py and benchmark.py. It
takes the flat dump of camera files (crack_images/PXL_20260810_140646075.jpg)
and produces the layout benchmark.py's load_manifest() expects:

    dataset/
      walls.csv                  image_id, wall_id, session, path
      images/wall01_s1_0001.jpg  renamed copies of the raw photos
      masks/                     EMPTY - fill with batch_segment.py
      labels/wall01_s1_0001.json click points, one per physical crack
      splits.json                {"val": [...], "test": [...]}
      wall_map.csv               editable source of truth, see below
      sheets/wall01.jpg          contact sheet per wall, for labelling

--------------------------------------------------------------------------
WHAT CAN AND CANNOT BE DERIVED FROM FILENAMES
--------------------------------------------------------------------------
A Pixel filename (PXL_<date>_<time>.jpg) gives the capture instant, and
nothing else. From that we can derive, reliably:

  * the SESSION - photos share a date and a time-of-day.
  * the BURST   - a run of photos with no gap longer than --gap seconds.
    A burst is one continuous stand-in-front-of-a-surface-and-shoot,
    so it is a good FIRST GUESS at "these are the same wall".

What a timestamp can NEVER tell us is whether the wall shot on 10 Aug is
the same physical wall shot again on 17 Aug. That is precisely the link
the re-ID protocol scores, so it must come from a human. The bursts here
are a PROPOSAL written to wall_map.csv; you correct that file and re-run.

--------------------------------------------------------------------------
wall_map.csv - the editable source of truth
--------------------------------------------------------------------------
    source,wall_id,session,keep
    PXL_20260810_140646075.jpg,wall01,2026-08-10-pm,1
    PXL_20260810_140736336.jpg,wall02,2026-08-10-pm,1

First run writes it from the burst proposal. Every later run READS it and
ignores the proposal, so your edits survive re-running. To say "the wall I
shot on the 10th is the one I re-shot on the 17th", give both rows the
same wall_id and leave their sessions different -- that pair is then the
thing the benchmark actually measures. Set keep=0 to drop a photo
(blurred, mis-fire) without deleting it from crack_images/.

Re-running is safe: image_ids are assigned deterministically from
(wall_id, session, capture time), so a photo keeps its name as long as
its row does, and labels/ stay attached.

--------------------------------------------------------------------------
MATCHABILITY - read this before running the benchmark
--------------------------------------------------------------------------
benchmark.py evaluates with same_wall_only=True and exclude_same_session
=True. A query therefore only has a findable answer when ONE wall_id
appears under TWO different session strings. If every wall was shot in a
single session, the benchmark runs but scores nothing -- the summary
printed at the end of this script tells you how many matchable
identities you currently have, so you find that out here and not after a
GPU afternoon.

    pip install opencv-python numpy
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import defaultdict, OrderedDict
from datetime import datetime

import cv2
import numpy as np

# Camera filenames carry the capture instant: PXL_20260810_140646075.jpg
# (Pixel, with milliseconds), IMG_20260812_124045.jpg (Xiaomi, without).
# Match the date+time pair wherever it appears, milliseconds optional.
RAW_RE = re.compile(r"(?<!\d)(\d{8})[_\-]?(\d{6})(?!\d{4})")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# ===========================================================================
# 1. Read the raw folder
# ===========================================================================

def capture_time(path: str) -> datetime | None:
    """Capture instant, from the filename if it carries one, else EXIF,
    else the file mtime.

    Filename first: it is what survives copying off the phone, and it is
    the same string whatever tool wrote the file. EXIF second, because
    DateTimeOriginal is still the true capture instant when a filename
    has been sanitised. mtime last and reluctantly -- it is the COPY
    time on most transfers, so it only happens to be right when the
    transfer preserved it."""
    m = RAW_RE.search(os.path.basename(path))
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            pass                                  # 8+6 digits that aren't a date
    t = _exif_time(path)
    if t is not None:
        return t
    try:
        return datetime.fromtimestamp(os.path.getmtime(path))
    except OSError:
        return None


def _exif_time(path: str) -> datetime | None:
    """EXIF DateTimeOriginal (tag 36867), if Pillow is installed."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            raw = exif.get(36867) or exif.get(306)          # Original, else DateTime
            if not raw:
                ifd = exif.get_ifd(0x8769)                  # Exif sub-IFD
                raw = ifd.get(36867) if ifd else None
        if raw:
            return datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None
    return None


def list_raw(input_dir: str) -> list[tuple[datetime, str]]:
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
             if f.lower().endswith(IMAGE_EXTS)]
    dated = [(capture_time(p), p) for p in files]
    missing = [p for t, p in dated if t is None]
    if missing:
        print(f"WARNING: no timestamp for {len(missing)} file(s); they sort last")
    fallback = datetime.max
    return sorted(((t or fallback, p) for t, p in dated), key=lambda r: (r[0], r[1]))


# ===========================================================================
# 2. Propose wall_id / session
# ===========================================================================

def session_of(t: datetime) -> str:
    """Session string in the convention benchmark.py's docstring uses:
    date plus half-of-day, e.g. 2026-08-10-pm."""
    return f"{t:%Y-%m-%d}-{'am' if t.hour < 12 else 'pm'}"


def bursts(items: list[tuple[datetime, str]], gap: float) -> list[int]:
    """Assign each photo a burst index; a gap longer than `gap` seconds
    starts a new burst."""
    out, b = [], 0
    for i, (t, _) in enumerate(items):
        if i and (t - items[i - 1][0]).total_seconds() > gap:
            b += 1
        out.append(b)
    return out


def propose(items: list[tuple[datetime, str]], gap: float) -> list[dict]:
    """First-pass wall_map rows: one wall per burst, session per date+half."""
    idx = bursts(items, gap)
    rows = []
    for (t, path), b in zip(items, idx):
        rows.append(dict(source=os.path.basename(path),
                         wall_id=f"wall{b + 1:02d}",
                         session=session_of(t),
                         keep="1"))
    return rows


# ===========================================================================
# 3. wall_map.csv round-trip
# ===========================================================================

FIELDS = ["source", "wall_id", "session", "keep"]


def write_map(path: str, rows: list[dict]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_map(path: str) -> list[dict]:
    with open(path, newline="") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    for r in rows:
        r.setdefault("keep", "1")
    return rows


def _offset_walls(new: list[dict], existing: list[dict]) -> list[dict]:
    """Renumber freshly proposed walls to start past the highest existing one.

    propose() numbers bursts from wall01 over the WHOLE input folder, so on
    an incremental run its numbering overlaps the wall_ids already in
    wall_map.csv. Appending as-is would file a new surface under an old
    wall_id -- and because the two were shot on different days, that wall
    would then look multi-session and the benchmark would score a pair
    that is not the same wall at all. Silent, and worse than a crash, so
    new bursts always get fresh numbers; merge them back deliberately by
    editing wall_map.csv if they really are a revisit."""
    used = set()
    for r in existing:
        m = re.fullmatch(r"wall(\d+)", r["wall_id"])
        if m:
            used.add(int(m.group(1)))
    base = max(used) if used else 0
    order = {}
    for r in new:
        order.setdefault(r["wall_id"], len(order))
    for r in new:
        r["wall_id"] = f"wall{base + 1 + order[r['wall_id']]:02d}"
    return new


# ===========================================================================
# 4. Build the layout
# ===========================================================================

def assign_ids(rows: list[dict], times: dict[str, datetime]) -> list[dict]:
    """image_id = <wall_id>_s<n>_<nnnn>.

    The session number n is the rank of that session's FIRST photo within
    the wall, so s1 is always the earliest visit to that wall. The
    counter runs per (wall, session) in capture order. Both are functions
    of the wall_map row alone, so ids are stable across re-runs."""
    rows = [r for r in rows if str(r.get("keep", "1")).strip() not in ("0", "false", "False", "")]
    rows.sort(key=lambda r: (r["wall_id"], times[r["source"]], r["source"]))

    first_seen: dict[str, OrderedDict] = defaultdict(OrderedDict)
    for r in rows:
        first_seen[r["wall_id"]].setdefault(r["session"], None)
    sess_no = {w: {s: i + 1 for i, s in enumerate(sess)} for w, sess in first_seen.items()}

    counter: dict[tuple[str, str], int] = defaultdict(int)
    for r in rows:
        key = (r["wall_id"], r["session"])
        counter[key] += 1
        n = sess_no[r["wall_id"]][r["session"]]
        r["image_id"] = f"{r['wall_id']}_s{n}_{counter[key]:04d}"
    return rows


def build(rows: list[dict], input_dir: str, root: str, link: bool):
    img_dir = os.path.join(root, "images")
    for sub in ("images", "masks", "labels", "sheets"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)

    for r in rows:
        src = os.path.abspath(os.path.join(input_dir, r["source"]))
        dst = os.path.join(img_dir, r["image_id"] + ".jpg")
        r["path"] = f"images/{r['image_id']}.jpg"
        if os.path.lexists(dst):
            os.remove(dst)
        if link:
            os.symlink(src, dst)
        else:
            shutil.copy2(src, dst)

    with open(os.path.join(root, "walls.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "wall_id", "session", "path",
                                          "capture_session"])
        w.writeheader()
        for r in rows:
            w.writerow({"image_id": r["image_id"], "wall_id": r["wall_id"],
                        "session": r["eval_session"], "path": r["path"],
                        "capture_session": r["session"]})

    _prune_orphans(rows, root)


def _prune_orphans(rows: list[dict], root: str):
    """Delete images/ and sheets/ entries that no longer belong to any row.

    Without this, dropping a photo from wall_map.csv (or off disk) leaves a
    dangling symlink that walls.csv no longer mentions but Dataset would
    still stumble over if it were ever re-listed. Label files are NOT
    deleted -- a stale label is a few bytes and may hold clicks you want
    back if the photo returns."""
    keep_img = {r["image_id"] + ".jpg" for r in rows}
    img_dir = os.path.join(root, "images")
    for f in os.listdir(img_dir):
        if f not in keep_img:
            os.remove(os.path.join(img_dir, f))
    keep_sheet = {r["wall_id"] + ".jpg" for r in rows}
    sheet_dir = os.path.join(root, "sheets")
    if os.path.isdir(sheet_dir):
        for f in os.listdir(sheet_dir):
            if f not in keep_sheet:
                os.remove(os.path.join(sheet_dir, f))


def write_label_stubs(rows: list[dict], root: str):
    """One empty click-point file per photo, so the labelling tool and
    benchmark.py both find a well-formed file from the start. Never
    overwrites a file that already has points in it."""
    made = kept = 0
    for r in rows:
        p = os.path.join(root, "labels", r["image_id"] + ".json")
        if os.path.exists(p):
            try:
                with open(p) as f:
                    if json.load(f).get("points"):
                        kept += 1
                        continue
            except (json.JSONDecodeError, OSError):
                pass
        with open(p, "w") as f:
            json.dump({"image_id": r["image_id"], "points": []}, f, indent=2)
        made += 1
    return made, kept


def write_splits(rows: list[dict], root: str, val_frac: float, force: bool = False):
    """Split BY WALL, never by photo -- two photos of one wall on either
    side of the split would leak the answer. Only walls that appear in
    >=2 sessions can be scored, so those are dealt out first; the rest go
    to test as distractors."""
    path = os.path.join(root, "splits.json")
    if os.path.exists(path) and not force:
        print(f"  splits.json exists, left alone (--resplit to regenerate)")
        return
    sessions = defaultdict(set)
    for r in rows:
        sessions[r["wall_id"]].add(r["eval_session"])
    multi = sorted(w for w, s in sessions.items() if len(s) >= 2)
    single = sorted(w for w, s in sessions.items() if len(s) < 2)

    n_val = int(round(len(multi) * val_frac))
    val = multi[:n_val]
    test = multi[n_val:] + single
    with open(path, "w") as f:
        json.dump({"val": val, "test": test}, f, indent=2)


def write_sheets(rows: list[dict], root: str, width: int = 6):
    """A contact sheet per wall, captioned with image_id, so you can see a
    whole wall at once while deciding identities and click points."""
    by_wall = defaultdict(list)
    for r in rows:
        by_wall[r["wall_id"]].append(r)
    for wall, rs in sorted(by_wall.items()):
        rs = sorted(rs, key=lambda r: r["image_id"])
        cell_w, cell_h, pad = 260, 220, 22
        cols = min(width, len(rs))
        n_rows = (len(rs) + cols - 1) // cols
        sheet = np.full((n_rows * cell_h, cols * cell_w, 3), 255, np.uint8)
        for i, r in enumerate(rs):
            img = cv2.imread(os.path.join(root, r["path"]), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            s = min((cell_w - 10) / w, (cell_h - pad - 10) / h)
            thumb = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
            y, x = (i // cols) * cell_h, (i % cols) * cell_w
            sheet[y + pad:y + pad + thumb.shape[0], x + 5:x + 5 + thumb.shape[1]] = thumb
            cv2.putText(sheet, r["image_id"], (x + 5, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(root, "sheets", f"{wall}.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 85])


# ===========================================================================
# 5. Summary - the number that decides whether the benchmark can score
# ===========================================================================

def summarize(rows: list[dict], mode: str = "date") -> str:
    sessions = defaultdict(set)
    per_wall = defaultdict(int)
    for r in rows:
        sessions[r["wall_id"]].add(r["eval_session"])
        per_wall[r["wall_id"]] += 1
    multi = [w for w, s in sessions.items() if len(s) >= 2]

    if mode == "viewpoint":
        caps = defaultdict(set)
        for r in rows:
            caps[r["wall_id"]].add(r["session"])
        lines = [f"{len(rows)} photos, {len(sessions)} walls, "
                 f"session-mode=VIEWPOINT (each photo its own session)",
                 "Scoring axis is viewpoint WITHIN one pass, not time. "
                 "See README.md for the wording this obliges you to use.", ""]
        for w in sorted(sessions):
            n = per_wall[w]
            lines.append(f"  {w}: {n:3d} photos -> {n * (n - 1):4d} ordered same-wall "
                         f"pairs, all shot {sorted(caps[w])[0]}")
        return "\n".join(lines)

    lines = [f"{len(rows)} photos, {len(sessions)} walls, "
             f"{len({r['eval_session'] for r in rows})} sessions"]
    for w in sorted(sessions):
        flag = "  <-- multi-session (scorable)" if w in multi else ""
        lines.append(f"  {w}: {per_wall[w]:3d} photos, "
                     f"{len(sessions[w])} session(s) {sorted(sessions[w])}{flag}")
    lines.append("")
    if multi:
        lines.append(f"{len(multi)} wall(s) appear in >=2 sessions -- these are the "
                     f"only ones benchmark.py can score.")
    else:
        lines.append("NO wall appears in more than one session. benchmark.py will run "
                     "but every query\nwill have zero valid gallery matches "
                     "(same_wall_only + exclude_same_session).\nEdit wall_map.csv to "
                     "give the same wall_id to photos of the same physical wall taken\n"
                     "in different sessions, then re-run this script.")
    return "\n".join(lines)


def temporal_stats(rows: list[dict], times: dict[str, datetime]) -> dict:
    """How near-duplicate are the within-wall pairs?

    Under --session-mode viewpoint these two numbers are the honest
    caveat on every metric in the results table, so they get computed
    here and written into the README rather than left to be
    reconstructed at writing time."""
    per_wall = defaultdict(list)
    for r in rows:
        per_wall[r["wall_id"]].append(times[r["source"]])
    gaps, spans = [], []
    for ts in per_wall.values():
        ts = sorted(ts)
        spans.append((ts[-1] - ts[0]).total_seconds())
        gaps += [(b - a).total_seconds() for a, b in zip(ts, ts[1:])]
    gaps.sort()
    return {"median_gap": gaps[len(gaps) // 2] if gaps else 0.0,
            "max_span": max(spans) if spans else 0.0,
            "pairs": sum(len(v) * (len(v) - 1) for v in per_wall.values())}


README = """# Crack re-ID dataset

Generated by `build_dataset.py` from `crack_images/`. Layout matches
`benchmark.py`'s `load_manifest()`.

    walls.csv      image_id, wall_id, session, path, capture_session
    images/        renamed photos
    masks/         <- fill this in, step 2
    labels/        click points, one JSON per photo
    splits.json    val / test wall lists
    wall_map.csv   EDITABLE source of truth for wall_id / session / keep
    sheets/        one contact sheet per wall, for labelling

`session` is the axis the benchmark scores across; `capture_session` is
always the true capture date, kept so the two can never be confused.

## Session modes

`--session-mode date` (default) -- `session` is the capture date and
half-day. This is the real temporal re-ID protocol: a query is answered
by a photo of the same wall from a DIFFERENT DAY. It only scores if a
wall was actually revisited.

`--session-mode viewpoint` -- every photo is its own session, so
`exclude_same_session=True` drops just the query's own photo and scores
it against every other view of the same wall. Use this when no wall was
revisited. It measures a strictly weaker property, and the paper must
say so.

## What you may claim under --session-mode viewpoint

You measured: cross-VIEWPOINT retrieval within a single pass. Given a
crack seen from one camera position, find the same crack from a
different position, scale, and angle in the same walk-around.

You did NOT measure: re-identification across time. Every pair shares a
day, a lighting condition, and a weather state. Nothing here shows a
method survives a wall being re-inspected weeks later, which is the
deployment case.

Wording that is safe:

> Each wall was captured in a single continuous pass, so the evaluation
> measures viewpoint and scale invariance rather than temporal
> re-identification. Query and gallery images of a wall share
> illumination and were captured seconds apart (median MEDIAN_GAP s
> between consecutive frames, maximum span MAX_SPAN s per wall).
> Robustness to inter-session appearance change therefore remains
> untested, and we expect these figures to be an upper bound on
> multi-session performance.

Report the median-gap and span numbers -- `build_dataset.py` prints
them -- so a reader can judge how near-duplicate the pairs are. Do not
describe the split as sessions, visits, inspections, or dates.

## Steps

1. **Check the grouping.** Open `sheets/*.jpg`. Each sheet should be one
   physical wall. If a sheet mixes two walls, or one wall is split over
   two sheets, fix `wall_id` in `wall_map.csv` and re-run
   `build_dataset.py`.

2. **Segment.** Masks must be named for the image stem:

       python batch_segment.py --model ~/Downloads/best_model_v2.pth \\
           --input dataset/images --output dataset/masks --mode tile

3. **Label.** One click per distinct physical crack. Under viewpoint
   mode the identity must be reused across the views of that wall:

       python label_points.py dataset --wall wall01

4. **Run.**

       python benchmark.py dataset --methods sift orb registration
"""


# ===========================================================================
# 6. Driver
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="crack_images", help="folder of raw photos")
    ap.add_argument("--root", default="dataset", help="dataset root to create")
    ap.add_argument("--gap", type=float, default=20.0,
                    help="seconds without a photo that starts a new burst (default 20)")
    ap.add_argument("--val-frac", type=float, default=0.34,
                    help="fraction of scorable walls held out for threshold calibration")
    ap.add_argument("--link", action="store_true",
                    help="symlink images instead of copying (saves disk, less portable)")
    ap.add_argument("--session-mode", choices=["date", "viewpoint"], default="date",
                    help="date: session = capture date+half-day, the real temporal "
                         "protocol. viewpoint: every photo is its own session, so the "
                         "benchmark scores cross-VIEWPOINT retrieval within one pass "
                         "(weaker claim, see README).")
    ap.add_argument("--resplit", action="store_true",
                    help="regenerate splits.json (needed after changing --session-mode)")
    ap.add_argument("--prune-missing", action="store_true",
                    help="drop wall_map rows whose source file no longer exists")
    ap.add_argument("--repropose", action="store_true",
                    help="discard wall_map.csv and re-derive it from bursts")
    a = ap.parse_args()

    items = list_raw(a.input)
    if not items:
        raise FileNotFoundError(f"no images in {a.input}")
    times = {os.path.basename(p): t for t, p in items}

    os.makedirs(a.root, exist_ok=True)
    map_path = os.path.join(a.root, "wall_map.csv")

    if os.path.exists(map_path) and not a.repropose:
        rows = read_map(map_path)
        known = {r["source"] for r in rows}
        new = [r for r in propose(items, a.gap) if r["source"] not in known]
        if new:
            new = _offset_walls(new, rows)
            print(f"{len(new)} new photo(s) since last run, appended to wall_map.csv "
                  f"as {', '.join(sorted({r['wall_id'] for r in new}))}")
            rows += new
            write_map(map_path, rows)
        print(f"Using existing {map_path} ({len(rows)} rows)")
    else:
        rows = propose(items, a.gap)
        write_map(map_path, rows)
        print(f"Wrote proposal {map_path} ({len(rows)} rows, gap={a.gap}s)\n"
              f"  -> one wall per capture burst. REVIEW IT, see sheets/.")

    stale = [r["source"] for r in rows if r["source"] not in times]
    if stale and a.prune_missing:
        print(f"Dropping {len(stale)} wall_map row(s) whose file is gone from {a.input} "
              f"(e.g. {stale[0]})")
        gone = set(stale)
        rows = [r for r in rows if r["source"] not in gone]
        write_map(map_path, rows)
    elif stale:
        raise FileNotFoundError(
            f"{len(stale)} row(s) in wall_map.csv have no file in {a.input}, e.g. "
            f"{stale[0]}.\nRestore the files, or re-run with --prune-missing to drop "
            f"those rows,\nor --repropose to rebuild wall_map.csv from scratch.")
    if not rows:
        raise SystemExit("no photos left after pruning")

    rows = assign_ids(rows, times)
    for r in rows:
        # image_id is unique per photo, so in viewpoint mode every photo is its
        # own session and exclude_same_session drops only the query's own photo.
        r["eval_session"] = r["image_id"] if a.session_mode == "viewpoint" else r["session"]
    build(rows, a.input, a.root, a.link)
    made, kept = write_label_stubs(rows, a.root)
    write_splits(rows, a.root, a.val_frac, a.resplit)
    write_sheets(rows, a.root)
    stats = temporal_stats(rows, times)
    with open(os.path.join(a.root, "README.md"), "w") as f:
        f.write(README.replace("MEDIAN_GAP", f"{stats['median_gap']:.0f}")
                      .replace("MAX_SPAN", f"{stats['max_span']:.0f}"))

    print(f"\n{summarize(rows, a.session_mode)}")
    if a.session_mode == "viewpoint":
        print(f"\nCAVEAT NUMBERS for the paper (also written into README.md):"
              f"\n  median gap between consecutive frames of a wall: {stats['median_gap']:.0f}s"
              f"\n  longest span any single wall covers:            {stats['max_span']:.0f}s"
              f"\n  ordered same-wall pairs available to score:     {stats['pairs']}")
    print(f"\nlabels/: {made} empty stub(s), {kept} already labelled")
    print(f"Wrote {a.root}/  (images, masks/ EMPTY, labels, walls.csv, splits.json, sheets)")
    print("\nNext: segment into masks/, then label, then run benchmark.py. See "
          f"{os.path.join(a.root, 'README.md')}.")


if __name__ == "__main__":
    main()
