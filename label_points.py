"""
Click-point labeller for the crack re-ID dataset.

benchmark.py identifies cracks by CLICK POINTS, not component indices, so
that labels survive re-tuning the segmenter (see its module docstring).
This is the tool that produces them:

    labels/wall01_s1_0001.json
    {
      "image_id": "wall01_s1_0001",
      "points": [
        {"identity": "wall01_crack01", "xy": [412, 690]},
        {"identity": "wall01_crack02", "xy": [980, 240]}
      ]
    }

--------------------------------------------------------------------------
THE ONE RULE
--------------------------------------------------------------------------
The same physical crack gets the same identity string in EVERY photo it
appears in, above all across sessions. `wall01_crack03` in a photo from
10 Aug and `wall01_crack03` in a photo from 17 Aug is exactly the pair
the benchmark scores. Get that wrong and the numbers are meaningless;
leave a crack unlabelled and it becomes a distractor, which is fine and
in fact useful -- you do not need to label everything, you need the
labels you do place to be right.

Work one wall at a time (--wall wall01) and keep its contact sheet
(dataset/sheets/wall01.jpg) open beside this, so you can see which crack
is which across the whole wall before committing a number.

Points can also arrive pre-guessed via prefill_labels.py, which propagates
identities across a wall's photos with homography matching. Those show up
dashed/orange with a "?" and "provisional": true in the JSON -- review
each one (fix wrong ones the usual way: right-click delete, then
left-click the right spot under the right crack number) and press `a` to
accept the rest for that photo. Never trust a provisional point you have
not looked at.

--------------------------------------------------------------------------
CONTROLS
--------------------------------------------------------------------------
  left click      place a point for the CURRENT identity
  right click     delete the nearest point (within 40 px on screen)
  [  /  ]         current crack number down / up
  0-9             set crack number directly
  n  /  p         next / previous photo   (saves first)
  m               toggle the segmentation mask overlay
  a               accept all provisional (prefilled) points in this photo
  c               clear all points in this photo
  s               save now
  q  /  Esc       save and quit

The point must land INSIDE the crack for the exact resolution path in
benchmark.py; press `m` to show the mask and click on a red pixel. If
you miss, it still resolves to the nearest component within
Dataset.point_tolerance (25 px), but inside is better.

    pip install opencv-python numpy
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import cv2
import numpy as np


def component_anchors(mask: np.ndarray, min_area: int = 200,
                      close_px: int = 5) -> list[tuple[int, int]]:
    """One click target per crack component, in original-image pixels.

    Mirrors extract_crack_instances(): same closing and same min_area, so
    the components offered here are exactly the ones benchmark.py will
    resolve a point against -- snapping to one of these guarantees the
    click lands inside a real instance rather than 3 px off it.

    The anchor is the component's deepest pixel by distance transform, i.e.
    the most solidly-interior point, which for a 2 px-wide crack is the
    only place a click reliably lands on the crack at all."""
    binary = (mask > 0).astype(np.uint8)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    dt = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    out = []
    for lid in range(1, n):
        if stats[lid, cv2.CC_STAT_AREA] < min_area:
            continue
        sel = labels == lid
        d = np.where(sel, dt, -1.0)
        y, x = np.unravel_index(int(np.argmax(d)), d.shape)
        out.append((int(x), int(y)))
    return out


def load_rows(root: str) -> list[dict]:
    with open(os.path.join(root, "walls.csv")) as f:
        return list(csv.DictReader(f))


class Labeller:
    def __init__(self, root: str, rows: list[dict], max_side: int = 1100,
                 min_area: int = 200, close_px: int = 5, snap: bool = True):
        self.root = root
        self.rows = rows
        self.max_side = max_side
        self.min_area = min_area
        self.close_px = close_px
        self.snap = snap
        self.i = 0
        self.crack_no = 1
        self.show_mask = False
        self.dirty = False
        self.points: list[dict] = []
        self.scale = 1.0
        self.disp: np.ndarray | None = None
        self._load()

    # ---- io ----
    @property
    def row(self) -> dict:
        return self.rows[self.i]

    def _label_path(self) -> str:
        return os.path.join(self.root, "labels", self.row["image_id"] + ".json")

    def _load(self):
        r = self.row
        img = cv2.imread(os.path.join(self.root, r["path"]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(r["path"])
        h, w = img.shape[:2]
        self.scale = min(1.0, self.max_side / max(h, w))
        self.base = cv2.resize(img, (int(w * self.scale), int(h * self.scale)))

        self.mask_small = None
        self.anchors: list[tuple[int, int]] = []
        mpath = os.path.join(self.root, "masks", r["image_id"] + ".png")
        if os.path.exists(mpath):
            m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                if m.shape[:2] != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                self.mask_small = cv2.resize(
                    m, (self.base.shape[1], self.base.shape[0]), interpolation=cv2.INTER_NEAREST)
                self.anchors = component_anchors(m, self.min_area, self.close_px)

        p = self._label_path()
        self.points = []
        if os.path.exists(p):
            try:
                with open(p) as f:
                    self.points = json.load(f).get("points", [])
            except (json.JSONDecodeError, OSError):
                pass
        self.dirty = False

    def save(self):
        if not self.dirty:
            return
        os.makedirs(os.path.join(self.root, "labels"), exist_ok=True)
        with open(self._label_path(), "w") as f:
            json.dump({"image_id": self.row["image_id"], "points": self.points}, f, indent=2)
        self.dirty = False

    # ---- identity ----
    def identity(self) -> str:
        return f"{self.row['wall_id']}_crack{self.crack_no:02d}"

    # ---- rendering ----
    def render(self) -> np.ndarray:
        out = self.base.copy()
        if self.show_mask and self.mask_small is not None:
            m = self.mask_small > 0
            out[m] = (0.45 * out[m] + 0.55 * np.array((0, 0, 255), np.float32)).astype(np.uint8)
        claimed = {tuple(pt["xy"]) for pt in self.points}
        for (ax, ay) in self.anchors:
            sx, sy = int(ax * self.scale), int(ay * self.scale)
            if (ax, ay) in claimed:
                continue
            cv2.circle(out, (sx, sy), 7, (200, 120, 0), 1, cv2.LINE_AA)
        for pt in self.points:
            x, y = int(pt["xy"][0] * self.scale), int(pt["xy"][1] * self.scale)
            cur = pt["identity"] == self.identity()
            provisional = pt.get("provisional", False)
            if provisional:
                col = (0, 165, 255)
                marker = cv2.MARKER_DIAMOND
            else:
                col = (0, 255, 255) if cur else (0, 200, 0)
                marker = cv2.MARKER_CROSS
            cv2.drawMarker(out, (x, y), col, marker, 18, 2)
            cv2.circle(out, (x, y), 11, col, 1, cv2.LINE_AA)
            label = pt["identity"].split("_crack")[-1] + ("?" if provisional else "")
            cv2.putText(out, label, (x + 13, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)

        n_lab = sum(1 for r in self.rows
                    if self._has_points(r))
        n_prov = sum(1 for pt in self.points if pt.get("provisional"))
        bar = [f"[{self.i + 1}/{len(self.rows)}] {self.row['image_id']}"
               f"  {self.row['session']}",
               f"current: {self.identity()}   points {len(self.points)}"
               f"/{len(self.anchors)} comp"
               f"{'' if self.snap else '  SNAP OFF'}"
               f"   photos labelled: {n_lab}/{len(self.rows)}"
               f"{f'   {n_prov} unreviewed' if n_prov else ''}"
               f"{'   *unsaved' if self.dirty else ''}",
               "click=add(snaps)  rclick=del  [/] 0-9=crack no  n/p=photo  m=mask  "
               "g=snap  a=accept  c=clear  q=quit"]
        pad = np.full((22 * len(bar) + 8, out.shape[1], 3), 32, np.uint8)
        for k, line in enumerate(bar):
            cv2.putText(pad, line, (8, 18 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (235, 235, 235), 1, cv2.LINE_AA)
        return np.vstack([pad, out])

    def _has_points(self, r: dict) -> bool:
        if r["image_id"] == self.row["image_id"]:
            return bool(self.points)
        p = os.path.join(self.root, "labels", r["image_id"] + ".json")
        if not os.path.exists(p):
            return False
        try:
            with open(p) as f:
                return bool(json.load(f).get("points"))
        except (json.JSONDecodeError, OSError):
            return False

    # ---- mouse ----
    def on_mouse(self, event, x, y, flags, param):
        y -= self.bar_h                       # canvas is [status bar; image]
        if y < 0:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            ox, oy = int(round(x / self.scale)), int(round(y / self.scale))
            if self.snap and self.anchors:
                ax, ay = min(self.anchors, key=lambda a: (a[0] - ox) ** 2 + (a[1] - oy) ** 2)
                # only snap if the anchor is near the click on SCREEN, so a
                # deliberate click on an unsegmented crack still stands
                if np.hypot((ax - ox) * self.scale, (ay - oy) * self.scale) <= 60:
                    ox, oy = ax, ay
            self.points.append({"identity": self.identity(), "xy": [ox, oy]})
            self.dirty = True
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            d = [np.hypot(p["xy"][0] * self.scale - x, p["xy"][1] * self.scale - y)
                 for p in self.points]
            j = int(np.argmin(d))
            if d[j] <= 40:
                self.points.pop(j)
                self.dirty = True

    # ---- loop ----
    def run(self):
        win = "crack labeller"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, self.on_mouse)
        while True:
            canvas = self.render()
            self.bar_h = canvas.shape[0] - self.base.shape[0]
            cv2.imshow(win, canvas)
            k = cv2.waitKey(20) & 0xFF
            if k in (ord("q"), 27):
                self.save()
                break
            elif k == ord("n"):
                self.save()
                self.i = (self.i + 1) % len(self.rows)
                self._load()
            elif k == ord("p"):
                self.save()
                self.i = (self.i - 1) % len(self.rows)
                self._load()
            elif k == ord("]"):
                self.crack_no += 1
            elif k == ord("["):
                self.crack_no = max(1, self.crack_no - 1)
            elif ord("0") <= k <= ord("9"):
                self.crack_no = max(1, k - ord("0"))
            elif k == ord("m"):
                self.show_mask = not self.show_mask
            elif k == ord("g"):
                self.snap = not self.snap
            elif k == ord("a"):
                if any(pt.get("provisional") for pt in self.points):
                    for pt in self.points:
                        pt.pop("provisional", None)
                    self.dirty = True
            elif k == ord("c"):
                if self.points:
                    self.points = []
                    self.dirty = True
            elif k == ord("s"):
                self.save()
        cv2.destroyAllWindows()


def report(root: str, rows: list[dict]):
    """What the labels currently buy you: identities seen in >=2 sessions
    are the only ones with a findable answer under the benchmark's
    same-wall / different-session protocol."""
    from collections import defaultdict
    seen = defaultdict(lambda: defaultdict(set))
    n_pts = 0
    n_prov = 0
    for r in rows:
        p = os.path.join(root, "labels", r["image_id"] + ".json")
        if not os.path.exists(p):
            continue
        try:
            with open(p) as f:
                pts = json.load(f).get("points", [])
        except (json.JSONDecodeError, OSError):
            continue
        for pt in pts:
            n_pts += 1
            if pt.get("provisional"):
                n_prov += 1
            seen[r["wall_id"]][pt["identity"]].add(r["session"])
    total = 0
    print(f"{n_pts} click point(s) over {len(rows)} photo(s)"
          f"{f'  ({n_prov} unreviewed prefill -- benchmark will use them as-is)' if n_prov else ''}")
    for wall in sorted(seen):
        multi = sum(1 for s in seen[wall].values() if len(s) >= 2)
        total += multi
        print(f"  {wall}: {len(seen[wall])} identities, {multi} in >=2 sessions")
    print(f"TOTAL matchable identities (queries with an answer): {total}")
    if total == 0:
        print("  -> benchmark.py will score nothing. Either no wall spans two sessions "
              "(fix wall_map.csv\n     and re-run build_dataset.py) or no identity has "
              "been clicked in two of them yet.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="dataset root (contains walls.csv)")
    ap.add_argument("--wall", nargs="*", help="only label these wall_ids")
    ap.add_argument("--report", action="store_true",
                    help="print label coverage and exit, no window")
    ap.add_argument("--max-side", type=int, default=1100, help="display size")
    ap.add_argument("--min-area", type=int, default=200,
                    help="must match benchmark.py's Dataset(min_area=...)")
    ap.add_argument("--close-px", type=int, default=5,
                    help="must match benchmark.py's Dataset(close_px=...)")
    ap.add_argument("--no-snap", action="store_true",
                    help="do not snap clicks onto crack components")
    a = ap.parse_args()

    rows = load_rows(a.root)
    if a.wall:
        rows = [r for r in rows if r["wall_id"] in set(a.wall)]
    if not rows:
        raise SystemExit("no photos match")
    rows.sort(key=lambda r: r["image_id"])

    if a.report:
        report(a.root, rows)
        return
    Labeller(a.root, rows, a.max_side, a.min_area, a.close_px,
             snap=not a.no_snap).run()
    report(a.root, rows)


if __name__ == "__main__":
    main()
