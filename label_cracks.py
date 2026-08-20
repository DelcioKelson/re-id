"""
Interactive crack-labelling tool.

Produces the  labels/<stem>.json  files the benchmark needs. You click one
point on each crack and give it an identity; the SAME physical crack must
get the SAME identity across every photo it appears in — that cross-photo
link is the ground truth the benchmark scores.

The tool helps you stay consistent: as you move between photos of one wall,
the sidebar shows every identity already used on that wall, colour-coded,
so you can reuse the right one instead of guessing. Clicks snap to the
nearest crack pixel, so a point always lands inside a mask component (which
is how the benchmark resolves point -> identity).

Runs locally with an OpenCV window (needs a desktop display — run it on
your machine, not in a headless Colab). Writes JSON directly into
labels/, and reloads existing labels so you can stop and resume.

    pip install opencv-python numpy

USAGE
    python label_cracks.py --root path/to/dataset --images Images --masks masks

CONTROLS
    left click     add a point on the crack under the cursor (snaps to mask),
                   tagged with the ACTIVE identity
    n              new identity (auto-named wallNN_crackMM) -> becomes active
    1..9           make that sidebar identity the active one
    z              undo last point on this image
    d / right      next image        a / left   previous image
    s              save this image now (also autosaves on nav / quit)
    q / Esc        save everything and quit
"""

from __future__ import annotations

import argparse
import json
import os
import re
import glob

import cv2
import numpy as np


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
_PALETTE = [(66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
            (171, 71, 188), (0, 172, 193), (255, 112, 67), (158, 157, 36),
            (120, 144, 156), (233, 30, 99)]


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without a display)
# ---------------------------------------------------------------------------

def parse_wall(stem: str) -> str:
    """wall id from a stem like 'wall03_s1_0007' -> 'wall03'. Falls back to
    the substring before the first underscore."""
    m = re.match(r"(wall\d+)", stem, re.IGNORECASE)
    return m.group(1) if m else stem.split("_")[0]


def discover(images_dir: str, masks_dir: str) -> list[dict]:
    """Pair every image with its same-stem mask. Images without a mask are
    still listed (you can label on the photo alone; clicks just won't snap)."""
    items = []
    paths = []
    for ext in IMAGE_EXTS:
        paths += glob.glob(os.path.join(images_dir, f"*{ext}"))
        paths += glob.glob(os.path.join(images_dir, f"*{ext.upper()}"))
    for p in sorted(set(paths)):
        stem = os.path.splitext(os.path.basename(p))[0]
        mask_p = os.path.join(masks_dir, f"{stem}.png")
        items.append({"stem": stem, "wall": parse_wall(stem),
                      "image": p, "mask": mask_p if os.path.exists(mask_p) else None})
    return items


def snap_to_mask(mask: np.ndarray | None, x: int, y: int, radius: int = 25):
    """Move (x,y) to the nearest crack pixel within `radius`. Keeps points
    inside a mask component so the benchmark resolves them exactly. Returns
    the original point if there's no mask or no crack pixel nearby."""
    if mask is None:
        return x, y
    h, w = mask.shape[:2]
    x0, y0 = max(x - radius, 0), max(y - radius, 0)
    x1, y1 = min(x + radius + 1, w), min(y + radius + 1, h)
    window = mask[y0:y1, x0:x1]
    ys, xs = np.nonzero(window > 0)
    if len(xs) == 0:
        return x, y
    d2 = (xs + x0 - x) ** 2 + (ys + y0 - y) ** 2
    k = int(np.argmin(d2))
    return int(xs[k] + x0), int(ys[k] + y0)


def label_path(labels_dir: str, stem: str) -> str:
    return os.path.join(labels_dir, f"{stem}.json")


def load_points(labels_dir: str, stem: str) -> list[dict]:
    p = label_path(labels_dir, stem)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return json.load(f).get("points", [])


def save_points(labels_dir: str, stem: str, points: list[dict]) -> None:
    os.makedirs(labels_dir, exist_ok=True)
    with open(label_path(labels_dir, stem), "w") as f:
        json.dump({"image_id": stem, "points": points}, f, indent=2)


def color_for(identity: str, order: list[str]):
    if identity not in order:
        order.append(identity)
    return _PALETTE[order.index(identity) % len(_PALETTE)]


def next_identity_name(wall: str, used: set[str]) -> str:
    n = 1
    while f"{wall}_crack{n:02d}" in used:
        n += 1
    return f"{wall}_crack{n:02d}"


# ---------------------------------------------------------------------------
# Rendering (pure: returns a canvas; testable headless)
# ---------------------------------------------------------------------------

def render_canvas(image_bgr, mask, points, id_order, active_id, stem, wall,
                  wall_identities, sidebar_w=280, max_h=900):
    img = image_bgr.copy()
    if mask is not None:
        m = mask > 0
        tint = img.copy(); tint[m] = (0.4 * tint[m] + 0.6 * np.array([0, 0, 255], np.float32)).astype(np.uint8)
        img = tint

    for pt in points:
        c = color_for(pt["identity"], id_order)
        x, y = pt["xy"]
        cv2.circle(img, (x, y), 9, c, -1)
        cv2.circle(img, (x, y), 9, (255, 255, 255), 2)
        cv2.putText(img, pt["identity"].split("_crack")[-1], (x + 12, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

    h, w = img.shape[:2]
    scale = min(1.0, max_h / h)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    h, w = img.shape[:2]

    panel = np.full((h, sidebar_w, 3), 30, np.uint8)
    cv2.putText(panel, stem, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    cv2.putText(panel, f"{wall}  |  {len(points)} pts", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(panel, "identities on this wall:", (10, 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    y = 110
    for i, ident in enumerate(wall_identities, 1):
        c = color_for(ident, id_order)
        cv2.rectangle(panel, (10, y - 14), (30, y + 2), c, -1)
        tag = f"[{i}] {ident}" + ("  <-" if ident == active_id else "")
        cv2.putText(panel, tag, (38, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255) if ident == active_id else (200, 200, 200), 1)
        y += 26
    cv2.putText(panel, "n:new  z:undo  a/d:prev/next", (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)
    cv2.putText(panel, "s:save  q:quit", (10, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)

    return np.hstack([img, panel]), scale


# ---------------------------------------------------------------------------
# Interactive session
# ---------------------------------------------------------------------------

def run(root: str, images_sub: str, masks_sub: str, labels_sub: str):
    images_dir = os.path.join(root, images_sub)
    masks_dir = os.path.join(root, masks_sub)
    labels_dir = os.path.join(root, labels_sub)

    items = discover(images_dir, masks_dir)
    if not items:
        raise FileNotFoundError(f"No images in {images_dir}")

    # identities already used, per wall (for the sidebar + reuse)
    wall_ids: dict[str, list[str]] = {}
    for it in items:
        for pt in load_points(labels_dir, it["stem"]):
            wall_ids.setdefault(it["wall"], [])
            if pt["identity"] not in wall_ids[it["wall"]]:
                wall_ids[it["wall"]].append(pt["identity"])

    print(f"{len(items)} image(s) across {len(set(i['wall'] for i in items))} wall(s). "
          f"Writing labels to {labels_dir}/")

    state = {"idx": 0, "active": None, "id_order": []}
    points_cache: dict[str, list[dict]] = {}

    def current():
        return items[state["idx"]]

    def get_points(stem):
        if stem not in points_cache:
            points_cache[stem] = load_points(labels_dir, stem)
        return points_cache[stem]

    def flush(stem):
        if stem in points_cache:
            save_points(labels_dir, stem, points_cache[stem])

    win = "label_cracks"
    cv2.namedWindow(win)

    render_state = {"scale": 1.0, "mask": None}

    def on_mouse(event, mx, my, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        it = current()
        img_w = int(cv2.getWindowImageRect(win)[2]) if False else None  # not reliable; use scale
        s = render_state["scale"]
        # click must be on the image area (left of sidebar)
        ox, oy = int(mx / s), int(my / s)
        mask = render_state["mask"]
        if mask is not None and (ox >= mask.shape[1] or oy >= mask.shape[0]):
            return
        if state["active"] is None:
            state["active"] = next_identity_name(it["wall"], set(sum(wall_ids.values(), [])))
            wall_ids.setdefault(it["wall"], []).append(state["active"])
        sx, sy = snap_to_mask(mask, ox, oy)
        get_points(it["stem"]).append({"identity": state["active"], "xy": [sx, sy]})

    cv2.setMouseCallback(win, on_mouse)

    while True:
        it = current()
        image = cv2.imread(it["image"], cv2.IMREAD_COLOR)
        mask = cv2.imread(it["mask"], cv2.IMREAD_GRAYSCALE) if it["mask"] else None
        if mask is not None and image is not None and mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        render_state["mask"] = mask

        pts = get_points(it["stem"])
        wl = wall_ids.setdefault(it["wall"], [])
        for pt in pts:
            if pt["identity"] not in wl:
                wl.append(pt["identity"])
        if state["active"] is None and wl:
            state["active"] = wl[-1]

        canvas, scale = render_canvas(image, mask, pts, state["id_order"],
                                      state["active"], it["stem"], it["wall"], wl)
        render_state["scale"] = scale
        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), 27):
            flush(it["stem"]); break
        elif key in (ord("d"), 83, 84):
            flush(it["stem"]); state["idx"] = (state["idx"] + 1) % len(items); state["active"] = None
        elif key in (ord("a"), 81, 82):
            flush(it["stem"]); state["idx"] = (state["idx"] - 1) % len(items); state["active"] = None
        elif key == ord("z") and pts:
            pts.pop()
        elif key == ord("s"):
            flush(it["stem"]); print(f"saved {it['stem']} ({len(pts)} pts)")
        elif key == ord("n"):
            new_id = next_identity_name(it["wall"], set(sum(wall_ids.values(), [])))
            wl.append(new_id); state["active"] = new_id
        elif ord("1") <= key <= ord("9"):
            i = key - ord("1")
            if i < len(wl):
                state["active"] = wl[i]

    for stem in points_cache:
        flush(stem)
    cv2.destroyAllWindows()
    total = sum(len(v) for v in points_cache.values())
    print(f"Done. {total} point(s) saved across {len(points_cache)} image(s).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Label cracks for the re-ID benchmark.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--images", default="Images")
    ap.add_argument("--masks", default="masks")
    ap.add_argument("--labels", default="labels")
    a = ap.parse_args()
    run(a.root, a.images, a.masks, a.labels)
