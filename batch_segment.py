"""
Batch crack-segmentation inference over a whole folder.

Reads every image in INPUT_DIR, runs the UNet++ crack segmenter, and
writes one binary mask per image into OUTPUT_DIR. Mask filenames match
the image STEM (photo.jpg -> photo.png), which is exactly the naming the
re-ID benchmark expects under masks/, so this drops straight into that
pipeline.

Two inference modes:

  * "resize"  (default, matches your current script): whole image scaled
    to IMG_SIZE, predicted, scaled back. Fast, but on a 12-megapixel wall
    photo a 2-3px crack becomes sub-pixel at 512 and can vanish.

  * "tile":  sliding window at native resolution. The image is covered by
    overlapping IMG_SIZE tiles, each predicted at the scale the model was
    trained on, and the probability maps are averaged where tiles overlap.
    Slower, but preserves thin cracks on large images. Use this for the
    real dataset; the mask quality feeds directly into re-ID accuracy.

Preprocessing (ImageNet mean/std) matches the training script. Do NOT
mix this with the divide-by-255-only variant from the old notebook — a
mismatch there silently degrades every downstream mask.

Optionally applies the shape-based post-filter (crack_postfilter.py) to
each mask before writing, removing the trim/moulding edges, paint lines,
and surface marks the segmenter fires on at native resolution — keeping
crack-only instances for re-ID. Toggle with POSTFILTER / --postfilter.
With --save-raw and --save-overlay you can dump the unfiltered mask and a
QA overlay alongside, to check the filter isn't eating real cracks.

    pip install torch segmentation-models-pytorch opencv-python numpy
"""

from __future__ import annotations

import os
import glob
import time

import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp


# =========================================================
# Configuration  (edit these, or override via argparse below)
# =========================================================

MODEL_PATH = "best_model_v2.pth"
INPUT_DIR = "/content/dataset/images"
OUTPUT_DIR = "/content/dataset/masks"

IMG_SIZE = 512
THRESHOLD = 0.5
MODE = "tile"          # "resize" or "tile"
TILE_OVERLAP = 0.25    # fraction of IMG_SIZE that adjacent tiles share (tile mode)
BATCH_SIZE = 8         # images per forward pass (resize mode) / tiles per pass (tile mode)
OVERWRITE = False      # re-generate masks that already exist

POSTFILTER = True      # drop trim edges / marks / blobs before writing the mask
SAVE_RAW = False       # also write the unfiltered mask to OUTPUT_DIR/raw/
SAVE_OVERLAY = False   # also write a crack-on-photo overlay to OUTPUT_DIR/overlays/

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =========================================================
# Model
# =========================================================

def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    checkpoint = torch.load(model_path, map_location=device)
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded checkpoint  epoch={checkpoint.get('epoch')}  best_f1={checkpoint.get('best_f1')}")
    return model


def _normalize(rgb: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 RGB -> CHW float32, ImageNet-normalized (matches training)."""
    x = rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)


def _overlay(bgr: np.ndarray, mask: np.ndarray, color=(0, 0, 255)) -> np.ndarray:
    out = bgr.copy()
    m = mask > 0
    out[m] = (0.45 * out[m] + 0.55 * np.array(color, np.float32)).astype(np.uint8)
    return out


# =========================================================
# Inference: resize mode
# =========================================================

@torch.no_grad()
def predict_resize(model, rgb: np.ndarray, device, img_size: int) -> np.ndarray:
    """Return a full-resolution probability map via single-shot resize."""
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(_normalize(small)).unsqueeze(0).to(device)
    prob = torch.sigmoid(model(t))[0, 0].cpu().numpy()
    return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)


# =========================================================
# Inference: tiled mode (native-resolution sliding window)
# =========================================================

def _tile_origins(length: int, tile: int, stride: int) -> list[int]:
    """Top-left coords covering [0, length) with the given tile/stride,
    always including a final tile flush to the edge."""
    if length <= tile:
        return [0]
    origins = list(range(0, length - tile + 1, stride))
    if origins[-1] != length - tile:
        origins.append(length - tile)
    return origins


@torch.no_grad()
def predict_tiled(model, rgb: np.ndarray, device, img_size: int,
                  overlap: float, batch_size: int) -> np.ndarray:
    """Sliding-window probability map at native resolution.

    Overlapping tiles are averaged (probability sum / coverage count) so
    there are no seams at tile boundaries. Images smaller than a tile are
    reflect-padded up to tile size and cropped back.
    """
    h, w = rgb.shape[:2]
    pad_b = max(0, img_size - h)
    pad_r = max(0, img_size - w)
    if pad_b or pad_r:
        rgb = cv2.copyMakeBorder(rgb, 0, pad_b, 0, pad_r, cv2.BORDER_REFLECT)
    H, W = rgb.shape[:2]

    stride = max(1, int(round(img_size * (1.0 - overlap))))
    ys = _tile_origins(H, img_size, stride)
    xs = _tile_origins(W, img_size, stride)

    prob_sum = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.float32)

    coords, batch = [], []

    def flush():
        if not batch:
            return
        t = torch.from_numpy(np.stack(batch)).to(device)
        p = torch.sigmoid(model(t))[:, 0].cpu().numpy()
        for (yy, xx), pm in zip(coords, p):
            prob_sum[yy:yy + img_size, xx:xx + img_size] += pm
            count[yy:yy + img_size, xx:xx + img_size] += 1.0
        coords.clear()
        batch.clear()

    for yy in ys:
        for xx in xs:
            tile = rgb[yy:yy + img_size, xx:xx + img_size]
            batch.append(_normalize(tile))
            coords.append((yy, xx))
            if len(batch) >= batch_size:
                flush()
    flush()

    prob = prob_sum / np.clip(count, 1e-6, None)
    return prob[:h, :w]


# =========================================================
# Folder driver
# =========================================================

def list_images(input_dir: str) -> list[str]:
    files = []
    for ext in IMAGE_EXTS:
        files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        files.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    return sorted(set(files))


def run_folder(model_path=MODEL_PATH, input_dir=INPUT_DIR, output_dir=OUTPUT_DIR,
               img_size=IMG_SIZE, threshold=THRESHOLD, mode=MODE,
               overlap=TILE_OVERLAP, batch_size=BATCH_SIZE, overwrite=OVERWRITE,
               postfilter=POSTFILTER, save_raw=SAVE_RAW, save_overlay=SAVE_OVERLAY):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    os.makedirs(output_dir, exist_ok=True)

    filter_fn = None
    if postfilter:
        try:
            from crack_postfilter import filter_crack_mask
            filter_fn = filter_crack_mask
        except ImportError:
            print("WARNING: crack_postfilter.py not found; writing unfiltered masks.\n")

    raw_dir = os.path.join(output_dir, "raw")
    overlay_dir = os.path.join(output_dir, "overlays")
    if save_raw:
        os.makedirs(raw_dir, exist_ok=True)
    if save_overlay:
        os.makedirs(overlay_dir, exist_ok=True)

    images = list_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {input_dir}")
    print(f"Found {len(images)} image(s) in {input_dir}  |  mode={mode}  "
          f"postfilter={'on' if filter_fn else 'off'}  device={device}\n")

    done = skipped = failed = 0
    t_start = time.time()

    for i, path in enumerate(images, 1):
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir, f"{stem}.png")

        if os.path.exists(out_path) and not overwrite:
            skipped += 1
            continue

        try:
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise IOError("cv2.imread returned None")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            if mode == "tile":
                prob = predict_tiled(model, rgb, device, img_size, overlap, batch_size)
            else:
                prob = predict_resize(model, rgb, device, img_size)

            mask = (prob >= threshold).astype(np.uint8) * 255

            raw_mask = mask
            if filter_fn is not None:
                mask = filter_fn(mask)          # drop trim edges / marks / blobs

            if save_raw:
                cv2.imwrite(os.path.join(raw_dir, f"{stem}.png"), raw_mask)
            cv2.imwrite(out_path, mask)
            if save_overlay:
                cv2.imwrite(os.path.join(overlay_dir, f"{stem}_overlay.jpg"),
                            _overlay(bgr, mask))
            done += 1

            frac = mask.mean() / 255.0
            note = ""
            if filter_fn is not None:
                raw_frac = raw_mask.mean() / 255.0
                if raw_frac > 0:
                    note = f"  (filtered {1 - frac / raw_frac:5.1%} of raw px)"
            print(f"[{i:>4}/{len(images)}] {stem:<40} crack px={frac:6.3%}{note}  -> {os.path.basename(out_path)}")
        except Exception as e:
            failed += 1
            print(f"[{i:>4}/{len(images)}] {stem:<40} FAILED: {type(e).__name__}: {e}")

    dt = time.time() - t_start
    print(f"\nDone in {dt:.1f}s  |  written={done}  skipped(existing)={skipped}  failed={failed}")
    print(f"Masks in {output_dir}/")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Batch crack segmentation over a folder.")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--input", default=INPUT_DIR)
    ap.add_argument("--output", default=OUTPUT_DIR)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--mode", choices=["resize", "tile"], default=MODE)
    ap.add_argument("--overlap", type=float, default=TILE_OVERLAP)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--overwrite", action="store_true", default=OVERWRITE)
    ap.add_argument("--postfilter", dest="postfilter", action="store_true", default=POSTFILTER,
                    help="drop trim edges / marks / blobs before writing (default on)")
    ap.add_argument("--no-postfilter", dest="postfilter", action="store_false",
                    help="write raw segmenter masks, unfiltered")
    ap.add_argument("--save-raw", action="store_true", default=SAVE_RAW,
                    help="also write unfiltered masks to OUTPUT/raw/")
    ap.add_argument("--save-overlay", action="store_true", default=SAVE_OVERLAY,
                    help="also write crack-on-photo overlays to OUTPUT/overlays/")
    a = ap.parse_args()
    run_folder(a.model, a.input, a.output, a.img_size, a.threshold,
               a.mode, a.overlap, a.batch_size, a.overwrite,
               a.postfilter, a.save_raw, a.save_overlay)