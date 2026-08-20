"""
Tiled (sliding-window) crack segmentation inference.

The single-resize approach — downscale a 3000px photo to 512, predict,
upscale the mask — throws away exactly the signal the model needs: a
2-4px crack becomes sub-pixel at 512 and vanishes. This runs the model
on 512x512 TILES at the image's native resolution (the same scale the
model saw during training, which cropped 512 windows from full-res
images), then blends the overlapping tile predictions with a cosine
weight window so there are no seams.

Correctness property (tested): because each tile is normalised and
predicted from pixel content alone, overlapping tiles agree on shared
pixels, and the weighted average reproduces a per-pixel-consistent
probability map — tiling changes speed and memory, not the answer.

    pip install torch segmentation-models-pytorch opencv-python numpy
"""

from __future__ import annotations

import os
import numpy as np
import cv2


# ===========================================================================
# Core sliding-window machinery  (pure numpy; model-agnostic)
# ===========================================================================

def _cosine_window(h: int, w: int) -> np.ndarray:
    """Separable raised-cosine (Hann) weights, ~1 at centre, ->0 at edges.
    A small floor keeps every pixel's total weight strictly positive so
    the final divide is safe even at tile corners."""
    wy = np.hanning(h + 2)[1:-1] if h > 1 else np.ones(1)
    wx = np.hanning(w + 2)[1:-1] if w > 1 else np.ones(1)
    win = np.outer(wy, wx).astype(np.float32)
    return np.clip(win, 1e-3, None)


def sliding_window_predict(image_rgb: np.ndarray, forward_fn,
                           tile: int = 512, overlap: int = 128,
                           batch_size: int = 4) -> np.ndarray:
    """Return a full-resolution probability map in [0, 1].

    image_rgb : HxWx3 uint8, native resolution (do NOT pre-resize).
    forward_fn: callable(list[HxWx3 uint8 tile]) -> list[HxW float prob].
                This is where the real model + ImageNet normalisation
                lives; kept as a callback so the tiling is testable
                without torch.
    tile      : window size — must match the model's training crop size.
    overlap   : overlap between adjacent tiles; more overlap = smoother
                blend and more compute. 128 (25% of 512) is a good default.
    """
    H, W = image_rgb.shape[:2]
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("overlap must be smaller than tile")

    # Pad up so the tiling covers every pixel; reflect (not zero) so we
    # don't paint a black — crack-coloured — border the model could
    # hallucinate cracks along.
    pad_h = (tile - H % stride) % stride if H > tile else max(tile - H, 0)
    pad_w = (tile - W % stride) % stride if W > tile else max(tile - W, 0)
    # ensure at least a full tile fits
    Hp, Wp = H + pad_h, W + pad_w
    if Hp < tile: pad_h += tile - Hp
    if Wp < tile: pad_w += tile - Wp
    padded = cv2.copyMakeBorder(image_rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    Hp, Wp = padded.shape[:2]

    weight = _cosine_window(tile, tile)
    prob_acc = np.zeros((Hp, Wp), np.float32)
    wsum_acc = np.zeros((Hp, Wp), np.float32)

    # Enumerate tile origins (inclusive of the bottom/right edge).
    tops = list(range(0, Hp - tile + 1, stride))
    lefts = list(range(0, Wp - tile + 1, stride))
    if tops[-1] != Hp - tile: tops.append(Hp - tile)
    if lefts[-1] != Wp - tile: lefts.append(Wp - tile)

    coords, tiles = [], []
    def flush():
        if not tiles:
            return
        preds = forward_fn(tiles)
        for (t, l), p in zip(coords, preds):
            prob_acc[t:t + tile, l:l + tile] += p * weight
            wsum_acc[t:t + tile, l:l + tile] += weight
        coords.clear(); tiles.clear()

    for t in tops:
        for l in lefts:
            tiles.append(padded[t:t + tile, l:l + tile])
            coords.append((t, l))
            if len(tiles) >= batch_size:
                flush()
    flush()

    prob = prob_acc / wsum_acc
    return prob[:H, :W]


# ===========================================================================
# Torch model wrapper  (the real forward_fn)
# ===========================================================================

def build_forward_fn(model, device, mean=(0.485, 0.456, 0.406),
                     std=(0.229, 0.224, 0.225)):
    """Wrap a loaded smp model as forward_fn for sliding_window_predict.
    Normalisation MUST match training — these are the ImageNet stats the
    training pipeline used."""
    import torch
    mean = np.array(mean, np.float32)
    std = np.array(std, np.float32)

    def forward_fn(tiles):
        batch = np.stack([((t.astype(np.float32) / 255.0) - mean) / std for t in tiles])
        batch = torch.from_numpy(batch.transpose(0, 3, 1, 2)).float().to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(batch))[:, 0].cpu().numpy()
        return [probs[i] for i in range(probs.shape[0])]

    return forward_fn


def load_model(model_path: str, device):
    import torch
    import segmentation_models_pytorch as smp
    model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights=None,
                             in_channels=3, classes=1)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded checkpoint  epoch={ckpt.get('epoch')}  best_f1={ckpt.get('best_f1')}")
    return model


def segment_image(image_path: str, model, device, tile: int = 512,
                  overlap: int = 128, threshold: float = 0.5,
                  tta: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Full-resolution crack mask for one photo. Returns (prob, mask_uint8)."""
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    forward_fn = build_forward_fn(model, device)
    prob = sliding_window_predict(rgb, forward_fn, tile=tile, overlap=overlap)

    if tta:   # average over horizontal/vertical flips — cheap accuracy bump
        for flip in (1, 0):
            f = cv2.flip(rgb, flip)
            pf = sliding_window_predict(f, forward_fn, tile=tile, overlap=overlap)
            prob = prob + cv2.flip(pf, flip)
        prob /= 3.0

    mask = (prob >= threshold).astype(np.uint8) * 255
    return prob, mask


# ===========================================================================
# Batch a whole dataset  (writes masks/<image_id>.png for the benchmark)
# ===========================================================================

def segment_folder(image_dir: str, out_dir: str, model_path: str,
                   tile: int = 512, overlap: int = 128, threshold: float = 0.5,
                   tta: bool = False):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    os.makedirs(out_dir, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = [f for f in sorted(os.listdir(image_dir))
             if os.path.splitext(f)[1].lower() in exts]
    for i, fn in enumerate(files, 1):
        stem = os.path.splitext(fn)[0]
        _, mask = segment_image(os.path.join(image_dir, fn), model, device,
                                tile=tile, overlap=overlap, threshold=threshold, tta=tta)
        cv2.imwrite(os.path.join(out_dir, f"{stem}.png"), mask)
        print(f"[{i}/{len(files)}] {fn} -> {stem}.png  ({mask.shape[1]}x{mask.shape[0]}, "
              f"{(mask>0).mean()*100:.2f}% crack)")


if __name__ == "__main__":
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("/content/best_model_v2.pth", device)

    prob, mask = segment_image(
        "/content/WhatsApp Image 2026-08-10 at 15.10.05 (1).jpeg",
        model, device, tile=512, overlap=128, threshold=0.5, tta=False,
    )
    cv2.imwrite("wall_crack_mask_4.png", mask)
    print(f"Saved wall_crack_mask_4.png  ({mask.shape[1]}x{mask.shape[0]}, "
          f"{(mask>0).mean()*100:.2f}% crack pixels)")

    # For the whole dataset (feeds run_dataset_benchmark.py):
    # segment_folder("dataset/images", "dataset/masks", "/content/best_model_v2.pth")
