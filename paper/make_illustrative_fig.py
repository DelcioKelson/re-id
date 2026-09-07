"""Generate figs/illustrative_evolution.pdf -- a SYNTHETIC mock-up, not a
measurement.

Everything else under paper/figs/ is produced by make_figs.py from committed
benchmark artefacts (banchmark_out/*.json, result.txt) and is data. This
figure is different in kind: it is one real photograph
(dataset/images/wall01_s1_0002.jpg) edited three times in GIMP to make its
crack progressively shorter, standing in for a revisit sequence CrackID does
not contain. It exists only to make Sec.~VII's "measurement that would change
the most" concrete to a reader. No metric in this paper is computed from it,
it is not part of the released dataset, and dataset/README.md's ban on
describing CrackID as multi-visit applies to it doubly -- these four panels
are not four visits, they are one photograph.

The edit, run headless via GIMP 3's Python-Fu (org.gimp.GIMP flatpak):
for each of the three synthetic stages, duplicate the layer, shift the copy
sideways by dx pixels (same rows, so the wall's vertical lighting gradient is
preserved -- only the horizontal position of the sampled texture changes),
and composite it over the lower portion of the crack through a feathered
layer mask so the cut fades rather than stops abruptly. dx is chosen per
stage so the shifted band contains no crack pixels of its own (checked
against the dilated full-image mask, not just this crack's mask). The fourth
panel is the original file, untouched.

Run:  python make_illustrative_fig.py
Requires: flatpak run org.gimp.GIMP (tested at 3.2.4), PIL, numpy, scipy.
"""
import json, os, signal, subprocess, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGS = os.path.join(HERE, "figs")

SRC_IMAGE = os.path.join(ROOT, "dataset/images/wall01_s1_0002.jpg")
SRC_MASK = os.path.join(ROOT, "dataset/masks/wall01_s1_0002.png")
SPLIT_FRACS = [0.22, 0.45, 0.70]   # fraction of the crack's length kept, per stage
TAPER = 200                        # px, feather zone at the cut
DX_CANDIDATES = [300, -300, 400, -400, 500, -500, 600, -600,
                  700, -700, 800, -800, 900, -900]

GIMP_SCRIPT = r'''
import gi
gi.require_version('Gimp', '3.0')
gi.require_version('Gegl', '0.4')
from gi.repository import Gimp, Gio, Gegl
import json

data = json.load(open({params_path!r}))
for p in data["stages"]:
    img = Gimp.file_load(Gimp.RunMode.NONINTERACTIVE, Gio.File.new_for_path(p['img_path']))
    base = img.get_layers()[0]
    patch = base.copy()
    img.insert_layer(patch, None, 0)
    dummy = Gegl.Color.new('black')
    patch.offset(True, Gimp.OffsetType.WRAP_AROUND, dummy, p['dx'], 0)
    mask = patch.create_mask(Gimp.AddMaskType.BLACK)
    patch.add_mask(mask)
    bx0, bx1 = p['bx0'], p['bx1']
    y_taper_end, y_erase_end = p['y_taper_end'], p['y_erase_end']
    img.select_rectangle(Gimp.ChannelOps.REPLACE, bx0, y_taper_end, bx1 - bx0, y_erase_end - y_taper_end)
    Gimp.Selection.feather(img, 120)
    mask.edit_fill(Gimp.FillType.WHITE)
    Gimp.Selection.none(img)
    img.flatten()
    Gimp.file_save(Gimp.RunMode.NONINTERACTIVE, img, Gio.File.new_for_path(p['out_path']), None)
    img.delete()
'''
# No Gimp.quit() call here: it is deprecated in this GIMP build and errors
# out rather than exiting cleanly, which would just add noise. GIMP's batch
# mode does not exit on its own once the batch commands finish either way --
# it drops into "running as a background process" and waits for Ctrl-C -- so
# run_gimp() below polls for the expected output files instead of trusting
# the process to terminate itself.


def compute_stage_params(tmpdir):
    mask = np.array(Image.open(SRC_MASK).convert('L')) > 100
    h, w = mask.shape
    lbl, n = ndimage.label(mask, structure=np.ones((3, 3)))
    sizes = ndimage.sum(mask, lbl, range(1, n + 1))
    main = np.argmax(sizes) + 1
    comp = lbl == main
    ys, xs = np.where(comp)
    y0, y1 = int(ys.min()), int(ys.max())
    dil = ndimage.binary_dilation(mask, iterations=15)

    stages = []
    for i, frac in enumerate(SPLIT_FRACS, start=1):
        y_keep_until = int(y0 + (y1 - y0) * frac)
        y_taper_end = y_keep_until + TAPER
        y_erase_end = y1 + 40
        band_xs = np.where(comp[y_keep_until:y_erase_end + 1, :].any(axis=0))[0]
        bx0 = max(0, int(band_xs.min()) - 160)
        bx1 = min(w - 1, int(band_xs.max()) + 160)

        def clean(dx, y_taper_end=y_taper_end, y_erase_end=y_erase_end, bx0=bx0, bx1=bx1):
            xa, xb = bx0 + dx, bx1 + dx
            return 0 <= xa and xb < w and not dil[y_taper_end:y_erase_end + 1, xa:xb + 1].any()

        dx = next((c for c in DX_CANDIDATES if clean(c)), None)
        if dx is None:
            raise RuntimeError(f"no clean shift found for stage {i}")
        stages.append(dict(stage=i, split_frac=frac, img_path=SRC_IMAGE, w=int(w), h=int(h),
                            bx0=bx0, bx1=bx1, y_taper_end=y_taper_end, y_erase_end=y_erase_end,
                            dx=dx, out_path=os.path.join(tmpdir, f"stage{i}.jpg")))
    return dict(stages=stages, bbox=[int(xs.min()), y0, int(xs.max()), y1])


def run_gimp(params_path, stages, poll_s=2, max_wait_s=240):
    # subprocess.run(timeout=...) only kills the immediate `flatpak` process,
    # which does not reach the bwrap/gimp children it spawns, so a plain
    # timeout leaves them running forever (confirmed: it did, twice). Put the
    # whole thing in its own process group so it can actually be killed, and
    # poll for the expected output files rather than waiting on the process
    # to exit -- it won't, on its own.
    import time
    script = GIMP_SCRIPT.format(params_path=params_path)
    script_path = params_path + ".py"
    open(script_path, "w").write(script)
    proc = subprocess.Popen(
        ["flatpak", "run", "org.gimp.GIMP", "--new-instance", "-i",
         "--batch-interpreter=python-fu-eval",
         "-b", f"exec(open({script_path!r}).read())"],
        start_new_session=True)
    targets = [s["out_path"] for s in stages]
    waited = 0
    try:
        while waited < max_wait_s:
            if all(os.path.exists(t) and os.path.getsize(t) > 0 for t in targets):
                return
            if proc.poll() is not None:
                raise RuntimeError(f"GIMP exited early (code {proc.returncode}) "
                                    f"before writing all of {targets}")
            time.sleep(poll_s)
            waited += poll_s
        raise TimeoutError(f"GIMP did not produce {targets} within {max_wait_s}s")
    finally:
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()


def compose(params, tmpdir):
    x0, y0, x1, y1 = params["bbox"]
    pad = 150
    box = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
    paths = [s["out_path"] for s in params["stages"]] + [SRC_IMAGE]
    labels = ["synthetic, shortest", "synthetic, medium", "synthetic, longest",
              "real photo (unedited)"]

    crops = [Image.open(p).crop(box) for p in paths]
    w, h = crops[0].size
    tw = 260
    scale = tw / w
    th = int(h * scale)
    crops = [c.resize((tw, th), Image.LANCZOS) for c in crops]

    gap, label_h, banner_h = 14, 34, 44
    n = len(crops)
    W = tw * n + gap * (n - 1)
    H = th + label_h + banner_h
    canvas = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(canvas)

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    banner = "SYNTHETIC — DIGITALLY EDITED. NOT A REAL REVISIT. NOT EVALUATED."
    size = 20
    font_banner = ImageFont.truetype(font_path, size)
    bb = d.textbbox((0, 0), banner, font=font_banner)
    while (bb[2] - bb[0]) > W - 24 and size > 8:
        size -= 1
        font_banner = ImageFont.truetype(font_path, size)
        bb = d.textbbox((0, 0), banner, font=font_banner)
    font_label = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)

    d.rectangle([0, 0, W, banner_h], fill=(178, 69, 60))
    tx = (W - (bb[2] - bb[0])) // 2
    ty = (banner_h - (bb[3] - bb[1])) // 2 - bb[1]
    d.text((tx, ty), banner, fill='white', font=font_banner)

    for i, (c, lab) in enumerate(zip(crops, labels)):
        x = i * (tw + gap)
        canvas.paste(c, (x, banner_h))
        lb = d.textbbox((0, 0), lab, font=font_label)
        d.text((x + (tw - (lb[2] - lb[0])) // 2, banner_h + th + 6), lab,
               fill='black', font=font_label)

    os.makedirs(FIGS, exist_ok=True)
    canvas.convert('RGB').save(os.path.join(FIGS, "illustrative_evolution.pdf"))


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        params = compute_stage_params(tmpdir)
        params_path = os.path.join(tmpdir, "params.json")
        json.dump(params, open(params_path, "w"))
        run_gimp(params_path, params["stages"])
        compose(params, tmpdir)
    print("wrote", os.path.join(FIGS, "illustrative_evolution.pdf"))


if __name__ == "__main__":
    main()
