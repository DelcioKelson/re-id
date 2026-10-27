"""Generate every figure in the paper from measured artefacts in this repo.

Sources, all committed:
  banchmark_out/result.txt        the benchmark table
  banchmark_out/pair_outcomes.json  per-image-pair registration success
  banchmark_out/viewpoint.json    per-pair viewpoint covariate
  dataset/quality.json            variance-of-Laplacian sharpness
"""
import json, math, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    # Camera-ready fonts: pdf.fonttype 42 embeds TrueType as vectors
    # instead of Type 3 bitmapped. All figure text then embeds (no
    # Type 3), satisfying IEEE PDF requirements.
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "text.usetex": False,
})

INK   = "#1a1a1a"
GREY  = "#9a9a9a"
ACC   = "#b2453c"    # proposed method
BLUE  = "#33608c"

# ---------------------------------------------------------------- table data
# PARSED from banchmark_out/result.txt, the same file Table I is copied from.
# This list used to be transcribed by hand and had drifted from the benchmark
# output it was supposed to plot (ORB's cost, SuperGlue's Rank-1 and DIR all
# differed), which is the same class of error the paper's own verifier exists
# to catch. One source, one parser: make_figs.py and verify_claims.py both read
# the file the table is copied from.
RESULT_TXT = os.path.join(ROOT, "banchmark_out/result.txt")

# result.txt method name -> (figure label, family)
_DISPLAY = {
    "ORB": ("ORB", "keypoint"),
    "SIFT": ("SIFT", "keypoint"),
    "SuperGlue": ("SuperGlue", "learned"),
    "LoFTR": ("LoFTR", "learned"),
    "YOLOv8-backbone": ("YOLOv8", "embedding"),
    "ViT": ("ViT-B/16", "embedding"),
    "DeiT": ("DeiT-S", "embedding"),
    "CLIP": ("CLIP", "embedding"),
    "DINOv2": ("DINOv2", "embedding"),
    "OSNet": ("OSNet", "re-ID"),
}
_FAMILY_ORDER = {"keypoint": 0, "learned": 1, "embedding": 2, "re-ID": 3}


def parse_results(path):
    """method -> {column: value} from a result.txt / results_table.txt."""
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
            raise SystemExit(f"result.txt row {parts[0]!r} has {len(nums)} numeric "
                             f"fields, header expects {len(header) - 2}")
        rows[parts[0]] = dict(zip(header[2:], nums))
    return rows


RESULTS = parse_results(RESULT_TXT)


def _load():
    crop = []
    for key, (label, fam) in _DISPLAY.items():
        r = RESULTS[key]
        crop.append((label, fam, r["R@1"], r["R@5"], r["mAP"],
                     r["DIR@FAR.1"], r["pairF1*"], r["assF1"], r["total_s"]))
    crop.sort(key=lambda c: _FAMILY_ORDER[c[1]])
    reg = RESULTS["registration+chamfer"]
    return crop, ("Registration+Chamfer", reg["R@1"], reg["R@5"], reg["mAP"],
                  reg["DIR@FAR.1"], reg["pairF1*"], reg["assF1"], reg["total_s"])


CROP, REG = _load()
NQ = 280
# The geometric baseline answers only 48% of pairs, and on that scorable
# subset the best-constant predictor is 0.544 (2p/(1+p) at the subset's
# prevalence 0.373), not the 0.305 the crop rows sit on. Drawing its bar
# against 0.305 alone is the misreading the caption has to warn about.
COVERAGE_ONLY_F1 = RESULTS["coverage-only"]["pairF1*"]

FAMCOL = {"keypoint": "#7c7c7c", "learned": "#4f7a9e", "embedding": "#6b8f5e",
          "re-ID": "#a3803e"}


def fig_ceiling():
    """Fig 1 (teaser): pairwise F1 is flat across the crop matchers; geometry
    breaks out, but against a different floor, so both floors are drawn."""
    fig, ax = plt.subplots(figsize=(3.45, 1.62))
    names = [c[0] for c in CROP]
    f1    = [c[6] for c in CROP]
    fam   = [c[1] for c in CROP]
    order = np.argsort(f1)
    x = np.arange(len(names))

    lo, hi = min(f1), max(f1)
    ax.axhspan(lo, hi, color=GREY, alpha=0.20, zorder=0, lw=0)
    # Sits just above the band, not at hi+0.075: the second floor line at
    # 0.544 and its label need the space above it, and two annotations that
    # collide are worse than one that is tight to what it annotates.
    ax.annotate(f"appearance band\n{lo:.3f}–{hi:.3f}  (width {hi-lo:.3f})",
                xy=(len(names) / 2.0, hi), xytext=(len(names) / 2.0, hi + 0.022),
                ha="center", va="bottom", fontsize=6.6, color="#5a5a5a")

    # The floor the geometric bar must be read against: it answers 48% of
    # pairs, on which the best-constant predictor is 0.544, not 0.305.
    ax.axhline(COVERAGE_ONLY_F1, color=ACC, ls=(0, (3, 2)), lw=0.8, zorder=2)
    ax.text(len(names) + 0.25, COVERAGE_ONLY_F1 + 0.022,
            f"floor on its 48% registered pool: {COVERAGE_ONLY_F1:.3f}",
            fontsize=6.0, color=ACC, ha="right", va="bottom")

    for k, i in enumerate(order):
        ax.bar(k, f1[i], color=FAMCOL[fam[i]], width=0.66, zorder=3, lw=0)
    ax.bar(len(names) + 0.6, REG[5], color=ACC, width=0.66, zorder=3, lw=0)
    ax.text(len(names) + 0.6, REG[5] + 0.012, f"{REG[5]:.3f}", ha="center",
            va="bottom", fontsize=7, color=ACC, fontweight="bold")

    labels = [names[i] for i in order] + ["Geometric ref."]
    ax.set_xticks(list(range(len(names))) + [len(names) + 0.6])
    ax.set_xticklabels(labels, rotation=42, ha="right")
    ax.get_xticklabels()[-1].set_color(ACC)
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_ylabel("pairwise F1")
    ax.set_ylim(0, 0.80)
    ax.grid(axis="y", color="#e2e2e2", lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMCOL[k]) for k in FAMCOL]
    ax.legend(handles, list(FAMCOL), ncol=3, frameon=False, loc="upper left",
              handlelength=0.9, columnspacing=0.9, handletextpad=0.4,
              borderpad=0.1, labelspacing=0.25)
    fig.savefig(os.path.join(FIGS, "ceiling.pdf"))
    plt.close(fig)


def fig_tradeoff():
    """Fig 2: open-set DIR@FAR=0.1 against cost per query."""
    fig, ax = plt.subplots(figsize=(3.45, 2.15))
    for name, fam, r1, r5, mAP, dirf, f1, af1, secs in CROP:
        ax.scatter(secs / NQ, dirf, s=26, color=FAMCOL[fam], zorder=3,
                   edgecolor="white", lw=0.6)
        ax.annotate(name, (secs / NQ, dirf), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=6.2, color="#4a4a4a")
    ax.scatter(REG[7] / NQ, REG[4], s=64, marker="*", color=ACC, zorder=4,
               edgecolor="white", lw=0.6)
    ax.annotate("Registration+Chamfer\n(reference)", (REG[7] / NQ, REG[4]),
                textcoords="offset points", xytext=(-6, -2), ha="right",
                fontsize=6.6, color=ACC, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("cost per query (s, log scale)")
    ax.set_ylabel("DIR @ FAR = 0.1")
    ax.set_ylim(-0.03, 0.88)
    ax.grid(color="#e8e8e8", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(FIGS, "tradeoff.pdf"))
    plt.close(fig)


def fig_coverage():
    """Fig 3: registration coverage per wall, ordered by median sharpness."""
    po = json.load(open(os.path.join(ROOT, "banchmark_out/pair_outcomes.json")))["registered"]
    sharp = json.load(open(os.path.join(ROOT, "dataset/quality.json")))["sharpness"]

    bywall = defaultdict(lambda: [0, 0])
    for k, v in po.items():
        w = k.split("_")[0]
        bywall[w][1] += 1
        bywall[w][0] += bool(v)
    med = {}
    for w in bywall:
        vals = [s for i, s in sharp.items() if i.split("_")[0] == w]
        med[w] = float(np.median(vals)) if vals else 0.0
    walls = sorted(bywall, key=lambda w: med[w])

    fig, ax = plt.subplots(figsize=(3.45, 1.62))
    rate = [bywall[w][0] / bywall[w][1] for w in walls]
    cols = [ACC if r == 0 else (BLUE if r < 0.5 else "#4f7a9e") for r in rate]
    ax.bar(range(len(walls)), rate, color=cols, width=0.68, zorder=3, lw=0)
    for i, w in enumerate(walls):
        ok, n = bywall[w]
        ax.text(i, rate[i] + 0.03, f"{ok}/{n}", ha="center", fontsize=5.6,
                color="#4a4a4a")
    ax.set_xticks(range(len(walls)))
    ax.set_xticklabels([w.replace("wall", "") for w in walls], fontsize=6.5)
    ax.set_xlabel("wall, ordered by median sharpness (low $\\rightarrow$ high)")
    ax.set_ylabel("pairs registered")
    ax.set_ylim(0, 1.16)
    ax.axhline(97 / 301, color=INK, ls=(0, (3, 2)), lw=0.8, zorder=4)
    ax.text(len(walls) - 0.4, 97 / 301 + 0.035, "overall 32.2%", ha="right",
            fontsize=6.4, color=INK)
    ax.grid(axis="y", color="#e8e8e8", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(FIGS, "coverage.pdf"))
    plt.close(fig)


def fig_gate():
    """Fig 4: the admission-gate trade-off - rate against walls kept pairable."""
    sp = json.load(open(os.path.join(ROOT, "dataset/splits.json")))
    sharp = json.load(open(os.path.join(ROOT, "dataset/quality.json")))["sharpness"]
    po = json.load(open(os.path.join(ROOT, "banchmark_out/pair_outcomes.json")))["registered"]
    timgs = [i for i in sharp if i.split("_")[0] in sp["test"]]

    gates = [0, 5, 10, 15, 25, 35, 50, 75, 100, 150, 200]
    rates, walls2, kept_reg = [], [], []
    for g in gates:
        ks = {i for i in timgs if sharp[i] >= g}
        bw = defaultdict(int)
        for i in ks:
            bw[i.split("_")[0]] += 1
        walls2.append(sum(1 for c in bw.values() if c >= 2))
        prs = [k for k in po if all(x in ks for x in k.split("|"))]
        reg = sum(1 for k in prs if po[k])
        rates.append(reg / max(len(prs), 1))
        kept_reg.append(reg)

    fig, ax = plt.subplots(figsize=(3.45, 1.62))
    ax.plot(gates, rates, "-o", color=BLUE, ms=3.2, lw=1.2, zorder=3,
            label="registration rate")
    ax.set_xlabel("sharpness admission gate (variance of Laplacian)")
    ax.set_ylabel("registration rate", color=BLUE)
    ax.tick_params(axis="y", colors=BLUE)
    ax.set_ylim(0.25, 0.78)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(gates, walls2, "-s", color=ACC, ms=3.0, lw=1.2, zorder=3,
             label="walls still pairable")
    ax2.set_ylabel("walls with $\\geq$2 admitted photos", color=ACC)
    ax2.tick_params(axis="y", colors=ACC)
    ax2.set_ylim(3, 11.8)

    ax.axvline(10, color=INK, ls=(0, (3, 2)), lw=0.8, zorder=2)
    ax.text(11.5, 0.72, "operating point", fontsize=6.4, color=INK)
    ax.grid(color="#eeeeee", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    fig.savefig(os.path.join(FIGS, "gate.pdf"))
    plt.close(fig)


def fig_viewpoint():
    """Fig 5: the viewpoint the dataset actually spans, from an independent front end."""
    vp = json.load(open(os.path.join(ROOT, "banchmark_out/viewpoint.json")))["pairs"]
    po = json.load(open(os.path.join(ROOT, "banchmark_out/pair_outcomes.json")))["registered"]
    ok = [k for k, v in vp.items() if v["viewpoint"].get("ok")]

    fields = [("scale_change", "scale ($\\times$)", (1, 5.4)),
              ("abs_rotation_deg", "|rotation| ($^\\circ$)", (0, 135)),
              ("tilt_deg", "tilt ($^\\circ$)", (0, 90))]
    fig, axes = plt.subplots(1, 3, figsize=(3.45, 1.30))
    for ax, (f, lab, xlim) in zip(axes, fields):
        reg = [vp[k]["viewpoint"][f] for k in ok if po.get(k)]
        fail = [vp[k]["viewpoint"][f] for k in ok if not po.get(k)]
        reg = [x for x in reg if np.isfinite(x)]
        fail = [x for x in fail if np.isfinite(x)]
        bins = np.linspace(xlim[0], xlim[1], 16)
        ax.hist([reg, fail], bins=bins, stacked=True, color=[BLUE, "#d8b7b3"],
                label=["registered", "not registered"], lw=0)
        ax.axvline(np.median(reg + fail), color=INK, ls=(0, (3, 2)), lw=0.9)
        ax.set_xlabel(lab, fontsize=6.5)
        ax.set_xlim(*xlim)
        ax.tick_params(labelsize=5.8)
        ax.locator_params(axis='x', nbins=4)
        ax.grid(axis="y", color="#eeeeee", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("pairs", fontsize=6.5)
    axes[1].legend(frameon=False, loc="upper right", handlelength=0.7,
                   handletextpad=0.3, borderpad=0.05, fontsize=5.6)
    fig.subplots_adjust(wspace=0.38)
    fig.savefig(os.path.join(FIGS, "viewpoint.pdf"))
    plt.close(fig)


def fig_dataset_examples():
    """Representative crops from labeled real captures, not new measurements."""
    examples = [
        ("wall01_s1_0001", (1425, 1452), "Same identity, frame 0001"),
        ("wall01_s1_0009", (1536, 1532), "Same identity, frame 0009"),
        ("wall01_s1_0001", (1425, 1452), "Stippled render"),
        ("wall10_s1_0001", (1447, 1873), "Smooth painted plaster"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 5.0))
    for ax, (image_id, (cx, cy), title) in zip(axes.flat, examples):
        path = os.path.join(ROOT, "dataset", "images", image_id + ".jpg")
        image = Image.open(path).convert("RGB")
        width, height = 1400, 1600
        box = (max(0, cx - width // 2), max(0, cy - height // 2),
               min(image.width, cx + width // 2), min(image.height, cy + height // 2))
        ax.imshow(image.crop(box))
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    fig.tight_layout(pad=0.6)
    fig.savefig(os.path.join(FIGS, "dataset_examples.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_ceiling(); fig_tradeoff(); fig_coverage(); fig_gate(); fig_viewpoint()
    fig_dataset_examples()
    print("figures written to", FIGS)
