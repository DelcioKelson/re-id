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

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGS = os.path.join(HERE, "figs")
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

INK   = "#1a1a1a"
GREY  = "#9a9a9a"
ACC   = "#b2453c"    # proposed method
BLUE  = "#33608c"

# ---------------------------------------------------------------- table data
# transcribed from banchmark_out/result.txt (see paper/results.tex)
CROP = [   # name, family, R@1, R@5, mAP, DIR@FAR.1, pairF1, assF1, total_s
    ("ORB",       "keypoint",  0.352, 0.759, 0.364, 0.018, 0.305, 0.325,   15.6),
    ("SIFT",      "keypoint",  0.432, 0.829, 0.417, 0.036, 0.313, 0.324,   33.3),
    ("SuperGlue", "learned",   0.555, 0.893, 0.453, 0.414, 0.353, 0.352, 1594.0),
    ("LoFTR",     "learned",   0.549, 0.824, 0.441, 0.336, 0.359, 0.401, 2331.1),
    ("YOLOv8",    "embedding", 0.561, 0.821, 0.453, 0.157, 0.356, 0.364,    4.6),
    ("ViT-B/16",  "embedding", 0.654, 0.893, 0.511, 0.286, 0.331, 0.392,  238.4),
    ("DeiT-S",    "embedding", 0.621, 0.879, 0.497, 0.250, 0.305, 0.381,   72.6),
    ("CLIP",      "embedding", 0.604, 0.850, 0.471, 0.143, 0.307, 0.367,   63.9),
    ("OSNet",     "re-ID",     0.654, 0.907, 0.520, 0.168, 0.309, 0.388,    1.8),
    ("CrackShape","shape",     0.554, 0.825, 0.442, 0.082, 0.305, 0.346,    0.9),
]
REG = ("Registration+Chamfer", 0.952, 0.992, 0.886, 0.768, 0.656, 0.784, 7817.3)
NQ = 280

FAMCOL = {"keypoint": "#7c7c7c", "learned": "#4f7a9e", "embedding": "#6b8f5e",
          "re-ID": "#a3803e", "shape": "#8a6f9e"}


def fig_ceiling():
    """Fig 1 (teaser): pairwise F1 is flat across ten matchers; geometry breaks out."""
    fig, ax = plt.subplots(figsize=(3.45, 1.62))
    names = [c[0] for c in CROP]
    f1    = [c[6] for c in CROP]
    fam   = [c[1] for c in CROP]
    order = np.argsort(f1)
    x = np.arange(len(names))

    lo, hi = min(f1), max(f1)
    ax.axhspan(lo, hi, color=GREY, alpha=0.20, zorder=0, lw=0)
    ax.annotate(f"appearance band\n{lo:.3f}–{hi:.3f}  (width {hi-lo:.3f})",
                xy=(len(names) / 2.0, hi), xytext=(len(names) / 2.0, hi + 0.075),
                ha="center", va="bottom", fontsize=6.6, color="#5a5a5a")

    for k, i in enumerate(order):
        ax.bar(k, f1[i], color=FAMCOL[fam[i]], width=0.66, zorder=3, lw=0)
    ax.bar(len(names) + 0.6, REG[5], color=ACC, width=0.66, zorder=3, lw=0)
    ax.text(len(names) + 0.6, REG[5] + 0.012, f"{REG[5]:.3f}", ha="center",
            va="bottom", fontsize=7, color=ACC, fontweight="bold")

    labels = [names[i] for i in order] + ["Ours"]
    ax.set_xticks(list(range(len(names))) + [len(names) + 0.6])
    ax.set_xticklabels(labels, rotation=42, ha="right")
    ax.get_xticklabels()[-1].set_color(ACC)
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_ylabel("pairwise F1")
    ax.set_ylim(0, 0.78)
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
    ax.annotate("Registration+Chamfer\n(ours)", (REG[7] / NQ, REG[4]),
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


if __name__ == "__main__":
    fig_ceiling(); fig_tradeoff(); fig_coverage(); fig_gate(); fig_viewpoint()
    print("figures written to", FIGS)
