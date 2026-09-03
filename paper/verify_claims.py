"""Re-derive every statistic quoted in the paper from committed artefacts.

Run from paper/:  python verify_claims.py
Exits non-zero if any claim fails to reproduce.
"""
import json, glob, os, math, sys, csv, re
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr, norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda *p: os.path.join(ROOT, *p)
fails = []

def check(label, got, want, tol=5e-3):
    ok = abs(got - want) <= tol
    print(f"  {'OK ' if ok else 'FAIL'}  {label:52s} got {got:>10.4f}  paper {want:>10.4f}")
    if not ok:
        fails.append(label)

print("\n[1] Appearance ceiling (Table I, Sec. IV)")
CROP = dict(  # name -> (R@1, mAP, DIR, pairF1, secs)
    ORB=(0.352, 0.364, 0.018, 0.305, 15.6), SIFT=(0.432, 0.417, 0.036, 0.313, 33.3),
    SuperGlue=(0.555, 0.453, 0.414, 0.353, 1594.0), LoFTR=(0.549, 0.441, 0.336, 0.359, 2331.1),
    YOLOv8=(0.561, 0.453, 0.157, 0.356, 4.6), ViT=(0.654, 0.511, 0.286, 0.331, 238.4),
    DeiT=(0.621, 0.497, 0.250, 0.305, 72.6), CLIP=(0.604, 0.471, 0.143, 0.307, 63.9),
    OSNet=(0.654, 0.520, 0.168, 0.309, 1.8), CrackShape=(0.554, 0.442, 0.082, 0.305, 0.9))
v = np.array(list(CROP.values()))
r1, dirf, f1, sec = v[:, 0], v[:, 2], v[:, 3], v[:, 4] / 280.0
check("pairF1 band width", f1.max() - f1.min(), 0.054)
check("pairF1 min", f1.min(), 0.305); check("pairF1 max", f1.max(), 0.359)
check("pairF1 mean", f1.mean(), 0.324); check("pairF1 sd", f1.std(ddof=1), 0.023)
check("pairF1 CV", f1.std() / f1.mean(), 0.068)
check("R@1 span", r1.max() - r1.min(), 0.302)
check("R@1 CV", r1.std() / r1.mean(), 0.164)
check("DIR span", dirf.max() - dirf.min(), 0.396)
check("DIR CV", dirf.std() / dirf.mean(), 0.655)
check("compute span (x)", sec.max() / sec.min(), 2590, tol=5)
check("Spearman R@1 vs pairF1", spearmanr(r1, f1)[0], 0.018, tol=0.01)
check("  ... its p-value", spearmanr(r1, f1)[1], 0.96, tol=0.02)
check("Spearman log(cost) vs R@1", spearmanr(np.log(sec), r1)[0], 0.018, tol=0.01)
check("Spearman log(cost) vs pairF1", spearmanr(np.log(sec), f1)[0], 0.497, tol=0.01)
check("REG DIR / best baseline DIR", 0.768 / dirf.max(), 1.86, tol=0.02)
check("REG calib F1 / best oracle F1", 0.582 / f1.max(), 1.62, tol=0.02)
check("REG cost / OSNet cost", (7817.3 / 280) / (1.8 / 280), 4343, tol=10)

print("\n[2] Dataset composition (Table II)")
sp = json.load(open(J('dataset/splits.json')))
files = [f for f in sorted(glob.glob(J('dataset/labels/*.json')))
         if not os.path.basename(f).startswith('_')]
idph, wid, wph, npts = defaultdict(set), defaultdict(set), defaultdict(set), 0
for f in files:
    d = json.load(open(f)); img = d['image_id']; w = img.split('_')[0]
    wph[w].add(img)
    for p in d['points']:
        npts += 1; idph[p['identity']].add(img); wid[w].add(p['identity'])
for name, walls, ids_, multi, ph in [("val", sp['val'], 66, 18, 66),
                                     ("test", sp['test'], 146, 19, 74)]:
    got = set().union(*[wid[w] for w in walls])
    check(f"{name} identities", len(got), ids_, tol=0)
    check(f"{name} multi-photo identities",
          len([i for i in got if len(idph[i]) >= 2]), multi, tol=0)
    check(f"{name} photos", sum(len(wph[w]) for w in walls), ph, tol=0)
check("labelled points", npts, 1051, tol=0)
check("photographs", len(files), 140, tol=0)

print("\n[3] Registration coverage and the gate (Sec. VI-D, VI-G)")
po = json.load(open(J('banchmark_out/pair_outcomes.json')))['registered']
sharp = json.load(open(J('dataset/quality.json')))['sharpness']
check("test image pairs", len(po), 301, tol=0)
check("registered pairs", sum(po.values()), 97, tol=0)
check("overall registration rate", sum(po.values()) / len(po), 0.322)
timgs = [i for i in sharp if i.split('_')[0] in sp['test']]
for gate, kept, walls2, rate in [(10, 53, 10, 0.561), (25, 40, 7, 0.683)]:
    ks = {i for i in timgs if sharp[i] >= gate}
    bw = defaultdict(int)
    for i in ks: bw[i.split('_')[0]] += 1
    prs = [k for k in po if all(x in ks for x in k.split('|'))]
    check(f"gate>={gate}: images kept", len(ks), kept, tol=0)
    check(f"gate>={gate}: walls pairable", sum(1 for c in bw.values() if c >= 2), walls2, tol=0)
    check(f"gate>={gate}: registration rate", sum(1 for k in prs if po[k]) / len(prs), rate)
check("captures rejected at gate 10", 74 - 53, 21, tol=0)

print("\n[4] Viewpoint covariate (Sec. VI-E, Fig. 3)")
vp = json.load(open(J('banchmark_out/viewpoint.json')))['pairs']
ok = [k for k, x in vp.items() if x['viewpoint'].get('ok')]
check("covariate-solved pairs", len(ok), 139, tol=0)
check("  ... as fraction", len(ok) / 301, 0.462)
solved_failed = [k for k in ok if k in po and not po[k]]
check("solved but scorer-failed", len(solved_failed), 51, tol=0)
check("  ... share of failures", len(solved_failed) / (301 - 97), 0.25)
def q(xs, p):
    xs = sorted(x for x in xs if math.isfinite(x))
    i = p * (len(xs) - 1); lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)
for fld, med, p90 in [('scale_change', 1.42, 2.26), ('abs_rotation_deg', 6.92, 39.58),
                      ('tilt_deg', 25.57, 75.32)]:
    xs = [vp[k]['viewpoint'][fld] for k in ok]
    check(f"{fld} median", q(xs, .5), med, tol=0.02)
    check(f"{fld} p90", q(xs, .9), p90, tol=0.05)
fg = np.array([vp[k]['frame_gap'] for k in ok], float)
for fld, rho in [('scale_change', 0.101), ('abs_rotation_deg', 0.311), ('tilt_deg', 0.351)]:
    x = np.array([vp[k]['viewpoint'][fld] for k in ok], float)
    m = np.isfinite(x)
    check(f"frame-gap vs {fld} rho", spearmanr(fg[m], x[m])[0], rho, tol=0.01)
ls, rot = [], []
for k, x in vp.items():
    be = x.get('viewpoint_by_estimator', {})
    if 'orb' in be and 'ecc' in be and be['orb']['scale'] > 0 and be['ecc']['scale'] > 0:
        ls.append(abs(np.log(be['orb']['scale']) - np.log(be['ecc']['scale'])))
        rot.append(abs(((be['orb']['rotation_deg'] - be['ecc']['rotation_deg']) + 180) % 360 - 180))
check("pairs solved by both estimators", len(ls), 32, tol=0)
check("median |log-scale| disagreement", float(np.median(ls)), 0.106, tol=0.002)
check("median rotation disagreement", float(np.median(rot)), 2.91, tol=0.02)

print("\n[5] Logistic model of registration failure (Sec. VI-E)")
keys = [k for k in vp if k in po]
X = np.column_stack([np.ones(len(keys)),
                     [np.log1p(vp[k]['sharp_min']) for k in keys],
                     [vp[k]['frame_gap'] for k in keys]])
y = np.array([1.0 if po[k] else 0.0 for k in keys])
b = np.zeros(3)
for _ in range(300):
    mu = 1 / (1 + np.exp(-(X @ b))); W = mu * (1 - mu) + 1e-9
    b = np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (W * (X @ b + (y - mu) / W)))
mu = 1 / (1 + np.exp(-(X @ b)))
se = np.sqrt(np.diag(np.linalg.inv(X.T @ (X * (mu * (1 - mu))[:, None]))))
check("sharpness Wald z", b[1] / se[1], 8.23, tol=0.02)
check("frame-gap Wald z", b[2] / se[2], -5.03, tol=0.02)
check("median sharpness | registered",
      float(np.median([vp[k]['sharp_min'] for k in keys if po[k]])), 133.8, tol=0.2)
check("median sharpness | failed",
      float(np.median([vp[k]['sharp_min'] for k in keys if not po[k]])), 7.2, tol=0.2)

print("\n[6] Annotation provenance (Sec. IV-C)")
cl = json.load(open(J('dataset/labels/_changelog.json')))
tot = merged = 0; gaps = defaultdict(list)
for wall, info in cl.items():
    for ident, meta in info.get('identities', {}).items():
        tot += 1
        if len(meta.get('absorbed_prefill_ids', [])) > 1:
            merged += 1
            if meta.get('max_merge_gap') is not None:
                gaps['test' if wall in sp['test'] else 'val'].append(meta['max_merge_gap'])
check("identities total", tot, 212, tol=0)
check("identities formed by merge", merged, 95, tol=0)
check("  ... as fraction", merged / tot, 0.45, tol=0.005)
check("median max merge gap (test, px)", float(np.median(gaps['test'])), 228.2, tol=0.2)
check("median max merge gap (val, px)", float(np.median(gaps['val'])), 285.6, tol=0.2)
# per-split merge counts and gap tails quoted in Sec. IV-C
per = defaultdict(lambda: [0, 0])
for wall, info in cl.items():
    sp_ = 'test' if wall in sp['test'] else 'val'
    for ident, meta in info.get('identities', {}).items():
        per[sp_][1] += 1
        if len(meta.get('absorbed_prefill_ids', [])) > 1:
            per[sp_][0] += 1
check("test identities merged", per['test'][0], 70, tol=0)
check("test merge rate", per['test'][0] / per['test'][1], 0.48, tol=0.005)
check("val identities merged", per['val'][0], 25, tol=0)
check("val merge rate", per['val'][0] / per['val'][1], 0.38, tol=0.005)
check("p90 bridged gap (test, px)", float(np.percentile(gaps['test'], 90)), 307.6, tol=1.0)
check("p90 bridged gap (val, px)", float(np.percentile(gaps['val'], 90)), 316.9, tol=1.0)
check("max bridged gap (test, px)", float(np.max(gaps['test'])), 318.7, tol=0.2)
check("max bridged gap (val, px)", float(np.max(gaps['val'])), 320.0, tol=0.2)
check("triage-flagged identities",
      len(json.load(open(J('dataset/labels/_triage.json')))), 57, tol=0)
# the audit that is deliberately NOT claimed must remain absent
import glob as _g
n_audit = len(_g.glob(J('dataset/labels/_audit_*.json')))
check("filled audit files (paper claims none)", n_audit, 0, tol=0)

print("\n[7] Capture separation (Sec. IV-B)")
rows = list(csv.DictReader(open(J('dataset/wall_map.csv'))))
bywall = defaultdict(list)
for r in rows:
    m = re.search(r'(\d{8})_(\d{6})', r['source'])
    if m:
        t = m.group(2)
        bywall[r['wall_id']].append(int(t[:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6]))
g, spans = [], []
for w, ts in bywall.items():
    ts = sorted(ts); spans.append(ts[-1] - ts[0]); g += [b - a for a, b in zip(ts, ts[1:])]
check("median consecutive-frame gap (s)", float(np.median(g)), 3, tol=0)
check("max consecutive-frame gap (s)", max(g), 20, tol=0)
check("max per-wall span (s)", max(spans), 81, tol=0)

print("\n[8] Designed viewpoint ablation (Sec. V-D, reported in prose)")
import collections
def bins(axis):
    rows = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "synth", f"rows_{axis}.json")))
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["method"], r["scale"], r["rotation_deg"], r["tilt_deg"])].append(r)
    out = {}
    for k, v in g.items():
        w = np.array([r["n_queries"] for r in v], float)
        if not w.sum():
            continue
        out[k] = (int(w.sum()), float((w / w.sum() @ [r["rank1"] for r in v])),
                  float(np.mean([r["dir_at_far10"] for r in v])))
    return out
R, T = bins("rot"), bins("tilt")
for (src, key, n, r1_) in [
        (R, ("sift", 1.0, 0.0, 0.0), 207, 0.469), (R, ("sift", 1.0, 45.0, 0.0), 201, 0.418),
        (R, ("orb", 1.0, 0.0, 0.0), 207, 0.456),  (R, ("orb", 1.0, 45.0, 0.0), 201, 0.502),
        (T, ("sift", 1.0, 0.0, 60.0), 192, 0.432), (T, ("orb", 1.0, 0.0, 60.0), 192, 0.489)]:
    got_n, got_r1, _ = src[key]
    check(f"synth {key[0]} rot{key[2]:.0f} tilt{key[3]:.0f}: n", got_n, n, tol=0)
    check(f"synth {key[0]} rot{key[2]:.0f} tilt{key[3]:.0f}: R@1", got_r1, r1_, tol=0.002)
check("SIFT R@1 drop over 45 deg rotation",
      R[("sift", 1.0, 0.0, 0.0)][1] - R[("sift", 1.0, 45.0, 0.0)][1], 0.051, tol=0.003)
check("ORB R@1 gain over 45 deg rotation",
      R[("orb", 1.0, 45.0, 0.0)][1] - R[("orb", 1.0, 0.0, 0.0)][1], 0.046, tol=0.003)
check("SIFT R@1 drop over 60 deg tilt",
      T[("sift", 1.0, 0.0, 0.0)][1] - T[("sift", 1.0, 0.0, 60.0)][1], 0.037, tol=0.003)
check("ORB R@1 gain over 60 deg tilt",
      T[("orb", 1.0, 0.0, 60.0)][1] - T[("orb", 1.0, 0.0, 0.0)][1], 0.033, tol=0.003)

print("\n" + "=" * 74)
if fails:
    print(f"{len(fails)} CLAIM(S) FAILED TO REPRODUCE:"); [print("  -", f) for f in fails]
    sys.exit(1)
print("All quoted claims reproduce from committed artefacts.")
