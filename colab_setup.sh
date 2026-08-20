#!/usr/bin/env bash
# Install every backend benchmark.py can use, on Google Colab.
#
#   !bash colab_setup.sh
#
# torch / torchvision / numpy / scipy / opencv are already on Colab -- do NOT
# reinstall them, you will get a CUDA build mismatch. Everything below is
# additive.
set -u

echo "== timm / transformers / kornia / ultralytics =="
pip -q install timm transformers kornia ultralytics

echo "== torchreid (OSNet) =="
# The PyPI repackage nests under torchreid.reid; crack_reid_baselines.py
# already tries both import paths. --no-deps keeps it from pulling tb-nightly
# and downgrading Colab's numpy; its real runtime needs are listed explicitly.
pip -q install --no-deps torchreid
pip -q install gdown yacs

echo "== SuperGlue (repo checkout, not a package) =="
# crack_reid_baselines.py looks for the checkout next to ITSELF, not next to
# the notebook's cwd, so clone by the script's own directory. Getting this
# wrong is what produces "SuperGlue skipped: SuperGlue checkout not found".
# Weights ship inside the repo, so no separate download.
# Research / non-commercial licence -- check it before using in a paper.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SG="$HERE/third_party/SuperGluePretrainedNetwork"
echo "  target: $SG"
if [ -f "$SG/models/matching.py" ]; then
  echo "  already present, left alone"
else
  # Clone into a temp dir and move it into place only on success, so a failed
  # or half-finished clone can never leave a broken tree that the check above
  # would then happily accept on the next run.
  TMP="$(mktemp -d)"
  # Redirect to a log rather than piping: a pipeline reports the LAST
  # command's status, so `git clone ... | sed` would report sed's 0 and a
  # failed clone would look like a success.
  if git clone --depth 1 \
       https://github.com/magicleap/SuperGluePretrainedNetwork \
       "$TMP/sg" > "$TMP/log" 2>&1
  then
    mkdir -p "$HERE/third_party"
    rm -rf "$SG"
    mv "$TMP/sg" "$SG"
    echo "  cloned OK"
  else
    sed 's/^/    git: /' "$TMP/log"
    echo "  !! git clone FAILED -- superglue will be skipped."
    echo "  !! Check network/proxy, or clone it yourself into the target above."
  fi
  rm -rf "$TMP"
fi
if [ -f "$SG/models/matching.py" ]; then
  echo "  verified: $SG/models/matching.py"
else
  echo "  !! MISSING: $SG/models/matching.py"
  echo "  !! contents of $HERE/third_party:"
  ls -la "$HERE/third_party" 2>&1 | sed 's/^/    /'
fi

echo
echo "== what is actually importable =="
SUPERGLUE_PATH="$SG" python - <<'PY'
import importlib, os, sys, torch
sys.path.insert(0, os.environ["SUPERGLUE_PATH"])
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()} "
      f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY - switch runtime to GPU'})")
checks = {
    "sift/orb":     "cv2",
    "registration": "scipy",
    "deit/vit":     "timm",
    "clip":         "transformers",
    "osnet":        "torchreid",
    "yolo":         "ultralytics",
    "loftr":        "kornia",
    "superglue":    "models.matching",
}
bad = []
for method, mod in checks.items():
    try:
        importlib.import_module(mod)
        print(f"  OK    {method:<13} ({mod})")
    except Exception as e:
        bad.append(method)
        print(f"  FAIL  {method:<13} ({mod}): {type(e).__name__}: {e}")
        if mod == "models.matching":
            sg = os.environ["SUPERGLUE_PATH"]
            print(f"        looked in: {sg}")
            print(f"        exists: {os.path.isdir(sg)}  "
                  f"has models/matching.py: "
                  f"{os.path.isfile(os.path.join(sg, 'models', 'matching.py'))}")
print("\nall backends ready" if not bad else f"\nwill be skipped: {' '.join(bad)}")
PY
