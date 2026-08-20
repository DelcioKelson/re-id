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
# Weights ship inside the repo, so no separate download.
# Research / non-commercial licence -- check it before using in a paper.
if [ ! -d third_party/SuperGluePretrainedNetwork ]; then
  git clone -q --depth 1 \
    https://github.com/magicleap/SuperGluePretrainedNetwork \
    third_party/SuperGluePretrainedNetwork
fi

echo
echo "== what is actually importable =="
python - <<'PY'
import importlib, os, sys, torch
sys.path.insert(0, os.path.join("third_party", "SuperGluePretrainedNetwork"))
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
print("\nall backends ready" if not bad else f"\nwill be skipped: {' '.join(bad)}")
PY
