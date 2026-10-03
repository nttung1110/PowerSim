#!/usr/bin/env bash
# Swing the telephone's coiled cord (paper Fig. 6): its top is pinned, the bottom is nudged
# sideways for 0.2 s, then gravity swings it. Test camera 10 is the checkpoint's own test/010 view.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
OUT="$OUT_ROOT/telephone_cord_swing"
python -m powersim.foam_evolve.simulate \
    --checkpoint-config data/powerfoam_ckpt/telephone_sfm@2ed8d7fa/config.yaml \
    --sim-config config/telephone_cord_swing.json \
    --selection data/selections/telephone_sfm@2ed8d7fa/cord_mask.pt \
    --split test --camera-index 10 \
    --output-dir "$OUT" --compile-video "$@"
echo "[run_telephone_cord_swing] wrote $OUT/output.mp4"
