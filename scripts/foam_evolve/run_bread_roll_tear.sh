#!/usr/bin/env bash
# Tear a bread roll apart: two opposing translation regions pull its ends (paper Fig. 2).
# Rendered over white like the paper figure; drop --background for the checkpoint's black.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
OUT="$OUT_ROOT/bread_roll_tear"
python -m powersim.foam_evolve.simulate \
    --checkpoint-config data/powerfoam_ckpt/bread_roll@ec2ced65/config.yaml \
    --sim-config config/bread_roll_tear.json \
    --background 1 1 1 \
    --output-dir "$OUT" --compile-video "$@"
echo "[run_bread_roll_tear] wrote $OUT/output.mp4"
