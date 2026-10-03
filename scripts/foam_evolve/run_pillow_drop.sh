#!/usr/bin/env bash
# Pillows drop onto a sofa (paper Fig. 6), PhysGaussian's pillow2sofa scene on a PowerFoam
# checkpoint; train camera 87 is the side view used in the paper.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
OUT="$OUT_ROOT/pillow_drop"
python -m powersim.foam_evolve.simulate \
    --checkpoint-config data/powerfoam_ckpt/pillow2sofa_gs_fewer_points/config.yaml \
    --sim-config config/pillow_drop.json \
    --split train --camera-index 87 \
    --background 1 1 1 \
    --output-dir "$OUT" --compile-video "$@"
echo "[run_pillow_drop] wrote $OUT/output.mp4"
