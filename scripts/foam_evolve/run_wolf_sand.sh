#!/usr/bin/env bash
# Wolf plushie turns to sand (Drucker-Prager, paper Fig. 6) on the interior-filled checkpoint.
# Filled interior primitives take the look of the nearest surface primitive when exposed
# (--appearance-source); --cull-density 70 drops the faint fur halo.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
OUT="$OUT_ROOT/wolf_sand"
python -m powersim.foam_evolve.simulate \
    --checkpoint-config data/powerfoam_ckpt/wolf_gs_filled/config.yaml \
    --sim-config config/wolf_sand.json \
    --split test --camera-index 18 \
    --radius-mode deformation \
    --appearance-source data/selections/wolf_gs_filled/surface_mask.pt \
    --cull-density 70 \
    --background 1 1 1 \
    --output-dir "$OUT" --compile-video "$@"
echo "[run_wolf_sand] wrote $OUT/output.mp4"
