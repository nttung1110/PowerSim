#!/usr/bin/env bash
# Pick the microphone capsule up, carry it sideways and drop it; the cable follows (paper Fig. 6).
# Head, cable and stand are one simulated body with a per-part material field; the stand is held
# kinematically (config mask_file). n_grid 128 so the thin cable stays coupled. Rendered over white.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
OUT="$OUT_ROOT/mic_pick_drop"
python -m powersim.foam_evolve.simulate \
    --checkpoint-config data/powerfoam_ckpt/mic@84daf179/config.yaml \
    --sim-config config/mic_pick_drop.json \
    --selection data/selections/mic@84daf179/all_mask.pt \
    --material-field data/selections/mic@84daf179/material_field_pick_drop.pt \
    --hide data/selections/mic@84daf179/hide_mask.pt \
    --split train --camera-index 3 \
    --background 1 1 1 \
    --output-dir "$OUT" --compile-video "$@"
echo "[run_mic_pick_drop] wrote $OUT/output.mp4"
