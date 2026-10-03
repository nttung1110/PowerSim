#!/usr/bin/env bash
# Poke the ficus (elastic swing, paper Fig. 6): the pot is held by a cuboid BC, a brief
# sideways impulse hits the foliage.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
OUT="$OUT_ROOT/ficus_poke"
python -m powersim.foam_evolve.simulate \
    --checkpoint-config data/powerfoam_ckpt/ficus@48d229db/config.yaml \
    --sim-config config/ficus_poke.json \
    --background 1 1 1 \
    --output-dir "$OUT" --compile-video "$@"
echo "[run_ficus_poke] wrote $OUT/output.mp4"
