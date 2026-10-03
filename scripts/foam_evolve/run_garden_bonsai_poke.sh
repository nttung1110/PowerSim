#!/usr/bin/env bash
# Poke the bonsai tree composited into the MipNeRF-360 garden (paper Fig. 1b). Only the tree's
# primitives (data/selections/.../bonsai_tree_mask.pt) are simulated; pot and garden stay static.
# Set MATERIAL_FIELD=data/mat_prop_ckpt/bonsai_tree/material_field.pt to use the recovered E/nu
# field (paper Fig. 1c) instead of the uniform jelly material in the config.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
OUT="$OUT_ROOT/garden_bonsai_poke"
python -m powersim.foam_evolve.simulate \
    --checkpoint-config data/powerfoam_ckpt/garden_v1_bonsai_foreground_edit/config.yaml \
    --sim-config config/garden_bonsai_poke.json \
    --selection data/selections/garden_v1_bonsai_foreground_edit/bonsai_tree_mask.pt \
    ${MATERIAL_FIELD:+--material-field "$MATERIAL_FIELD"} \
    --show-force-indicator \
    --output-dir "$OUT" --compile-video "$@"
echo "[run_garden_bonsai_poke] wrote $OUT/output.mp4"
