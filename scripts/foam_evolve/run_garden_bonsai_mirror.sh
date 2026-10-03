#!/usr/bin/env bash
# The bonsai poke of run_garden_bonsai_poke.sh, ray traced with a wide bordered mirror standing behind
# the table: the reflection follows the tree's motion (paper Fig. 1d). Exact power-diagram adjacency is
# rebuilt every frame (build the shim once: bash third_party/geogram_psm/build.sh); a few seconds per frame.
#   MATERIAL_FIELD=data/mat_prop_ckpt/bonsai_tree/material_field.pt  uses the recovered E/nu field
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
OUT="${OUT:-$OUT_ROOT/garden_bonsai_mirror}"
python -m powersim.foam_evolve.simulate_raytrace \
    --checkpoint-config data/powerfoam_ckpt/garden_v1_bonsai_foreground_edit/config.yaml \
    --sim-config config/garden_bonsai_poke.json \
    --selection data/selections/garden_v1_bonsai_foreground_edit/bonsai_tree_mask.pt \
    ${MATERIAL_FIELD:+--material-field "$MATERIAL_FIELD"} \
    --transform-json config/mirrors/garden_bonsai_wide.json \
    --camera-json config/cameras/garden_bonsai_wide.json \
    --output-dir "$OUT" --compile-video "$@"
echo "[run_garden_bonsai_mirror] wrote $OUT/output.mp4"
