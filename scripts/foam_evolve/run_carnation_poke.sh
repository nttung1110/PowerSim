#!/usr/bin/env bash
# Poke the carnation (paper Fig. 4): the flower primitives (data/selections/.../flower_mask_dilated20.pt)
# are simulated with a brief sideways impulse; the pot and background stay static.
# MATERIAL_FIELD selects the per-primitive Young's modulus field:
#   data/mat_prop_ckpt/carnation/material_field.pt   the field recovered from video (default)
#   <any .pt written by powersim.material_estimation.export_field>, e.g. --random for Fig. 4's top row
#   MATERIAL_FIELD=none                               the config's uniform E (1e7 Pa)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
MATERIAL_FIELD="${MATERIAL_FIELD:-data/mat_prop_ckpt/carnation/material_field.pt}"
OUT="${OUT:-$OUT_ROOT/carnation_poke}"
python -m powersim.foam_evolve.simulate \
    --checkpoint-config data/powerfoam_ckpt/carnations_sfm@ae005aa7/config.yaml \
    --sim-config config/carnations_poke.json \
    --selection data/selections/carnations_sfm@ae005aa7/flower_mask_dilated20.pt \
    $([ "$MATERIAL_FIELD" != "none" ] && echo "--material-field $MATERIAL_FIELD") \
    --show-force-indicator \
    --output-dir "$OUT" --compile-video "$@"
echo "[run_carnation_poke] wrote $OUT/output.mp4"
