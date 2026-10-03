#!/usr/bin/env bash
# Recover the carnation's Young's modulus field from the captured poke video (paper Fig. 4), then
# resimulate with it. About an hour on one GPU. Results in outputs/material_estimation/carnation/
# (see powersim/material_estimation/run_video.py for the file list); the recovered field is then
# usable by the forward simulator:
#   MATERIAL_FIELD=outputs/material_estimation/carnation/material_field.pt bash scripts/foam_evolve/run_carnation_poke.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../foam_evolve/_env.sh"
OUT="${OUT:-$POWERSIM_ROOT/outputs/material_estimation/carnation}"
python -m powersim.material_estimation.run_video \
    --checkpoint-config data/powerfoam_ckpt/carnations_sfm@ae005aa7/config.yaml \
    --sim-config config/carnations_poke.json \
    --selection data/selections/carnations_sfm@ae005aa7/flower_mask_dilated20.pt \
    --fill-cache data/selections/carnations_sfm@ae005aa7/interior_fill_cache_flower_mask_dilated20.pt \
    --video data/gt_video/carnation/carnation_poke.mp4 \
    --output-dir "$OUT" "$@"
echo "[run_carnation] done: $OUT"
