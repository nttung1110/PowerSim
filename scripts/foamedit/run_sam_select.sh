#!/usr/bin/env bash
# From photos to a simulation of one object, with no manual 3D picking:
#   1. Grounded-SAM: text prompts -> one binary mask per training photo (powersim.foamedit.generate_masks)
#   2. render-weighted voting: masks -> the object's primitives (powersim.foamedit.select)
#   3. simulate just those primitives (here the garden's vase + frond, toppled by an impulse and gravity)
# Needs the optional `transformers` dependency; the two models (~0.7 GB) download on first use.
#   PROMPTS="name=text ..." overrides the prompts (default: the paper's two for the garden vase).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../foam_evolve/_env.sh"
OUT="${OUT:-$POWERSIM_ROOT/outputs/foamedit/sam_select}"
GARDEN=data/powerfoam_ckpt/garden_v1/config.yaml
IMAGES="${IMAGES:-data/images/garden_v1}"
mkdir -p "$OUT"
echo "=== 1/3 Grounded-SAM masks for every photo in $IMAGES"
python -m powersim.foamedit.generate_masks --images-dir "$IMAGES" --out-root "$OUT/masks" \
    --prompt vase="a vase with a plant." --prompt frond="a dried palm frond."
echo "=== 2/3 render-weighted voting over the training views"
python -m powersim.foamedit.select --checkpoint-config $GARDEN \
    --masks "$OUT/masks/vase" "$OUT/masks/frond" --bg-weight 0.5 --drop-outliers --extend-hops 1 --extend-frame config/garden_vase_topple.json \
    --output "$OUT/vase_and_frond_mask.pt" --verify-dir "$OUT/select_verify"
echo "=== 3/3 simulate the selected primitives"
python -m powersim.foam_evolve.simulate --checkpoint-config $GARDEN \
    --sim-config config/garden_vase_topple.json --selection "$OUT/vase_and_frond_mask.pt" \
    --recenter-selection "$OUT/vase_and_frond_mask_unextended.pt" \
    --output-dir "$OUT/garden_vase_topple" --compile-video "$@"
echo "[run_sam_select] wrote $OUT/garden_vase_topple/output.mp4"
