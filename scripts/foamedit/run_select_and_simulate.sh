#!/usr/bin/env bash
# Select an object's primitives from 2D masks (render-weighted voting, paper §4.3) and simulate just
# those: the garden's vase + dried frond, kicked off balance by an impulse and toppled by gravity
# (paper Fig. 1 teaser scene). The rest of the garden is rendered static.
#   SELECTION=<mask.pt>  reuse a selection (e.g. data/selections/garden_v1/vase_and_frond_mask_paper.pt)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../foam_evolve/_env.sh"
OUT="${OUT:-$OUT_ROOT/garden_vase_topple}"
GARDEN=data/powerfoam_ckpt/garden_v1/config.yaml
if [ -z "${SELECTION:-}" ]; then
  SELECTION="$OUT/vase_and_frond_mask.pt"
  echo "=== 1/2 select: vase + frond primitives from the 2D masks"
  python -m powersim.foamedit.select --checkpoint-config $GARDEN \
      --masks data/masks/garden_v1/vase data/masks/garden_v1/frond --bg-weight 0.5 --drop-outliers --extend-hops 1 --extend-frame config/garden_vase_topple.json \
      --output "$SELECTION" --verify-dir "$OUT/select_verify"
fi
echo "=== 2/2 simulate the selection: impulse + gravity topple"
python -m powersim.foam_evolve.simulate --checkpoint-config $GARDEN \
    --sim-config config/garden_vase_topple.json --selection "$SELECTION" \
    $([ -f "${SELECTION%.pt}_unextended.pt" ] && echo "--recenter-selection ${SELECTION%.pt}_unextended.pt") \
    --output-dir "$OUT" --compile-video "$@"
echo "[run_select_and_simulate] wrote $OUT/output.mp4"
