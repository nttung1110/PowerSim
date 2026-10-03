#!/usr/bin/env bash
# Paper Fig. 5: select the garden's vase from 2D masks, remove it, and insert the separately captured
# bonsai in its place -- producing a simulation-ready scene (the input of run_garden_bonsai_poke.sh).
#   SELECTION=<mask.pt>  skips the voting step and uses that selection (e.g.
#                        data/selections/garden_v1/vase_and_frond_mask_paper.pt, the paper's own)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../foam_evolve/_env.sh"
OUT="${OUT:-$POWERSIM_ROOT/outputs/foamedit/garden_bonsai}"
mkdir -p "$OUT"
GARDEN=data/powerfoam_ckpt/garden_v1/config.yaml
if [ -z "${SELECTION:-}" ]; then
  SELECTION="$OUT/vase_and_frond_mask.pt"
  echo "=== 1/3 select: vase + frond primitives of garden_v1 from the 2D masks (render-weighted vote)"
  python -m powersim.foamedit.select --checkpoint-config $GARDEN \
      --masks data/masks/garden_v1/vase data/masks/garden_v1/frond --bg-weight 0.5 --drop-outliers \
      --output "$SELECTION" --labels-out "$OUT/vase_and_frond_labels.pt" --verify-dir "$OUT/select_verify"
fi
echo "=== 2/3 remove: delete the selection (grown 5 hops) and rebuild adjacency"
python -m powersim.foamedit.remove --checkpoint-config $GARDEN --selection "$SELECTION" --hops 5 \
    --output-dir "$OUT/garden_v1_vase_removed"
echo "=== 3/3 insert: the bonsai capture in place of the vase (similarity transform onto the tabletop)"
python -m powersim.foamedit.insert --target-config $GARDEN \
    --source-config data/powerfoam_ckpt/bonsai_foreground/config.yaml \
    --target-frame config/frames/garden_v1.json --source-frame config/frames/bonsai_foreground.json \
    --replace-selection "$SELECTION" --mask-hops 5 --scale-multiple 1.5 \
    --carry-mask tree=data/selections/bonsai_foreground/tree_mask.pt \
    --output-dir "$OUT/garden_v1_bonsai_edit"
echo "=== done. Simulate the composite with:"
echo "  python -m powersim.foam_evolve.simulate --checkpoint-config $OUT/garden_v1_bonsai_edit/config.yaml \\"
echo "      --sim-config config/garden_bonsai_poke.json --selection $OUT/garden_v1_bonsai_edit/tree.pt \\"
echo "      --show-force-indicator --output-dir outputs/sim_results/garden_bonsai_poke_edited --compile-video"
