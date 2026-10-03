#!/usr/bin/env bash
# Crush a coke can from the top with a descending press (von Mises plasticity, paper Fig. 6).
# Only the outer wall shell is simulated; the inner reconstruction primitives follow it for
# rendering (--passive-follow). A per-primitive material field makes lid and base rigid, and a
# small inward displacement seeds the diamond dents.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
OUT="$OUT_ROOT/coke_can_compress"
python -m powersim.foam_evolve.simulate \
    --checkpoint-config data/powerfoam_ckpt/coke_can_mesh_fewer_points/config.yaml \
    --sim-config config/coke_can_compress.json \
    --split test --camera-index 12 \
    --selection data/selections/coke_can_mesh_fewer_points/shell_selection.pt --passive-follow \
    --material-field data/selections/coke_can_mesh_fewer_points/material_field.pt \
    --displace data/selections/coke_can_mesh_fewer_points/displace.pt \
    --radius-mode neighbor \
    --background 1 1 1 \
    --output-dir "$OUT" --compile-video "$@"
echo "[run_coke_can_compress] wrote $OUT/output.mp4"
