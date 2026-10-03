#!/usr/bin/env bash
# Runs every foam_evolve demo in sequence (pass scene names to run a subset).
# Each writes outputs/sim_results/<scene>/{frame_%04d.png,output.mp4}.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
SCENES=${@:-"garden_bonsai_poke bread_roll_tear ficus_poke pillow_drop coke_can_compress wolf_sand telephone_cord_swing mic_pick_drop"}
for s in $SCENES; do
  echo "=== [$(date +%H:%M:%S)] $s"
  bash "run_$s.sh" > "../../outputs/sim_results/$s.log" 2>&1 && echo "=== $s OK" || echo "=== $s FAILED (see outputs/sim_results/$s.log)"
done
