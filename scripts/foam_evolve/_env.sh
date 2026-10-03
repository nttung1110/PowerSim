# Sourced by every run script: activates the `powersim` env and moves to the repo root.
# PYTHONNOUSERSITE=1 keeps packages in ~/.local (the user site) from shadowing the env.
export PYTHONNOUSERSITE=1
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-$HOME/.cache/warp}"
POWERSIM_ENV="${POWERSIM_ENV:-powersim}"
if [ "${CONDA_DEFAULT_ENV:-}" != "$POWERSIM_ENV" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$POWERSIM_ENV"
fi
POWERSIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$POWERSIM_ROOT"
OUT_ROOT="${OUT_ROOT:-$POWERSIM_ROOT/outputs/sim_results}"
mkdir -p "$OUT_ROOT"
