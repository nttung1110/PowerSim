#!/usr/bin/env bash
# Builds the exact power-diagram adjacency shim used by the dynamic ray tracer (regular
# triangulation via geogram's single-file Delaunay PSM, BSD-3 licensed, bundled here).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
g++ -O3 -fopenmp -frounding-math -ffp-contract=off --std=c++17 -w geo_power_adj.cpp Delaunay_psm.cpp -o geo_power_adj
echo "built $(pwd)/geo_power_adj"
