#!/usr/bin/env bash
# End-to-end proof of the Stage 1 combinatorial core:
#   build -> unit tests -> generate instances -> solve (Lagrangian LB + primal UB)
#   -> validate bounds against the exact HiGHS LP optimum (small + medium)
#   -> certified-gap run at ~1M-arc scale.
# Requires: cmake, a C++17 compiler, python3 with highspy + numpy
#   (pip install highspy numpy).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1  # progress must stream when piped (Colab, CI)

echo "=== [1/8] build ==="
cmake -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j"$(nproc)" >/dev/null

echo "=== [2/8] unit tests ==="
ctest --test-dir build --output-on-failure

echo "=== [3/8] small instance: solve + HiGHS validation ==="
mkdir -p data
./build/arcedge gen --out data/small.txt --width 10 --height 10 --time 8 \
  --commodities 10 --cap 3 --hubs 2 --seed 42
./build/arcedge solve data/small.txt --iters 300 --tol 0.002 --quiet \
  --result data/small.result
python3 scripts/validate_lp.py data/small.txt data/small.result

echo "=== [4/8] medium instance: solve + HiGHS validation ==="
./build/arcedge gen --out data/medium.txt --width 20 --height 20 --time 12 \
  --commodities 30 --cap 4 --hubs 3 --seed 7
./build/arcedge solve data/medium.txt --iters 400 --tol 0.002 --quiet \
  --result data/medium.result
python3 scripts/validate_lp.py data/medium.txt data/medium.result

echo "=== [5/8] large instance (~1M arcs): certified LB/UB gap ==="
./build/arcedge gen --out data/large.txt --width 70 --height 70 --time 45 \
  --commodities 250 --cap 3 --hubs 4 --hub-frac 0.7 --seed 11
./build/arcedge solve data/large.txt --iters 150 --tol 0.01 --primal-every 15 \
  --result data/large.result

# Real street data (Overture Maps, San Francisco). The committed .graph files
# were produced by scripts/overture_to_graph.py from an Overture
# transportation-segment GeoParquet; rerun that step with:
#   python3 scripts/overture_to_graph.py streets.parquet data/sf_streets.graph
echo "=== [6/8] downtown SF (Overture import): solve + HiGHS validation ==="
./build/arcedge gen --street data/sf_downtown.graph --out data/sf_dt_te.txt \
  --time 12 --commodities 12 --cap 4 --hubs 2 --hub-frac 0.6 --seed 23
./build/arcedge solve data/sf_dt_te.txt --iters 400 --tol 0.002 --quiet \
  --result data/sf_dt_te.result
python3 scripts/validate_lp.py data/sf_dt_te.txt data/sf_dt_te.result

echo "=== [7/8] full SF street network (~1.7M TE arcs): certified gap ==="
./build/arcedge gen --street data/sf_streets.graph --out data/sf_te.txt \
  --time 36 --commodities 150 --cap 8 --hubs 5 --hub-frac 0.5 --seed 17
./build/arcedge solve data/sf_te.txt --iters 120 --tol 0.01 --primal-every 10 \
  --result data/sf_te.result

echo "=== [8/8] downtown SF hub design (POI access network) vs HiGHS MIP ==="
# data/sf_dt_access.* were produced by scripts/connect_pois.py: every POI is
# connected to its nearest street edge by a perpendicular drop that splits
# the edge at the foot point.
./build/arcedge design --graph data/sf_dt_access.graph \
  --pois data/sf_dt_access.pois --cap 500 --hub-cost 20000 --cable-cost 10 \
  --quiet --result data/sf_dt_design.result
# The MIP reference typically runs to its time limit before printing the
# verdict -- that's HiGHS proving a bound, not a hang.
python3 scripts/validate_design_mip.py data/sf_dt_access.graph \
  data/sf_dt_access.pois data/sf_dt_design.result --time-limit 120

echo
echo "E2E PASSED: unit tests green, HiGHS confirms lb <= LP* <= ub on"
echo "synthetic small/medium and downtown SF, the synthetic-large and"
echo "full-SF instances close to certified gaps, and the hub design beats"
echo "or matches the time-limited HiGHS MIP incumbent on downtown SF."
