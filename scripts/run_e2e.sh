#!/usr/bin/env bash
# End-to-end proof of the Stage 1 combinatorial core:
#   build -> unit tests -> generate instances -> solve (Lagrangian LB + primal UB)
#   -> validate bounds against the exact HiGHS LP optimum (small + medium)
#   -> certified-gap run at ~1M-arc scale.
# Requires: cmake, a C++17 compiler, python3 with highspy + numpy + pyarrow
#   + shapely + pyyaml (pip install highspy numpy pyarrow shapely pyyaml).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1  # progress must stream when piped (Colab, CI)

echo "=== [1/10] build ==="
NCORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
cmake -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j"$NCORES" >/dev/null

echo "=== [2/10] unit tests ==="
ctest --test-dir build --output-on-failure

echo "=== [3/10] small instance: solve + HiGHS validation ==="
mkdir -p data
./build/arcedge gen --out data/small.txt --width 10 --height 10 --time 8 \
  --commodities 10 --cap 3 --hubs 2 --seed 42
./build/arcedge solve data/small.txt --iters 300 --tol 0.002 --quiet \
  --result data/small.result
python3 scripts/validate_lp.py data/small.txt data/small.result

echo "=== [4/10] medium instance: solve + HiGHS validation ==="
./build/arcedge gen --out data/medium.txt --width 20 --height 20 --time 12 \
  --commodities 30 --cap 4 --hubs 3 --seed 7
./build/arcedge solve data/medium.txt --iters 400 --tol 0.002 --quiet \
  --result data/medium.result
python3 scripts/validate_lp.py data/medium.txt data/medium.result

echo "=== [5/10] large instance (~1M arcs): certified LB/UB gap ==="
./build/arcedge gen --out data/large.txt --width 70 --height 70 --time 45 \
  --commodities 250 --cap 3 --hubs 4 --hub-frac 0.7 --seed 11
./build/arcedge solve data/large.txt --iters 150 --tol 0.01 --primal-every 15 \
  --result data/large.result

# Real street data (Overture Maps, San Francisco). The committed .graph files
# were produced by scripts/overture_to_graph.py from an Overture
# transportation-segment GeoParquet; rerun that step with:
#   python3 scripts/overture_to_graph.py streets.parquet data/sf_streets.graph
echo "=== [6/10] downtown SF (Overture import): solve + HiGHS validation ==="
./build/arcedge gen --street data/sf_downtown.graph --out data/sf_dt_te.txt \
  --time 12 --commodities 12 --cap 4 --hubs 2 --hub-frac 0.6 --seed 23
./build/arcedge solve data/sf_dt_te.txt --iters 400 --tol 0.002 --quiet \
  --result data/sf_dt_te.result --flow data/sf_dt_te.flow
python3 scripts/validate_lp.py data/sf_dt_te.txt data/sf_dt_te.result

echo "=== [7/10] full SF street network (~1.7M TE arcs): certified gap ==="
./build/arcedge gen --street data/sf_streets.graph --out data/sf_te.txt \
  --time 36 --commodities 150 --cap 8 --hubs 5 --hub-frac 0.5 --seed 17
./build/arcedge solve data/sf_te.txt --iters 120 --tol 0.01 --primal-every 10 \
  --result data/sf_te.result

echo "=== [8/10] downtown SF hub design (POI access network) vs HiGHS MIP ==="
# data/sf_dt_access.* were produced by scripts/connect_pois.py: every POI is
# connected to its nearest street edge by a perpendicular drop that splits
# the edge at the foot point.
./build/arcedge design --graph data/sf_dt_access.graph \
  --pois data/sf_dt_access.pois --cap 500 --hub-cost 20000 --cable-cost 10 \
  --quiet --result data/sf_dt_design.result --solution data/sf_dt_design.solution
# The MIP reference typically runs to its time limit before printing the
# verdict -- that's HiGHS proving a bound, not a hang.
python3 scripts/validate_design_mip.py data/sf_dt_access.graph \
  data/sf_dt_access.pois data/sf_dt_design.result --time-limit 120

echo "=== [9/10] solution output: map to street geometry, write GeoParquet ==="
# Requires pyarrow + shapely (same as the Overture importer).
python3 scripts/solution_to_geoparquet.py flow \
  --street data/sf_downtown.graph --instance data/sf_dt_te.txt \
  --flow data/sf_dt_te.flow --out data/sf_dt_flow.parquet \
  --check-ub data/sf_dt_te.result
python3 scripts/solution_to_geoparquet.py design \
  --graph data/sf_dt_access.graph --pois data/sf_dt_access.pois \
  --solution data/sf_dt_design.solution --out data/sf_dt_design.parquet

echo "=== [10/10] model config: compile examples, solve, check acceptance ==="
# Requires pyyaml. R3-7a acceptance: the declarative configs reproduce the
# hand-built pipelines -- design must land on the known 392,123 total, and
# the soft-capacity MCF must close to <= 1% (it cannot go hard-infeasible).
python3 scripts/arcedge_modelc.py examples/sf_dt_access_design.yaml \
  -o out/dt_design > /dev/null
./build/arcedge design --graph out/dt_design/access.graph \
  --pois out/dt_design/tier0.pois --cap 500 --hub-cost 20000 \
  --cable-cost 10 --quiet --result out/dt_design/result.txt
python3 scripts/arcedge_modelc.py examples/sf_dt_te_mcf.yaml \
  -o out/dt_mcf > /dev/null
./build/arcedge solve out/dt_mcf/instance.txt --iters 300 --tol 0.005 \
  --quiet --result out/dt_mcf/result.txt
# Multi-tier FTTH chain: 1 address -> terminal (12) -> FDH (512) -> OLT (4000)
# with joint-feedback rounds (best chain by true cost can never be worse
# than the greedy round-1 chain).
python3 scripts/arcedge_modelc.py examples/sf_dt_ftth_tiers.yaml \
  -o out/dt_ftth --solve --rounds 2
python3 - <<'PYEOF'
import json, math
d = dict(l.split() for l in open('out/dt_design/result.txt')
         if not l.startswith('hub_nodes'))
total = float(d['total_cost'])
assert abs(total - 392123) <= 0.01 * 392123, f'design total {total} off baseline'
m = dict(l.split() for l in open('out/dt_mcf/result.txt'))
assert float(m['gap']) <= 0.01, f"mcf gap {m['gap']} > 1%"
t = json.load(open('out/dt_ftth/tiers_summary.json'))
tiers = {x['tier']: x for x in t['tiers']}
assert tiers['terminal']['hubs'] >= math.ceil(400 / 12), 'terminal cap violated'
assert tiers['fdh']['hubs'] >= 1 and tiers['olt']['hubs'] >= 1
assert t['total_cost'] < 1e6, 'FTTH chain cost unreasonable'
rounds = t['rounds']
assert t['total_cost'] <= rounds[0]['total_cost'] + 1e-6, \
    'joint feedback made the chain worse than greedy'
print(f"model-config acceptance OK: design total {total:.0f} "
      f"(baseline 392123), mcf gap {100 * float(m['gap']):.3f}%, "
      f"ftth chain {tiers['terminal']['hubs']}t/{tiers['fdh']['hubs']}f/"
      f"{tiers['olt']['hubs']}o = {t['total_cost']:.0f}")
PYEOF

echo
echo "E2E PASSED: unit tests green, HiGHS confirms lb <= LP* <= ub on"
echo "synthetic small/medium and downtown SF, the synthetic-large and"
echo "full-SF instances close to certified gaps, the hub design beats"
echo "or matches the time-limited HiGHS MIP incumbent on downtown SF,"
echo "both solutions round-trip to GeoParquet geometry, and the model"
echo "compiler reproduces the hand-built pipelines from example configs."
