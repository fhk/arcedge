# arcedge

Custom combinatorial solver for time-expanded flow and edge-assignment problems
on street networks (capacitated multicommodity flow, fixed-charge network
design, PCST/Steiner connectivity).

This repository currently contains the **Stage 1 combinatorial core**:
Lagrangian relaxation of capacitated MCF with batched shortest-path
subproblems (CPU threads now, GPU in Stage 2), a multiplier-guided primal
repair heuristic, and an exact-LP validation harness (HiGHS). See
[`PLAN.md`](PLAN.md) for the full Stage 1 implementation plan, milestones, and
exit criteria.

## Build & prove end to end

```bash
pip install highspy numpy      # LP validation reference
./scripts/run_e2e.sh
```

The script builds, runs the unit tests, then runs the pipeline on synthetic
grids and real San Francisco street data, checking the solver's bounds against
the exact LP optimum where HiGHS can compute it. Measured results (4-core
container, GCC 13, -O3):

| Instance | Time-expanded arcs | Commodities | Gap | Time | Reference |
|----------|-------------------:|------------:|----:|-----:|-----------|
| grid 10×10, T=8 | 3,220 | 10 | 0.77% | 0.08 s | HiGHS: LP* = 458.47 ∈ [457.94, 461.47] ✓ |
| grid 20×20, T=12 | 21,120 | 30 | 0.00% | 0.05 s | HiGHS: LP* = 2101.75 = LB = UB ✓ |
| grid 70×70, T=45 | 1,065,680 | 250 | 0.85% | 14.2 s | certified LB/UB sandwich |
| SF downtown (Overture), T=12 | 20,702 | 12 | 0.96% | 0.27 s | HiGHS: LP* = 27,666.26 m = LB (to 4 s.f.) ✓ |
| SF full city (Overture), T=36 | 1,683,885 | 150 | 0.68% | 9.8 s | certified LB/UB sandwich |

All gaps are within the Stage 1 exit threshold (≤ 2%). On the large instances
the pure-cost routing pass is infeasible (capacities bind) and the
multiplier-guided heuristic recovers a near-optimal feasible flow — the core
mechanism the architecture depends on.

## Real street data (Overture Maps)

`scripts/overture_to_graph.py` converts Overture transportation segments
(GeoParquet) into a static street graph: every LineString endpoint becomes a
node (deduplicated, so segments meeting at an intersection share a node), and
each segment yields arcs in **both directions** weighted by geodesic polyline
length in meters. Pedestrian-only classes (footway, steps, path, bridleway)
and non-road subtypes are excluded by default (`--all-classes` keeps them),
and only the largest connected component is kept so all sampled commodities
are routable.

```bash
pip install pyarrow shapely
python3 scripts/overture_to_graph.py streets_sf.parquet data/sf_streets.graph
# optional crop: --bbox XMIN YMIN XMAX YMAX

# time-expand + sample commodities on the real graph, then solve
./build/arcedge gen --street data/sf_streets.graph --out data/sf_te.txt \
    --time 36 --commodities 150 --cap 8 --hubs 5 --hub-frac 0.5 --seed 17
./build/arcedge solve data/sf_te.txt --iters 120 --tol 0.01
```

For the April 2026 SF extract (54,645 segments), the importer keeps 24,572
street segments; the largest component has 11,487 nodes / 36,624 directed
arcs / 2,102 km of street. The derived graphs are committed as
`data/sf_streets.graph` and `data/sf_downtown.graph` so the E2E run does not
need the parquet.

Capacity envelopes matter on real graphs: street networks have low-degree
cuts, so over-concentrated demand (few hubs, low `--cap`) can make the
instance genuinely infeasible. The solver detects this — an unbounded
Lagrangian dual is an infeasibility certificate — and reports it instead of
iterating forever.

## CLI

```bash
# generate a time-expanded capacitated MCF instance over a street grid
./build/arcedge gen --out inst.txt --width 20 --height 20 --time 12 \
    --commodities 30 --cap 4 --hubs 3 --seed 7

# solve: Lagrangian lower bound + heuristic upper bound + certified gap
./build/arcedge solve inst.txt --iters 400 --tol 0.005 --result inst.result

# validate bounds against the exact LP optimum (small/medium instances)
python3 scripts/validate_lp.py inst.txt inst.result
```

## Layout

```
src/arcedge/   core library: instance I/O, CSR graph, Dijkstra, street-graph
               loader, time-expanded generator (grid + real street graphs),
               Lagrangian solver, primal heuristic
src/main.cpp   CLI (gen / solve)
tests/         assert-style unit tests (ctest)
scripts/       Overture importer, HiGHS LP validator, end-to-end runner
data/          committed SF street graphs derived from Overture Maps
PLAN.md        Stage 1 implementation plan
```
