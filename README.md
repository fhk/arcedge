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

The script builds, runs the unit tests, then runs the pipeline on three
instance tiers and checks the solver's bounds against the exact LP optimum
where HiGHS can compute it. Measured results (4-core container, GCC 13, -O3):

| Instance | Time-expanded arcs | Commodities | Gap | Time | Reference |
|----------|-------------------:|------------:|----:|-----:|-----------|
| small 10×10, T=8 | 3,220 | 10 | 0.77% | 0.08 s | HiGHS: LP* = 458.47 ∈ [457.94, 461.47] ✓ |
| medium 20×20, T=12 | 21,120 | 30 | 0.00% | 0.05 s | HiGHS: LP* = 2101.75 = LB = UB ✓ |
| large 70×70, T=45 | 1,065,680 | 250 | 0.85% | 14.2 s | certified LB/UB sandwich |

All gaps are within the Stage 1 exit threshold (≤ 2%). On the large instance
the pure-cost routing pass is infeasible (capacities bind) and the
multiplier-guided heuristic recovers a near-optimal feasible flow — the core
mechanism the architecture depends on.

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
src/arcedge/   core library: instance I/O, CSR graph, Dijkstra,
               time-expanded generator, Lagrangian solver, primal heuristic
src/main.cpp   CLI (gen / solve)
tests/         assert-style unit tests (ctest)
scripts/       HiGHS LP validator, end-to-end runner
PLAN.md        Stage 1 implementation plan
```
