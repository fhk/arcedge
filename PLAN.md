# arcedge — Implementation Plan (Stage 1 ✅ → Stage 2)

**Stage 1 (weeks 0–8): Combinatorial core MVP.**
Build the Lagrangian decomposition for capacitated multicommodity flow (MCF) on
time-expanded street networks, CPU-first, with bounds validated against the
HiGHS LP relaxation on medium instances.

> **Exit criterion (go/no-go for Stage 2): ≤ 2% empirical optimality gap on
> representative instances**, measured as `(UB − LB) / UB` where LB is the
> Lagrangian dual bound and UB comes from the primal heuristic.

This plan follows the architecture study's recommendation (Architecture A): a
combinatorial/algorithmic core — Lagrangian relaxation with batched
shortest-path subproblems + primal repair — rather than a general-purpose
LP/MIP kernel. GPU work (batched SSSP, LNS, Steiner engine) is deferred to
Stage 2; GPU LP companions (cuOpt PDLP, cuDSS IPM) to Stage 3.

---

## 1. Problem scope for Stage 1

Capacitated multicommodity min-cost flow on a time-expanded graph:

```
min   Σ_k Σ_a c_a · x_a^k
s.t.  N x^k = b^k                    ∀k   (per-commodity flow conservation)
      Σ_k x_a^k ≤ u_a                ∀a ∈ C   (joint arc capacities)
      x ≥ 0
```

- The static graph is a sparse, near-planar street network; time expansion
  replicates it across `T` layers with movement arcs (advance one layer, carry
  the street capacity per time step) and uncapacitated waiting arcs.
- Stage 1 solves the **linear** (continuous-flow) problem. The fixed-charge
  binary variant (arc-open decisions, knapsack relaxation) enters at M6 as a
  stretch goal and is the Stage 2 default.
- Target scale: ~1M arcs after time expansion, hundreds of commodities.

## 2. Algorithmic core

**Lagrangian flow relaxation.** Dualize the capacity constraints with
multipliers λ ≥ 0. The relaxed problem separates by commodity into shortest
paths under reduced costs `c_a + λ_a`:

```
L(λ) = Σ_k q_k · d_λ(o_k, d_k) − Σ_a λ_a u_a  ≤  LP* ≤ OPT
```

Reduced costs stay non-negative, so plain Dijkstra applies. The K shortest-path
subproblems are independent — batch them across CPU threads now, across GPU
warps in Stage 2 (this interface boundary is deliberate).

**Dual ascent.** Projected subgradient with Polyak steps
`θ = α (UB − L(λ)) / ‖g‖²`, subgradient `g_a = load_a − u_a` on dualized arcs
(components projected out where λ_a = 0 and g_a < 0), α halved after stall
periods. Upgrade path inside Stage 1: multiplier averaging, then the **volume
algorithm**, then a bundle method if dual convergence is the bottleneck
(literature: bundle converges faster and more robustly; volume is cheaper per
iteration at the largest scales).

**Primal repair heuristic (UB).** Sequential successive-shortest-path routing
in the residual network, largest demands first, saturated arcs removed; run
once with multiplier-guided costs `c + λ` (λ ≈ LP duals steer flow off
congested arcs) and once with pure costs, keep the cheaper feasible solution.
Refresh every N subgradient iterations. Upgrade path: k-shortest-path
diversification, local rerouting of the most expensive commodities,
fix-and-optimize over time windows with HiGHS on sub-MIPs (the LNS pattern —
full LNS is Stage 2).

**Validation harness.** HiGHS (via `highspy`) solves the exact LP on small and
medium instances and must satisfy `LB ≤ LP* ≤ UB` within tolerance. On large
instances (where the flat LP is intractable — ~10⁸ variables) the LB/UB
sandwich is self-certifying: the reported gap is a proven bound on
suboptimality.

## 3. Milestones (weeks 0–8)

| # | Weeks | Deliverable | Acceptance test |
|---|-------|-------------|-----------------|
| M0 | 0–1 | Repo scaffolding: CMake, CI, instance schema, time-expanded grid generator, unit-test harness | `ctest` green; generator invariants tested |
| M1 | 1–3 | Lagrangian flow relaxation: CSR graph, batched multi-threaded Dijkstra, projected subgradient with Polyak steps → valid LB | LB ≤ LP* on every validation instance; uncapacitated instances close to 0% gap in 1 iteration |
| M2 | 3–4 | Primal repair heuristic → feasible UB; gap reporting; HiGHS validation harness | `LB ≤ LP* ≤ UB` holds on the full small/medium suite |
| M3 | 4–5 | Dual method hardening: multiplier averaging → volume algorithm; step-size auto-tuning; convergence diagnostics | Gap at fixed iteration budget improves vs. M1 baseline on the benchmark suite |
| M4 | 5–6 | Exact subproblem plumbing: vendor **LEMON** (Boost license) network simplex for single-commodity min-cost-flow subproblems; vendor **KaMinPar** (MIT) and emit spatial partitions of the static graph replicated across time layers (decomposition scaffolding for Stage 2 LNS) | LEMON reproduces Dijkstra results on unit-demand cases; partitioner produces balanced blocks with small boundary cut on street graphs |
| M5 | 6–7 | Benchmark suite: instance families (grid + real street-graph import), scale ladder to ~1M arcs, reproducible runner + results table | Full suite runs unattended; results archived per commit |
| M6 | 7–8 | Stage-gate evaluation (+ stretch: fixed-charge variant via knapsack relaxation) | **≤ 2% gap on representative instances within the wall-clock budget** → Stage 2 go |

Note on vendoring (M4): this PoC environment cannot fetch LEMON/KaMinPar
(network-restricted), so the MVP ships a self-contained Dijkstra — which is
the actual Lagrangian subproblem kernel anyway. In the real Stage 1 build-out,
vendor LEMON and KaMinPar as `third_party/` submodules; neither changes the
interfaces below.

## 4. Architecture & interfaces

```
instance file (text)          # graph-agnostic: solver never sees "time"
  └─ generator: grid → time expansion → commodities   (src/arcedge/generator.*)
Instance { nodes, arcs(tail,head,cost,cap), commodities(src,dst,demand) }
  └─ Graph::build → CSR                                (src/arcedge/graph.*)
solve(instance, opts):                                 (src/arcedge/lagrangian.*)
  loop:
    reduced = c + λ
    batched_shortest_paths(reduced)   # ← the GPU boundary for Stage 2
    LB, subgradient, Polyak step, project λ ≥ 0
    every N iters: primal_heuristic(λ) → UB            (src/arcedge/primal.*)
  until gap ≤ tol
validate:  scripts/validate_lp.py  (HiGHS LP reference: LB ≤ LP* ≤ UB)
```

Design rules carried through Stage 1:

- **The solver core is time-agnostic.** Time expansion lives entirely in the
  generator; the Lagrangian core sees a plain directed graph. Rolling-horizon
  and time-window decomposition (Stage 2) then compose without touching the core.
- **`batched_shortest_paths` is the GPU seam.** It takes (CSR graph, arc-cost
  vector, commodity list) and returns distances + paths. Stage 2 swaps the
  CPU thread pool for batched delta-stepping/Bellman-Ford on GPU (FP32)
  behind the same signature.
- Struct-of-arrays / CSR layouts everywhere; multipliers, loads, and costs are
  flat `double` vectors indexed by arc id — directly transferable to GPU
  buffers.
- Uncapacitated arcs carry no multiplier and are excluded from the subgradient,
  keeping the dual dimension at the number of capacitated arcs.

## 5. Benchmark & validation plan

Instance ladder (generator is seeded/deterministic):

| Tier | Static grid | T | K | TE arcs | Reference |
|------|-------------|---|---|---------|-----------|
| small | 10×10 | 8 | 10 | ~3.2k | HiGHS exact LP |
| medium | 20×20 | 12 | 30 | ~21k | HiGHS exact LP (~6·10⁵ LP vars) |
| large | 70×70 | 45 | 250 | ~1.07M | LB/UB self-certified gap |

- Congestion is induced by hub destinations (many commodities share a sink), so
  capacity duals are genuinely active — uncongested instances trivially close
  to 0% and prove nothing.
- Correctness identities in unit tests: uncapacitated ⇒ gap 0 at iteration 1;
  hand-solvable capacitated diamond ⇒ LB and UB hit the known optimum;
  Dijkstra, street-graph, and generator invariants.
- Real street graphs (M5, **done early**): `scripts/overture_to_graph.py`
  imports Overture Maps transportation segments — endpoints become
  deduplicated nodes, every segment yields bidirectional arcs weighted by
  geodesic length in meters, largest connected component kept. The San
  Francisco extract (11,487 nodes / 36,624 directed arcs / 2,102 km) is
  committed under `data/` and wired into the E2E run at two tiers (downtown
  crop with HiGHS validation, full city at ~1.7M TE arcs).
- Lesson from real data: street networks have low-degree cuts, so demand
  concentration can make instances genuinely infeasible; the solver flags a
  diverging dual bound (an infeasibility certificate) instead of iterating
  forever. Instance calibration (capacity vs. demand envelopes) is part of
  the M5 benchmark design.

Metrics recorded per run: best LB, best UB, gap, iterations, wall-clock,
threads. Regression rule: no commit may worsen gap-at-fixed-budget by >10% on
the suite.

## 6. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Subgradient dual converges slowly on hard congestion (LB tail-off) | M3 volume/bundle upgrade; Polyak target uses best UB; α-halving on stalls already in place |
| Primal heuristic fails to find feasible routings under tight capacity | Successive shortest paths already split flow across paths; add rerouting/cancellation, then LNS repair (Stage 2); generator keeps demand within provable capacity envelopes meanwhile |
| Gap target missed on representative instances | Decision rule from the architecture study: shift weight toward stronger bounds (node-based Lagrangian, cutting planes) before considering Path B |
| HiGHS reference intractable beyond medium | Expected; validation relies on the LB/UB sandwich at scale — LB validity is *structural* (any λ ≥ 0 yields a valid bound), and is itself covered by unit identities |
| LEMON/KaMinPar integration friction | Both are header-friendly CMake projects with permissive licenses (Boost/MIT); vendored as submodules, no ABI exposure in public headers |

## 7. Out of scope for Stage 1 (deferred)

- GPU batched SSSP (Stage 2) — the seam is in place, port `batched_shortest_paths`.
- Large Neighborhood Search / fix-and-optimize with HiGHS sub-MIPs (Stage 2).
- PCST/Steiner engine: pcst_fast warm starts, dual ascent, reductions (Stage 2).
- cuOpt PDLP bounding companion, cuDSS-based IPM (Stage 3, hardware-dependent).
- Proven optimality / branch-and-bound: not a product requirement.

## 8. Proof of architecture (this repository)

The repo contains a working end-to-end PoC of M0–M2 (see `README.md` to
reproduce with `scripts/run_e2e.sh`). Measured on this environment
(4-core container, GCC 13, `-O3`):

| Instance | TE arcs | K | Result | Reference check |
|----------|---------|---|--------|-----------------|
| small (10×10, T=8) | 3,220 | 10 | gap **0.77%** in 80 ms | HiGHS LP* = 458.47 ∈ [LB 457.94, UB 461.47] ✓ |
| medium (20×20, T=12) | 21,120 | 30 | gap **0.00%** in 53 ms (21 iters) | HiGHS LP* = 2101.75 = LB = UB ✓ |
| large (70×70, T=45) | 1,065,680 | 250 | gap **0.85%** in 14.2 s | LB/UB sandwich (self-certified); pure-cost routing infeasible, λ-guided heuristic recovers feasibility |
| SF downtown, Overture real data (T=12) | 20,702 | 12 | gap **0.96%** in 0.27 s | HiGHS LP* = 27,666.26 m; LB matches LP* to 4 s.f. ✓ |
| SF full city, Overture real data (T=36) | 1,683,885 | 150 | gap **0.68%** in 9.8 s | LB/UB sandwich (self-certified) |
| SF downtown hub design (400 POIs, fixed-charge) | — | — | total **392,123** | beats the 240 s HiGHS MIP incumbent (397,618) ✓ |
| SF full-city hub design (54,921 POIs, fixed-charge) | — | — | 155 hubs + 2,152 km cable = **24,620,259** | heuristic (matheuristic, no bound at this scale yet) |

The fixed-charge edge-assignment problem family (M6 stretch) is exercised
early through `arcedge design`: POI access design over the Overture graph
(hub location + capacitated fixed-charge routing, SPH Steiner consolidation).
Exact bounding for it at full scale is Stage 2 work (knapsack relaxation /
dual ascent); at validation scale the HiGHS MIP harness covers it.

All unit tests pass (`ctest`): Dijkstra correctness, hand-computed capacitated
optimum, uncapacitated zero-gap identity, generator invariants. Both HiGHS
validations satisfy `LB ≤ LP* ≤ UB`, and gaps are inside the ≤ 2% Stage 1
threshold — the combinatorial core architecture is demonstrated end to end.

**Stage 1 exit review: PASSED.** Every validated instance is inside the ≤ 2%
gap threshold, real Overture street + places data flows through the whole
pipeline, and the fixed-charge design mode beats a time-limited HiGHS MIP
incumbent at validation scale. Performance baseline to beat (4-core CPU):
1.7M-arc time-expanded MCF at 0.68% gap in 9.8 s; full-SF access design
(54,921 POIs) at total cost 24,620,259.

---

# Stage 2 (weeks 8–16): GPU batched subproblems + LNS + Steiner engine

Per the architecture study: port the per-commodity shortest paths to GPU
(batched delta-stepping/Bellman-Ford, FP32), add Large Neighborhood Search
with fix-and-optimize sub-MIPs (HiGHS), and add the dual-ascent + reduction +
local-search Steiner engine.

> **Exit criterion: end-to-end 1M-arc instance within target wall-clock at an
> empirically validated gap ≤ 1–3%**, and LNS demonstrably improving
> design-mode solutions over the Stage 1 matheuristic baseline.

## Stage 2 milestones

| # | Weeks | Deliverable | Acceptance test |
|---|-------|-------------|-----------------|
| S2-M0 | 8–9 | **GPU-shape SSSP kernel, CPU-portable**: batched frontier Bellman-Ford (near-far/delta-stepping-ready), SoA layout, no priority queue — the exact loop structure a CUDA warp executes — behind the same `batched_shortest_paths` seam, selectable per run | bit-equal distances vs Dijkstra backend on the full validation suite; wall-clock parity report CPU-vs-CPU |
| S2-M1 | 9–11 | **CUDA backend** for the same kernel (`-DARCEDGE_CUDA=ON`): graph + costs resident on device across subgradient iterations, one commodity per block batch, FP32 with FP64 accumulation of L(λ) | compiles + equivalence-tested on a CUDA box (RTX 3090 target); ≥ 10× batch throughput vs 4-core CPU expected per literature |
| S2-M2 | 10–12 | **LNS fix-and-optimize for design mode**: cluster-neighborhood destroy/rebuild, sub-MIP per (merged) hub cluster with HiGHS warm-started from the incumbent — never worsens, strictly improves where the MIP finds better | measurable total-cost reduction on the SF design baseline (24,620,259) within a fixed time budget |
| S2-M3 | 12–14 | **LNS for time-expanded MCF**: fix-and-optimize over time windows / spatial partitions (KaMinPar blocks), HiGHS on sub-MIPs; volume-algorithm dual upgrade if the subgradient tail is the binding constraint | gap ≤ 1% on the benchmark ladder at fixed budget |
| S2-M4 | 13–16 | **Steiner/PCST engine v0**: Wong dual ascent on the directed formulation for design-mode lower bounds at full scale + bound-based reductions; pcst_fast-style GW as primal warm start | full-SF design gets a certified gap (LB, not just heuristic); reductions shrink instances ≥ 50% on benchmarks |
| S2-M5 | 16 | Stage-gate: benchmark ladder re-run, results table, go/no-go for Stage 3 GPU-LP companion | exit criterion above |

Notes:
- This dev container has no GPU; S2-M0 is deliberately CPU-portable so the
  algorithmic port is proven (equivalence, frontier behavior, memory layout)
  before touching CUDA. S2-M1 code lands compile-guarded and is exercised on
  GPU hardware via **`notebooks/arcedge_stage2_colab.ipynb`** (Google Colab,
  T4/A100): build with `-DARCEDGE_CUDA=ON`, unit tests, three-backend
  benchmark with equivalence assertions.
- Design-mode clusters are edge-disjoint by construction (SPH routing is
  region-restricted), so per-cluster sub-MIPs compose into a globally feasible
  solution — LNS neighborhoods are sound without boundary duals.

## Stage 2 progress

| Milestone | Status | Evidence |
|-----------|--------|----------|
| S2-M0 CPU DAG level-sweep backend | **done** | `--sp-backend dag`: bit-equal LB/UB with Dijkstra on the SF 1.7M-arc instance (lb 1390235.88, ub 1399750.63, 21 iters both) and already 1.26× faster on 4 CPU cores (7.9 s vs 10.0 s); unit equivalence test in `ctest` |
| S2-M1 CUDA backend | **validated on Colab T4**; readback optimization landed, needs re-run | First T4 run: FP32 LB within 3e-8 of FP64 (1390235.835 vs .880), solve reached 0.287% gap in 3.15 s vs 3.40 s CPU-dag on the same box. Profiling insight: the per-iteration K×N packed readback (~0.5 GB over PCIe) dominated — now replaced by device-side path walking (loads accumulated on GPU, ~6.7 MB back per iteration) plus FP64 re-certification of the final LB at the best multipliers |
| S2-M2 LNS for design mode | next | — |
| S2-M3 LNS for MCF | pending | — |
| S2-M4 Steiner/PCST bounds | pending | — |

Known S2-M1 caveats to close on GPU hardware: FP32 LB needs a final FP64
re-evaluation of L(λ) for a certified bound; path/load extraction currently
copies the packed array back per iteration (device-side extraction is the
follow-up optimization).

## Platform validation

| Platform | Status |
|----------|--------|
| Linux x86-64 (GCC 13, 4-core container) | canonical benchmark platform; all numbers in this file |
| Google Colab (T4 GPU, CUDA) | full E2E + CUDA backend pass |
| macOS Apple Silicon (M3, AppleClang 17) | full E2E passes unmodified; large-instance iterations ~1.3–2.7× faster than the 4-core Linux container |

**Known limitation — instances are not bit-identical across C++ standard
libraries.** The generator drives `std::mt19937` (portable) through
`std::uniform_int_distribution` / `uniform_real_distribution`, whose output
sequences are implementation-defined: libstdc++ (Linux) and libc++ (macOS)
produce *different* instances from the same seed, so objective values differ
across platforms even though every platform validates correctly against its
own HiGHS reference. Benchmark comparisons are therefore within-platform
only, with Linux as the canonical baseline. Fix scheduled with the S2-M5
benchmark suite: replace the std distributions with hand-rolled portable
ones (single-sweep renumbering of all documented baselines).
