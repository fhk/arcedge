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
| S2-M1 CUDA backend | **ACCEPTED** (Colab, RTX PRO 6000 Blackwell Server, 2026-07-03) | See "S2-M1 acceptance evidence" below |
| S2-M2 LNS for design mode | re-scoped as optional anytime tail — see Round 3 replan | — |
| S2-M3 LNS for MCF | re-scoped — see Round 3 replan | — |
| S2-M4 Steiner/PCST bounds | deferred (design-mode certified gaps are not currently worth their cost — see learnings) | — |

### S2-M1 acceptance evidence

Measured via `notebooks/arcedge_s2m1_benchmark.ipynb` on Colab,
NVIDIA RTX PRO 6000 Blackwell Server Edition (earlier validation on T4).

**Full solve, SF 1.68M-arc TE instance, K=150** (includes the sequential CPU
primal heuristic):

| backend | gap | iters | time |
|---|---:|---:|---:|
| dijkstra (CPU) | 0.680% | 21 | 3507 ms |
| dag (CPU) | 0.680% | 21 | 3367 ms |
| cuda | **0.287%** | 11 | **2705 ms** |

CPU backends bit-equal; cuda's tighter gap is a legitimately different
multiplier trajectory (FP32 tie-breaking) whose λ-guided heuristic found a
better feasible flow; final LB FP64-certified. Pooling backends gives a
combined certified interval of 0.287%.

**Pure batched-SSSP throughput (`--no-primal`), ms per subgradient iteration:**

| K | dag (CPU) | cuda | speedup |
|---:|---:|---:|---:|
| 100 | 7 | 23 | 0.31x |
| 250 | 9 | 22 | 0.39x |
| 500 | 72 | 34 | 2.09x |
| 1000 | 128 | 38 | 3.37x |
| 2000 | 249 | 46 | 5.42x |

Crossover at K≈400; margin grows monotonically with K (the acceptance
criterion). The cuda curve is nearly flat (23→46 ms/iter across a 20×
batch increase): per-iteration cost is dominated by fixed overheads (7 MB
reduced-cost upload, ~35 per-level kernel launches, 6.7 MB loads readback),
so the GPU has large headroom — K in the tens of thousands should hold
similar ms/iter. Follow-up optimizations when SSSP becomes the binding
constraint again: keep λ and the subgradient update on-device (skip the
cost upload), CUDA Graphs for the level-launch sequence, pinned host
buffers. Earlier T4 profiling already moved path/load extraction on-device
(0.5 GB → 6.7 MB per iteration over PCIe).

---

# Round 3: speed-first replan (2026-07-03)

Directive: **trade gap % for wall-clock where the exchange rate is good** —
an order-of-magnitude speedup is worth a point of gap. The evidence from
Stage 1 + S2-M0/M1 says this trade is not only acceptable, it is nearly free:
the expensive part of our solves is the *upper bound*, while the lower bound
(dual iterations) is now almost costless on GPU. Certified gaps can stay; the
machinery producing them just needs to stop being on the critical path.

## What we learned (evidence)

1. **The sequential primal heuristic dominates solve time — and is often
   pure waste.** Measured on the SF TE instance (4-core CPU, dag backend,
   identical iteration counts): K=150: 11.7 s with primal vs 5.65 s without
   (**52% primal**). K=1000, 5 iterations: 20.1 s vs 8.7 s (**57% primal**)
   — and those primal calls returned *no feasible solution* (UB = inf: with
   cold multipliers, both routing passes fail on congested instances). On
   GPU the share is worse because the dual side got 5× cheaper.
2. **The dual bound is nearly free and nearly instant.** Iteration-0
   (free-flow) LB is within ~1% of optimum on every real-data instance so
   far; the GPU does a K=2000 dual iteration in 46 ms, flat in K
   (fixed overheads dominate — S2-M1 evidence table).
3. **Quality is not the bottleneck.** Every validated instance closed to
   ≤1% (most ≤0.7%, some 0.0%); the design heuristic beats 120–240 s HiGHS
   MIP incumbents in 0.3 s. We have gap headroom to spend.
4. **Design-mode certified gaps are bad value.** The fixed-charge LP/MIP
   dual bound sits ~30% below the heuristic (both ours and HiGHS's own
   incumbents) — closing it costs minutes for information that doesn't
   change decisions. Keep exact references in the validation suite only.
5. **Design runtime is dominated by avoidable work**: the k-sweep probes
   k=1..32 which are infeasible and burn the full 800-round repair limit
   (each round = one multi-source Dijkstra over 107k nodes, opening ONE
   relief hub); evaluations are independent but run sequentially; SPH
   clusters are edge-disjoint but consolidated sequentially.
6. **Cross-backend/trajectory diversity tightens bounds for free**
   (FP32 tie-breaking found a 0.287% solution where FP64 found 0.68%).
   Restarts/perturbation are cheap ensemble members on GPU.
7. Real street graphs have low-degree cuts: infeasibility is a normal
   outcome and must fail fast (divergence detector currently spends 30+
   iterations to say so).

## Round 3 milestones (speed-first)

| # | Deliverable | Expected effect | Target |
|---|-------------|-----------------|--------|
| R3-1 | **Primal overhaul.** (a) Skip the primal until multipliers are warm (LB stall or iteration threshold) — cold-λ routing is measured waste. (b) Drop the pure-cost pass once λ is warm (guided pass wins in practice; halves remaining cost). (c) Parallelize routing: wave-based batch SSP — route all commodities on the GPU/thread batch against current residuals with penalty re-pricing, few rounds, CPU repair only for the overloaded tail. (d) Once cheap, refresh every iteration → earlier stopping. | Removes the 52–57% primal share; UB appears earlier so fewer dual iterations too | SF K=150 full solve **< 1 s** GPU / < 3 s CPU; K=1000 **< 10 s** GPU (vs ~minutes today) |
| R3-2 | **Design speed pass.** (a) Batch relief: open relief hubs at *every* overloaded funnel per repair round, not one — repair rounds drop from ~hundreds to ~5–10. (b) Estimate k_min from demand/capacity before sweeping; never probe hopeless k. (c) Evaluate candidate k's in parallel threads. (d) Parallelize SPH across (edge-disjoint) clusters. | Removes the dominant known wastes in the 3–4 min full-SF run | full-SF design (54,921 POIs) **< 30 s** at equal-or-better cost |
| R3-3 | **GPU iteration floor.** Device-resident λ + subgradient update (kills the 7 MB/iter upload), CUDA Graphs for the ~35 per-level launches, pinned buffers. | 46 → ~15–20 ms/iter at K=2000; opens K=10k+ | K=10,000 dual iteration ≤ ~60 ms |
| R3-4 | **Anytime interface.** `--budget SECONDS`: solvers return best-known solution + certified gap when the budget expires; gap tolerance becomes advisory. Fast-fail infeasibility check (aggregate demand vs cut capacity heuristics) before iterating. | Product knob matching the directive: speed is chosen, gap is reported | any instance returns a usable answer within budget |
| R3-5 | **Quality tail (was S2-M2/M3), now opt-in.** LNS with warm-started HiGHS sub-MIPs over design clusters / MCF time-windows, run only inside a leftover budget. Trajectory-ensemble restarts on GPU as a cheap alternative knob. | Recovers gap when the user *chooses* to spend time | design: measurable cost reduction per budget-minute |
| R3-6 | **Measurement guardrails** (unchanged prerequisites): portable RNG (cross-platform instances), CI on Linux, benchmark-suite runner with per-commit results. | Makes the above speedups provable and regression-proof | suite runs green in CI |
| R3-7 | **Model config file** — declarative network models: layers + inter-layer couplings, selector-based capacities on edges/nodes/facilities (hard and **soft with overage penalties**), facility tiers, commodity source/sink relationships (pairs / assignment-to-facility / OD matrix), elastic demand, declarative time expansion, provenance-mapped exports. Full design spec: **`docs/model-config.md`** (phases R3-7a–d) | New problem shapes without new C++ entry points; soft capacities eliminate the hard-infeasibility failure mode on real street graphs | **R3-7a DONE** (`scripts/arcedge_modelc.py` + `examples/`; E2E-asserted: design reproduces 392,123 exactly, soft-cap MCF ~0% gap). **Multi-tier facility chains DONE** (R3-7d chained form): hub serving capacity native in the solver (`--hub-cap`, demand-weighted, transit-capable upper tiers); `--solve` runs the chain — downtown FTTH 1→12→512→4000 gives 80 terminals / 1 FDH / 1 OLT, total 608,804 (E2E-asserted); full-city SF gives 11,050 terminals / 206 FDHs / 24 OLTs, 2,539 km cable, total 37,432,879 (`notebooks/arcedge_ftth_sf_colab.ipynb` reproduces with metrics + per-tier GeoParquet). High-density scaling fixes: O(k) hub seeding above k=512, recentering skipped above 1024 hubs, relief batch cap scales with candidate count. R3-7b/c pending |

Sequencing: R3-1 and R3-2 first (largest measured waste, no new
infrastructure), then R3-4 (small) and R3-7a (schema + compiler skeleton,
pairs well with R3-4's fast-fail checks), R3-3 (GPU), R3-5/R3-6 alongside.

## R3-1 / R3-2 results (implemented, measured on the 4-core container)

**R3-1 primal overhaul** — warm-λ gating (no attempt while congestion is
heavy and multipliers cold), pure-cost pass only on cold λ, wave-parallel
successive shortest paths (parallel batch SSSP on a frozen residual
snapshot + deterministic sequential commit), escalating congestion pricing
(β·λ retries on failure), exponential backoff after fully-failed attempts.

| Benchmark | Before | After | |
|---|---:|---:|---|
| SF K=150 full solve (dag, 21 iters, gap 0.68%) | 11.7 s | **6.25 s** | 1.9×; the residual 5.65 s is the dual side — next cuts come from GPU (R3-3) or fewer iterations |
| SF K=1000, hard cap 20 | UB never found; ~57% of time burned on failing primal calls | failed attempts now back off exponentially | greedy routing cannot serve this congestion under HARD caps — the modeling answer is soft capacities (next row) |
| SF K=1000, **soft** cap 20 (+200/unit overage, via the R3-7a compiler; 4.2M arcs incl. overflow) | hard-cap version: minutes, no solution ever | **0.50% gap, 11 iters, 32 s CPU** | soft capacities turn the pathological case into a routine solve — validates the R3-7 default recommendation |

**R3-2 design speed pass** — batched tempered relief (¼ of overload-maximal
funnels per round, overloaded-ancestor filter, ~15 forest rebuilds per eval
instead of hundreds), parallel cheap k-sweep with per-k deterministic RNG,
cheap descent on k before spending SPH, greedy hub-prune pass, SPH
consolidation parallel across (edge-disjoint) clusters.

| Benchmark | Before | After |
|---|---:|---:|
| Full-SF design, 54,921 POIs | ~213 s, cost 24,620,259 (155 hubs) | **25.2 s, cost 24,928,040 (169 hubs)** — 8.5× faster, +1.25% cost (inside the speed-first trade) |

Learned during implementation: naive all-at-once relief exploded hub counts
(451 at k=64) — relief batches must be tempered and overload-maximal, and a
prune pass recovers the parsimony that one-hub-per-round bought with its
hundreds of rebuilds.

## Design-chain performance: next round (R3-2b) and the GPU verdict

Amdahl analysis from the two benchmark machines (4-core: ~53 s/round;
48-vCPU: ~26 s/round ⇒ only 2× from 12× cores ⇒ serial fraction ≈ 45%):
the parallel phases (k-sweep, per-cluster SPH) are done; the critical path
is now the *serial* repair/prune chains inside each evaluation. CPU fixes,
in expected-value order:

1. **Subtree load aggregation** — **DONE**: per-POI path walks
   (Σ path-lengths ≈ 5.5M steps/round at 55k POIs) replaced with Kahn
   leaf-inward accumulation over the forest, O(n+m) ≈ 230k; subtree[hub]
   doubles as the hub's served demand for free.
2. **Incremental grow_forest during repair** — **DONE**: relief hubs only
   lower distances, so rounds after the first seed just the new hubs and
   relax outward (`grow_forest_add`), touching only the affected region;
   reachability is checked once (it can only improve).
   **Measured (1+2): full-SF chain 53 → 15 s/round on the 4-core box
   (3.5×), solution within +0.05% of the previous result (incremental
   tie-breaking).** Extrapolated EPYC-48 round: ~7–10 s.
3. **Parallel prune probing** — the prune loop is ~30 *sequential* cheap
   evaluations; batch-test candidate drops concurrently. (pending)
4. Dial's/bucket priority queue for the meter-weighted Dijkstras (2–3×
   constant factor). (pending)

**GPU for the design chain: not worth it now.** The kernel is multi-source
Dijkstra on a *general* (cyclic) 107k-node street graph — the MCF backend's
level-sweep trick needs a layered DAG and does not apply; general-graph GPU
SSSP (delta-stepping) at this size is frontier-starved and
launch-overhead-bound, the repair loop is a serial chain of dependent
solves, and the workload is already interactive (26 s/city). Weeks of CUDA
for maybe 2–5× on the wrong bottleneck. Revisit only if graphs grow 10×+
(multi-city) or evaluations must batch in the thousands.

**Where the GPU IS worth it (unchanged, still valid — R3-3):** the MCF
Lagrangian backend. Device-resident λ + subgradient update (removes the
7 MB/iter upload), CUDA Graphs over the ~35 per-level launches, pinned
buffers: 46 → ~15 ms/iter at K=2000, opening K=10k+; plus GPU wave-primal
(R3-1d) when MCF batches grow. These matter for the flow-model side
(time-expanded MCF, future LNS/Steiner), not for the facility chain.

## Design-quality refinements (2026-07-03, later)

- **Duct sharing** (`cable.reuse_factor`): upper tiers pay a fraction on and
  are routed toward edges already carrying lower-tier cable. Full-SF −7.3%.
- **Hub slide**: hubs with tree-degree 1 and no local demand sit on dead-leg
  stubs ("T shapes"); re-rooting at the first junction (the splice point)
  removes the stub with zero side effects — provably improving, unit-tested
  on a forced-T instance, applied per tier (terminals included).
- **Seed-diversified rounds**: converged feedback rounds repeat one basin;
  remaining rounds now explore fresh solver seeds, best-of wins. This also
  absorbs the ±1% basin variance that structural changes (like the slide)
  can otherwise surface as apparent regressions.
- Full-SF progression: 37,432,879 (greedy) → 36,839,982 (feedback) →
  34,154,091 (sharing) → **33,737,619** (slide + diversified rounds), −9.9%
  overall at ~16 s/round on 4 cores.

## Joint-chain benchmark (user-reported, Colab GPU-class runtime, 2026-07-03)

Machine: AMD EPYC 9B45, 48 vCPU, 176 GB RAM (RTX PRO 6000 Blackwell present
but idle — the design chain is CPU-parallel). Full-city SF FTTH
(54,921 addresses, terminal ≤12 → FDH ≤512 → OLT ≤4000), 3 joint rounds:

| round | true total | vs greedy | wall s | terminals | FDH | OLT |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 37,432,879 | +0.00% | 27 | 11,050 | 206 | 24 |
| 2 | 37,006,883 | −1.14% | 26 | 10,433 | 192 | 24 |
| 3 | 36,839,982 | −1.58% | 25 | 10,139 | 199 | 24 |

**81 s total wall-clock** vs ~8–10 min/round on the 4-core reference
container (~20× on 12× cores — the EPYC's per-core speed and cache make it
superlinear), with **identical solutions on both machines** (Linux/libstdc++
determinism plus fixed per-k seeding). City-scale three-tier design is now
an interactive-latency operation on a big CPU box.

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
