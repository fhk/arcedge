# arcedge — Stage 1 Implementation Plan

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
  Dijkstra and generator invariants.
- Later in M5: import a real street graph (OSM extract) to confirm behavior is
  not grid-specific.

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

All unit tests pass (`ctest`): Dijkstra correctness, hand-computed capacitated
optimum, uncapacitated zero-gap identity, generator invariants. Both HiGHS
validations satisfy `LB ≤ LP* ≤ UB`, and gaps are inside the ≤ 2% Stage 1
threshold — the combinatorial core architecture is demonstrated end to end.
