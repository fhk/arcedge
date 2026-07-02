# Custom C++/CUDA Solver Architecture for Time-Expanded Flow & Edge-Assignment Problems

## TL;DR
- **Do not build a general-purpose custom LP/MIP numerical kernel from scratch.** The winning architecture for your problem class (fixed-charge MCF, PCST/Steiner connectivity, capacitated MCF on ~1M-arc time-expanded street networks, no proven optimality required) is a **combinatorial/algorithmic core**: Lagrangian relaxation with GPU-batched shortest-path subproblems + a dual-ascent/primal-dual Steiner engine + Large Neighborhood Search that calls HiGHS (or cuOpt) only as a black box on tiny sub-MIPs. This is Architecture A below.
- **Use GPU LP (PDLP via embeddable cuOpt) as an optional bounding/relaxation companion, not the primary engine**, and use cuDSS-based interior-point only if you later need high-accuracy LP bounds on structured subproblems — its direct factorization is memory-bound and the RTX 3090's 1/64 FP64 rate makes a from-scratch GPU IPM the worst effort-to-payoff option of everything evaluated.
- **Reuse, don't reinvent:** KaMinPar/KaHIP (MIT) for partitioning, LEMON (Boost) network simplex for exact flow subproblems, pcst_fast (MIT) and SCIP-Jack reduction ideas for Steiner, cuOpt (Apache-2.0) for embeddable GPU LP/MILP. All are license-compatible with closed-source commercial SaaS; cuDSS is proprietary but redistributable in binary under NVIDIA's SLA.

## Key Findings

1. **Your problem structure favors combinatorial decomposition over monolithic LP/MIP.** Fixed-charge MCF Lagrangian relaxations decompose into per-commodity shortest paths (flow relaxation) or per-arc knapsacks (knapsack relaxation), both massively parallel and GPU-friendly. Gendron/Frangioni et al. ("Node-based Lagrangian relaxations for multicommodity capacitated fixed-charge network design," *Discrete Applied Mathematics*, 2021) report that "the Lagrangian matheuristic produces almost the same upper bounds (with an average gap of 0.24% and a maximum gap of 0.72%) in much less computational times" than exact methods — far faster than solving the LP relaxation with simplex.

2. **The dual-ascent + reduction + local-search pipeline is the proven state of the art for Steiner/PCST at scale**, using no general LP. Per Rehfeldt & Koch (ZIB/UNSW), "SCIP-Jack has participated in the 11th DIMACS Implementation Challenge and been demonstrated to be the fastest solver in two categories"; its distributed variant ug[SCIP-Jack] solved three previously open SteinLib instances and updated 14 best-known solutions. Reduction techniques alone solve >90% of benchmark instances to optimality by preprocessing.

3. **pcst_fast (Goemans-Williamson) is nearly-linear time but the undirected-cut formulation gives weak bounds** — large duality gaps are reported in the literature (Hegde-Indyk-Schmidt DIMACS 2014, discussed in Ljubić et al.'s survey). It is excellent as a fast primal-solution generator/warm-start, not as a gap certificate. Directed/dual-ascent formulations (Leitner, Ljubić, Luipersbeck & Sinnl, *INFORMS J. Computing* 30(2):402–420) give much stronger bounds: "For the largest instance from the DIMACS Challenge with ≈200 000 nodes and almost 2.5 million of edges, we provide a solution of 0.05% optimality gap obtained within one hour of computing time" on a single core.

4. **cuDSS is real and fast but memory-bound.** Per NVIDIA's Technical Blog ("cuDSS Library Removes Barriers to Optimizing the US Power Grid"), MadNLP+cuDSS solved the Eastern Interconnection — "more than 70K nodes and corresponding to a 674K dimensional system" — in "less than 20 seconds" on an A100, "more than a 10x speedup over the previous state-of-the-art implementation on an AMD EPYC 7443 CPU"; the blog states the numerical-factorization-plus-solve step was accelerated ~30x by replacing HSL MA27 with cuDSS (a separate benchmark, MadNCL / arXiv:2510.05885, measures cuDSS "17 times faster than HSL MA27 at factorizing" — use ~17–30x as the range). Direct-factorization memory is the binding constraint: multi-million-variable OPF problems exceed A100 40GB and require GH200/hybrid-memory/multi-GPU modes. On near-planar ~1M-row systems with good nested-dissection ordering, fill-in should stay manageable, but no published number pins the exact VRAM — benchmark before committing.

5. **The RTX 3090 has FP64 at 1/64 of FP32.** Per the TechPowerUp GPU Database: "FP32 (float) 35.58 TFLOPS · FP64 (double) 556.0 GFLOPS (1:64)," corroborated by the NVIDIA GA102 whitepaper ("The FP64 TFLOP rate is 1/64th the TFLOP rate of FP32 operations"). This makes a from-scratch double-precision GPU interior-point method a poor fit on your primary hardware; you'd be forced into FP32/mixed-precision or FP64 emulation with accuracy/dynamic-range risk. PDLP-style first-order methods, which are FP32-friendly and only need SpMV, are the better GPU-LP fit for consumer Ampere.

6. **PDLP/cuPDLP is the pragmatic GPU LP path, with a known accuracy ceiling.** cuOpt's PDLP defaults to 1e-4 relative accuracy; its barrier to 1e-8; its dual simplex to 1e-6. Independent commentary (Gurobi's Ed Rothberg, HiGHS's Julian Hall via the GAMS blog) flags that PDLP can be faster but struggles to reach high precision on some instances. For your inexact production path this is acceptable; for tight Lagrangian dual bounds it is not.

7. **cuOpt is now Apache-2.0 and embeddable in C++** (C++ core with C/Python APIs), giving you a GPU PDLP LP solver plus GPU MILP primal heuristics (feasibility pump, feasibility jump, fix-and-propagate) without building them. Çördük et al. ("GPU-Accelerated Primal Heuristics for MIP," arXiv:2510.20499, NeurIPS 2025 ScaleOPT) report the GPU heuristic framework "achieving 221 feasible solutions and 22% objective gap in the MIPLIB2017 benchmark on a presolved dataset."

## Details

### Problem framing
Your three problem families share an arc-based, time-expanded structure on a sparse, near-planar street graph (~1M arcs after time expansion, low degree). This matters enormously:
- **Sparsity + planarity** means partitioners and planar/nested-dissection orderings work extremely well. The Lipton-Tarjan planar separator theorem gives O(√n) separators and, via Lipton-Rose-Tarjan generalized nested dissection, O(n log n) fill and O(n^{3/2}) operation count for Gaussian elimination on planar graphs. Real road networks are near-planar, so any factorization-based method (IPM) inherits good orderings, and any decomposition method inherits small boundaries.
- **Decomposability**: fixed-charge MCF and time-expanded MCF split naturally by commodity (shortest-path subproblems), by arc (knapsack subproblems), or by node. Time expansion adds a temporal axis you can decompose along (rolling time windows).

### PATH A — Custom combinatorial/algorithmic solver (RECOMMENDED CORE)

**Lagrangian relaxation for fixed-charge / capacitated MCF.** The classical relaxations are the flow relaxation (dualize coupling/capacity constraints → per-commodity shortest paths) and the knapsack relaxation (→ per-arc subproblems). Node-based Lagrangian relaxations (Gendron et al.) improve on the LP bound, which the flow/knapsack relaxations only match. Solve the Lagrangian dual with a **bundle method** (Crainic, Frangioni & Gendron show bundle methods "converge faster and are more robust" than subgradient) or the **volume algorithm** (cheaper iterations, better on the very largest instances — a volume-based branch-and-cut with a Lagrangian feasibility-pump heuristic is competitive with state of the art on large-scale fixed-charge MCF, per Ann. Oper. Res. 2024). Lagrangian matheuristics reach 0.24% average / 0.72% max gap on benchmark fixed-charge MCF.
- **GPU acceleration**: the per-commodity shortest-path subproblems are the workhorse. Batched Bellman-Ford/Dijkstra/delta-stepping on GPU (Gunrock-style frontier processing) gives order-of-magnitude speedups; one CUDA study on a New York road network (400k vertices / 1M edges) reduced SSSP runtime by 99.4% — from ~3.25 hours sequential to ~16 seconds. Thousands of commodities per subgradient iteration map perfectly to the GPU.

**PCST/Steiner connectivity.** Use the SCIP-Jack recipe without the MIP: Wong dual ascent on the *directed* formulation for strong lower bounds + dual guidance, aggressive reduction tests (bottleneck Steiner distance, degree tests, bound-based reductions), then local search (key-path exchange, vertex insertion/swap) and a recombination/path-relinking phase. pcst_fast (nearly-linear GW, O(d·|E|·log|V|)) is a fast primal generator for warm starts. Evidence: reduction techniques alone solve >90% of benchmark instances; SCIP-Jack was fastest in two DIMACS 2014 categories; dual-ascent B&B achieves 0.05% gap on a 200k-node/2.5M-edge instance in one hour single-core; Feofiloff et al.'s corrected JMP gives an O(n² log n) 2-approximation.

**LNS / fix-and-optimize over time windows.** Wrap the above in a Large Neighborhood Search: fix most binary (arc-open) decisions, free a spatial partition or a time window, and re-optimize the small subproblem exactly with HiGHS or cuOpt as a black-box MIP. MIP-neighborhood-search heuristics for service network design (Katayama, *J. Heuristics* 2020) find high-quality solutions in short times; this is the standard matheuristic pattern and keeps an exact solver in the loop only where it's cheap.

### PATH B — Custom GPU LP/MIP numerical kernel (SELECTIVE USE ONLY)

**cuDSS (proprietary, redistributable under NVIDIA SLA).** Sparse Cholesky/LDLᵀ/LU on GPU; three phases (symbolic, numeric, solve) with a cuSPARSE-like API; supports hybrid host/device memory, INT64 indexing, and multi-GPU modes to spill factors beyond VRAM. It powered the 17–30x factorization speedups behind MadNLP OPF. For your problem it would only make sense inside an interior-point method for high-accuracy LP bounds on well-structured subproblems. Caveats: (1) memory-bound — direct factors can exceed 24GB VRAM at multi-million-variable scale; (2) closed source (redistributable in binary only, with "material additional functionality," per NVIDIA's SLA); (3) needs FP64, which the RTX 3090 does poorly. Note the symbolic phase is a bottleneck (cuDSS symbolic factorization is ~2x slower than MA27), amortized over IPM iterations.

**Custom IPM.** The reference open design is HiPO (Zanetti & Gondzio, 2025), the new factorization-based regularized IPM in HiGHS: it "uses a direct factorisation to solve the Newton systems, choosing the best approach between the normal equations and augmented system," with multifrontal factorization and static pivoting + regularization for stability. On energy-model LPs, factorizing the augmented system beats normal equations. This is exactly the design you'd copy — but it already exists, is open source (Apache-2.0 in HiGHS-with-HiPO), and is CPU-parallel. Building your own GPU version on a 1/64-FP64 card is not justified.

**PDLP / cuPDLP from scratch.** cuPDLP.jl and its C successors (cuPDLP-C, cuPDLPx) implement restarted PDHG — averaged in cuPDLP, then Halpern in cuPDLPx — with adaptive restart (KKT-error restarting), adaptive step size, and PID-controlled primal weight. cuPDLPx reports "2.5×–5× speedups on MIPLIB LP relaxations and 3×–6.8× on Mittelmann's benchmark set." Accuracy ceiling is real (1e-4 default; high precision hard). Reimplementing PDLP in CUDA C++ is moderate effort but largely redundant given cuOpt/cuPDLPx are open source and embeddable. Network LPs with wide coefficient ranges (from time-expanded costs) are exactly where FOMs' conditioning weaknesses bite.

**Network-simplex / min-cost-flow specialized kernels.** For single-commodity flow subproblems, LEMON's network simplex (Boost license) is among the fastest available; Király & Kovács found LEMON's network simplex "one of the most competent algorithms to solve the MCFP in large-scale networks," outperforming CPLEX's network optimizer in several studies, with cost-scaling competitive with CS2. Use LEMON as the exact flow subproblem solver inside Lagrangian/LNS loops rather than any general LP. GPU min-cost-flow is immature; don't build it.

### PATH C — cuOpt-embedded hybrid
Embed cuOpt (Apache-2.0) as your GPU LP/MILP black box: PDLP for LP relaxations/bounds, GPU MILP primal heuristics (feasibility pump, feasibility jump, fix-and-propagate on GPU; B&B on CPU). cuOpt's MILP heuristics-only mode reached 221 feasible solutions and 22% objective gap on presolved MIPLIB2017 — decent for a generic GPU heuristic but not competitive with a problem-specific combinatorial core. Best used as the LNS sub-solver and as a bounding companion.

### Partitioning & decomposition infrastructure
- **KaMinPar vs KaHIP vs METIS.** KaMinPar (deep multilevel, MIT) is "at least an order of magnitude faster than competing algorithms" at large block counts while matching quality, "consistently produces balanced solutions," and scales on 64 cores (~5x faster than Mt-KaHiP / ~4.4x faster than Mt-Metis at k up to 64). KaHIP (KaFFPa/evolutionary, MIT) gives the highest quality when you can spend time (it has improved best-known Walshaw benchmark results). METIS (Apache-2.0 in 5.x) is the baseline. Recommendation: KaMinPar for fast production partitioning, KaHIP for offline high-quality reference partitions.
- **Time-expanded partitioning.** Two strategies: (a) partition the static spatial graph once, then replicate the partition across time layers (cheap, exploits temporal regularity, keeps boundary arcs consistent); (b) partition the full time-expanded graph (better cut, more expensive). Start with (a). Coordinate boundaries with Lagrangian price coordination on boundary arcs, ADMM consensus, or a Benders-style master.
- **Planar separators / H3.** Lipton-Tarjan separators or H3 spatial cells give geometry-aware partitions that respect road structure; nested-dissection ordering from separators is what makes any factorization cheap. Use these for the ordering even if you use KaMinPar for the block decomposition.

### Hardware reality check (single node)
- **RTX 3090**: 24GB GDDR6X, 936 GB/s, 35.58 TFLOPS FP32, **0.556 TFLOPS FP64 (1:64)**. Great for FP32 SpMV / first-order / graph primitives; poor for FP64 factorization. A future A100/H100 flips this (strong FP64 + 40–80GB), which is when a cuDSS-IPM path becomes attractive.
- **64-core CPU / 128GB RAM**: run KaMinPar/KaHIP, LEMON, bundle method, and HiGHS sub-MIPs here; be NUMA-aware (pin partition blocks to sockets, first-touch allocation). Use CSR / structure-of-arrays layouts, pinned host memory for CPU-GPU transfer, and keep the time-expanded graph resident on GPU in CSR to avoid repeated PCIe transfers across subgradient iterations.

### Library stack & licensing (commercial closed-source SaaS)
| Component | Library | License | SaaS-safe? |
|---|---|---|---|
| Graph partitioning (fast) | KaMinPar | MIT | Yes |
| Graph partitioning (quality) | KaHIP | MIT | Yes |
| Nested-dissection ordering | METIS 5.x | Apache-2.0 | Yes (avoid **ParMETIS** — non-commercial only) |
| Network simplex / min-cost flow | LEMON | Boost 1.0 | Yes |
| PCST warm-start | pcst_fast | MIT | Yes |
| Steiner reductions/dual ascent | SCIP-Jack / SCIP ≥8.0.3 | Apache-2.0 | Yes (SCIP <8.0.3 proprietary; Zimpl/UG components are LGPL) |
| Black-box sub-MIP | HiGHS | MIT (Apache-2.0 if built with HiPO) | Yes |
| GPU LP/MILP | cuOpt | Apache-2.0 | Yes |
| GPU PDLP reference | cuPDLPx / cuPDLP-C | Apache-2.0 / MIT | Yes |
| GPU sparse direct solver | cuDSS | Proprietary (NVIDIA SLA) | Redistributable in binary only, not open source |
| GPU graph primitives | Gunrock | Apache-2.0 | Yes |
| Sparse LA / CSR | cuSPARSE, Eigen, SuiteSparse | proprietary-CUDA / MPL2 / mixed | cuSPARSE & Eigen fine; check SuiteSparse module licenses (some GPL) |

## Recommendations

**Stage 1 (weeks 0–8): Combinatorial core MVP.** Build the Lagrangian decomposition for capacitated MCF with CPU subgradient/volume first, using LEMON for flow subproblems and KaMinPar for partitioning. Validate lower/upper bounds against HiGHS LP relaxation on medium instances. *Threshold to proceed: ≤2% empirical gap on representative instances.*

**Stage 2 (weeks 8–16): GPU batched subproblems + LNS.** Port per-commodity shortest paths to GPU (batched delta-stepping/Bellman-Ford, Gunrock or custom CUDA, FP32). Add LNS with time-window/spatial-partition fix-and-optimize calling HiGHS on sub-MIPs. Add the dual-ascent + reduction + local-search Steiner engine (fork pcst_fast for warm starts; implement SCIP-Jack-style reductions). *Threshold: end-to-end 1M-arc instance within target wall-clock at empirically validated gap ≤1–3%.*

**Stage 3 (optional, weeks 16+): GPU LP companion.** Embed cuOpt PDLP for fast relaxation bounds and as an LNS sub-solver. Only if you need tighter bounds AND move to A100/H100, prototype a cuDSS-based IPM (or simply use HiGHS HiPO on CPU). **Do not build a from-scratch GPU IPM on the 3090.**

**What changes the recommendation:**
- Gap tolerance tightens to <0.5% with proof needs → shift weight to SCIP-Jack/branch-and-cut and cutting planes (Path B bounds matter more).
- Hardware moves to A100/H100 (strong FP64, 40–80GB) → cuDSS-IPM becomes viable for high-accuracy bounds; re-evaluate Path B.
- Development time is the hard constraint → skip custom kernels; wrap cuOpt + HiGHS with your Lagrangian/LNS orchestration (closest to your prior report's recommendation, but with a combinatorial outer loop instead of pure Benders).
- Instances grow >10M arcs → PDLP's factorization-free memory profile becomes decisive; lean harder on Path C.

**Decision matrix**

| Criterion | A: Lagrangian+GPU SP+LNS | B: Custom GPU LP/IPM | C: cuOpt-embedded hybrid |
|---|---|---|---|
| Expected solution quality (your problem) | High (0.2–3% gap) | Medium-High (bound quality) | Medium |
| Runtime at 1M arcs on 3090 | Best | Poor (FP64 bottleneck) | Good |
| Engineering effort | Medium-High | Very High | Low-Medium |
| Risk | Medium | High | Low |
| Maintainability | Medium (your IP) | Low (numerical fragility) | High (vendor-maintained) |
| Best when | Production default | High-accuracy bounds + A100/H100 | Fast time-to-market |

## Caveats
- Gap figures (0.24% Lagrangian matheuristic, 0.05% dual-ascent Steiner, 22% cuOpt heuristics-only) are from published benchmarks on **different problem instances** than yours; treat them as directional and validate empirically on your fiber/broadband time-expanded instances.
- pcst_fast's speed comes with weak bounds (undirected-cut formulation); use it for primal warm starts, not gap certification.
- cuDSS VRAM footprint for a specific 1M-row near-planar KKT system is **not published**; benchmark before committing. Direct factorization is the memory-limiting step, and cuDSS is closed-source ("we have no insight into the details of the algorithm," per the multi-period OPF study).
- The cuDSS-vs-MA27 factorization speedup is reported as ~30x for the factorization-plus-solve step (NVIDIA blog) but ~17x for factorization alone (MadNCL); treat ~17–30x as the range.
- cuOpt/PDLP accuracy is ~1e-4 by default; unsuitable where you need high-precision duals for exact Lagrangian bounds.
- FP64 emulation on consumer GPUs exists but loses effective precision and dynamic range — do not rely on it for IPM stability.
- License notes are current as of mid-2026: SCIP is Apache-2.0 only from v8.0.3 (Zimpl/UG remain LGPL); ParMETIS (distinct from METIS) is restricted to non-profit/government use; cuDSS is proprietary. Re-verify all license terms at integration time.