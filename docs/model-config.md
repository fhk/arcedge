# arcedge model configuration — design spec (v1)

Status: **R3-7a implemented, plus multi-tier facilities (R3-7d chained
form)** — `scripts/arcedge_modelc.py` compiles: one street layer, demand
layers from CSV/GeoParquet, `nearest_edge_split` couplings, hard/soft edge
capacities (soft compiled as overflow arcs), assignment / pairs / sampled
commodities, elastic demand via `unserved_penalty`, declarative time
expansion — and **facility tier chains**: tiers declared in serving order
(e.g. address → terminal ≤12 → FDH ≤512 → OLT ≤4000) run as chained design
passes via `--solve`, each pass's opened facilities becoming the next pass's
demand-weighted terminals. Facility *serving* capacity is native in the C++
solver (`--hub-cap`, over-capacity hubs shed their heaviest child subtree
during repair), demands are weighted, and upper-tier demand nodes are
transit-capable (`--demand-transit`). Checked-in examples under `examples/`
are exercised by E2E step 10.

**Joint-aware chains** (`--rounds N`): the exact joint multi-tier MIP is
intractable at city scale (a single tier already stalls HiGHS at 400
addresses), so joint awareness is delivered as measured-cost feedback:
after each chain round, tier i's open cost is inflated by the marginal
upper-tier cable per facility actually paid (upper open costs propagate
through successive rounds), the chain re-solves, and the best round by
true cost wins — provably never worse than the greedy chain. Full-SF:
-1.6% total cost in 3 rounds (terminals 11,050 → 10,139, feeder
619 → 571 km). Remaining phases: R3-7b (OD matrices, node capacities),
R3-7c (multiplier-clamp soft caps, overage exports); an exact joint
formulation stays roadmap-only for small instances.

Schema note learned in implementation: the capacity-rule selector key is
**`where:`** (the spec originally used `on:`, which YAML 1.1 parses as the
boolean `True`; the compiler tolerates both). Commodity sampling uses
Python's `random.Random(seed)`, which is platform-portable — config-driven
instances do not suffer the C++ `std::uniform_*` cross-platform divergence.

## Side-of-street, splices, dual-ascent bounds (R3-10, implemented)

```yaml
layers:
  streets:
    source: { file: data/sf_streets.graph }
    sides:
      enabled: true
      mid_block_crossing_cost: 800   # money per bore; null/omitted = forbidden
      offset_m: 3                    # cosmetic L/R offset for exports

splices:
  cost: 150                 # per branch at a non-hub tree node (degree >= 3)
  mid_block_surcharge: 100  # extra when the branch sits mid-block
```

**Sides.** Each street edge becomes two side chains `u–mid–v` (the midpoint
keeps the parallel copies distinct through edge dedup in every reader).
Chains share the original intersection nodes, so **corner crossings are
free**; every drop foot splits *both* sides and gets a mid-block crossing
edge between the two feet at cost-equivalent length
`mid_block_crossing_cost / cable.fixed_cost_per_m` — the optimizer chooses
between trenching the second side and boring across. Drops attach to the
chain on their geometric side of the centerline; feet within `snap_m`
share one station (stacked address points must not spawn ~0-length
micro-station chains — unmerged, capacity relief bounces between them and
the tier goes spuriously infeasible). Midpoints, feet, and POI nodes are
flagged mid-block in the emitted graph (`v id lon lat m`). Expect roughly
2× nodes / 2.3× edges and **higher, realism-corrected totals** — the
centerline model undercounts double-side service.

**Splices.** A cable branch at a non-hub tree node of degree ≥ 3 costs
`splices.cost` per extra branch (a degree-d node carries d−2), plus the
surcharge at mid-block nodes; hubs are exempt (a hub IS a splice cabinet).
Charging is exact in the solver's objective (`--splice-cost`,
`--splice-mid-surcharge`); SPH tree growth is steered away from creating
new branches near existing ones by pricing branch-creating steps
(approximate; the exact charge decides all accept/reject comparisons).

**Dual-ascent cable bounds.** Wong dual ascent runs per cluster where SPH
consolidation runs: it yields a per-cluster **lower bound on cable meters**
(summed and reported as `cable_lb` in result files, `cable_lb_m` /
`tree_gap_pct` per tier in `tiers_summary.json`) and a candidate
replacement tree, adopted when it beats the SPH tree (uncapacitated edges
only). The bound is **conditional on the hub set + POI assignment** — it
certifies tree quality, not global optimality — and is clamped after hub
slides (re-rooting invalidates the old-root bound at stub-length scale).
Measured on downtown SF FTTH: terminal-tier gap 0.0–0.3%, FDH gap 4–11%;
the centerline chain total is unchanged (546,379) while the sided chain
lands at 621,895 (+13.8% realism correction, 393 crossing options, 240
terminal-tier splices).

## Goals

- One declarative file specifies a **network model**: layers, entities,
  capacities, commodities, facilities, objective — decoupled from *run*
  parameters (budget, backend, tolerance) and from *data* (graphs, parquet).
- Everything the CLI currently hard-codes (`gen` flags, `design` flags, the
  POI-drop convention, time expansion) becomes config, so new problem shapes
  stop requiring new C++ entry points.
- The config compiles down to the existing solver cores (Lagrangian MCF,
  design matheuristic) — the solvers stay dumb and fast; the *compiler*
  carries the modeling generality.

## Format

- **Authoring format: YAML** (comments, multi-line, human diffs).
- **Canonical format: JSON** with a versioned schema — what the C++ binary
  actually reads (small vendored JSON parser; YAML never touches C++).
- `arcedge-modelc` (Python) validates YAML, resolves data references and
  selectors, applies graph transforms (time expansion, node splitting, POI
  drops), and emits: canonical JSON + the compiled instance files + a
  compilation report (entity counts, demand/capacity envelopes, fast-fail
  feasibility warnings).
- Determinism: any sampling in compilation uses the portable RNG (R3-6);
  same config + same data ⇒ bit-identical instance on every platform.

## Schema overview (YAML, worked FTTH-style example)

```yaml
model_version: 1
name: sf-access-design

# ---------------------------------------------------------------- layers --
# A layer is a named graph. Sources: file (arcedge .graph), geoparquet
# (Overture importer inlined), generated (grid), or a transform of another
# layer. Inter-layer edges are declared as "couplings".
layers:
  streets:
    source: { file: data/sf_streets.graph }          # or geoparquet: {...}
    directed: false                                   # both directions built
  demand:
    source: { geoparquet: data/places_sf.parquet, geometry: point }
    role: terminals                                   # nodes are leaves: no transit
  backbone:                                           # optional higher tier
    source: { file: data/sf_backbone.graph }

couplings:
  - name: drops              # today's connect_pois.py, as config
    from: demand
    to: streets
    method: nearest_edge_split   # perpendicular foot point, break the edge
    max_length_m: 500            # drop longer than this -> warning + skip/penalize
    edge: { cost_per_m: 10, fixed_cost_per_m: 10, capacity: 1 }
  - name: uplinks
    from: streets
    to: backbone
    method: colocated_nodes      # join nodes within tolerance_m
    tolerance_m: 1.0

# ------------------------------------------------------------- selectors --
# Named node/edge sets used everywhere below. Composable predicates:
# layer, tag (carried from source data, e.g. Overture class), bbox, id list.
selectors:
  arterials:   { edges: { layer: streets, tag: { class: [primary, secondary] } } }
  poi_nodes:   { nodes: { layer: demand } }
  hub_sites:   { nodes: { layer: streets, not_role: terminals } }

# ------------------------------------------------------------ capacities --
# Rules apply in order; later rules override. Capacity semantics:
#   hard: cap                      -> flow <= cap (current behavior)
#   soft: cap + penalty_per_unit   -> flow may exceed cap; overage costs
#         penalty (piecewise-linear; optional hard ceiling `max`)
# Node capacity = throughput cap, compiled via node splitting (in->out arc).
capacities:
  - where: { edges: { layer: streets } }
    capacity: { soft: 500, penalty_per_unit: 40, max: 800 }
  - where: arterials
    capacity: { soft: 1000, penalty_per_unit: 25 }
  - where: { nodes: { layer: streets } }
    capacity: { hard: 2000 }                # node throughput

# ------------------------------------------------------------ facilities --
# Facility tiers generalize "hubs": open-cost, serving capacity, allowed
# sites, count bounds. Multiple tiers = hierarchical location (cabinet -> CO).
facilities:
  cabinet:
    sites: hub_sites
    open_cost: 20000
    capacity: { soft: 500, penalty_per_unit: 100 }   # units served
    count: { min: 0, max: unlimited }
  central_office:
    sites: { nodes: { layer: backbone } }
    open_cost: 250000
    capacity: { hard: 20000 }
    count: { min: 1 }

# ----------------------------------------------------------- commodities --
# Source/sink relationships. Three forms, mixable:
#   pairs:      explicit (src, dst, demand) triples or a CSV/parquet ref
#   assignment: every node in `from` ships demand to ANY open facility of
#               `tier` (today's design mode)
#   od_matrix:  parquet/CSV of (origin, destination, demand) — many-to-many
commodities:
  - name: poi_service
    assignment: { from: poi_nodes, demand: 1, tier: cabinet }
  - name: cabinet_backhaul               # induced tier-to-tier commodity
    assignment: { from: { facilities: cabinet, opened: true },
                  demand: served_units, tier: central_office }
  # - name: freight
  #   od_matrix: { file: od.parquet, origin: o, destination: d, demand: q }
  #   time: { release: t0, deadline: t1 }    # only with time expansion

# -------------------------------------------------------- transforms -----
# Declarative time expansion (replaces gen --time): applied to named layers.
# travel_time: edge attribute name or constant (layers per step).
time_expansion: null
# time_expansion:
#   layers: [streets]
#   steps: 36
#   travel_time: 1
#   wait: { cost: 0.01, capacity: unbounded }

# ------------------------------------------------------------- objective --
# Weighted sum; every term optional. `unserved` makes demand elastic:
# a commodity may be (partially) dropped at a price -> always-feasible
# models and prize-collecting behavior.
objective:
  flow_cost: 1.0            # sum over arcs: cost_per_unit * flow
  fixed_cost: 1.0           # edges/facilities opened (fixed charges)
  overage_penalty: 1.0      # soft-capacity violations
  unserved_penalty: { per_unit: 1.0e6 }   # effectively "must serve"
```

Run parameters stay out of the model (CLI or a tiny sibling `run:` file):
backend, threads, budget seconds, gap tolerance, seed.

## Capacity violation semantics (the important design point)

Soft capacity `cap` with `penalty_per_unit` compiles to a **piecewise-linear
convex arc cost**: flow up to `cap` at base cost, overage at
`base + penalty`. Two implementation routes, both cheap in our architecture:

1. **Arc duplication** (compiler-level, zero solver change): a parallel
   "overflow arc" with cost `base + penalty` and capacity `max - cap`
   (unbounded if no `max`). Works today for both solvers.
2. **Multiplier clamp** (solver-level, better): in the Lagrangian, a soft
   capacity is *exactly* a bound on its multiplier: `lambda_a <= penalty_a`.
   One `min()` in the projection step.

Soft capacities are not just a modeling nicety — they fix a measured pain:
real street graphs have low-degree cuts, hard caps make instances
**infeasible**, and we currently burn 30+ iterations detecting that. With
soft caps a feasible primal always exists, the dual is bounded, the primal
heuristic cannot fail, and "infeasibility" becomes a legible answer:
*serving this demand requires paying N in overage here* — visible in the
GeoParquet export as `overage > 0` edges. Recommended default for real-data
models: soft with a stiff penalty, hard caps only where physics demands.

## Feature matrix: v1 vs later

| Feature | v1 (compiler on today's solvers) | Solver work needed |
|---|---|---|
| Multi-layer graphs + couplings (drops, colocated joins) | yes | none (compiles to one flat graph with layer tags) |
| Edge capacity hard/soft, per-selector overrides | yes | none via overflow arcs; multiplier clamp later (small) |
| Node capacity (throughput) | yes | none (node-split transform) |
| Facility tier: open cost, serving capacity, count bounds | one tier (= design mode today) | multi-tier: medium (extends design matheuristic; cabinet_backhaul compiles to a second design pass in v1) |
| Commodities: explicit pairs / assignment / OD matrix | yes | none (pairs+OD = MCF; assignment = design) |
| Soft/elastic demand (`unserved_penalty`) | yes | none (virtual sink arc per commodity) |
| Declarative time expansion, release/deadline windows | yes | none (windows = restricted source/sink layers) |
| Existing infrastructure (brownfield: `fixed_cost: 0` on built edges) | yes | none |
| Duct/right-of-way sharing across tiers (`cable.reuse_factor`) | **implemented** for facility chains: later tiers pay `reuse_factor` × cost on edges already carrying cable and are routed toward them (full-SF: OLT trunk 94% on existing duct, −7.3% total) | exact simultaneous multi-layer sharing: medium |
| Unsplittable commodities | no | large (branching / rounding; roadmap) |
| Multi-period investment, stochastic scenarios | no | large (roadmap; schema reserves `periods:`/`scenarios:` keys) |
| Protection / diverse paths | no | medium-large (roadmap) |

## Compilation pipeline

```
YAML --(validate: schema, units, selector resolution)-->
  transforms (couplings -> drops/joins; node splits; overflow arcs;
              time expansion; virtual unserved sinks)
  --> canonical JSON model + flat instance (.graph/.txt) + provenance map
  --> report: entity counts, demand vs cut-capacity envelopes (fast-fail,
      R3-4), per-selector capacity coverage, warnings (long drops, isolated
      terminals, unreachable commodities)
```

The **provenance map** (compiled entity -> source layer/id/geometry) is what
`solution_to_geoparquet.py` becomes: exports stop being mode-specific and
just join solution values back through the map, so every model gets
geometry output (including overage and unserved as first-class columns).

## Implementation phases

1. **R3-7a** — schema + `arcedge-modelc` validating compiler covering: one
   street layer, demand layer, nearest-edge-split coupling, hard/soft edge
   capacities via overflow arcs, single facility tier, assignment + pairs
   commodities, unserved penalty, provenance map. Reproduces today's SF
   design and TE-MCF runs from two example configs (checked in E2E).
2. **R3-7b** — declarative time expansion + OD matrices + node capacities;
   retire `gen`'s bespoke flags (kept as thin wrappers).
3. **R3-7c** — multiplier-clamp soft capacities in the Lagrangian; overage
   reporting in exports; fast-fail envelope checks feeding R3-4.
4. **R3-7d** — second facility tier via chained design passes; duct-sharing
   approximation.
