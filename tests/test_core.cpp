// Assert-style unit tests for the Stage 1 combinatorial core. Each check has a
// known-correct expected value (hand-computed optimum or a structural
// identity), so a pass is a correctness proof for that scenario.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

#include "arcedge/design.hpp"
#include "arcedge/dijkstra.hpp"
#include "arcedge/generator.hpp"
#include "arcedge/graph.hpp"
#include "arcedge/lagrangian.hpp"
#include "arcedge/street_graph.hpp"

static int failures = 0;
#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
      ++failures;                                                          \
    }                                                                      \
  } while (0)

using namespace arcedge;

// Dijkstra picks the cheap two-hop path over the expensive direct arc.
static void test_dijkstra() {
  Instance inst;
  inst.num_nodes = 3;
  inst.arcs = {{0, 1, 1.0, -1}, {1, 2, 1.0, -1}, {0, 2, 5.0, -1}};
  Graph g = Graph::build(inst);
  std::vector<double> cost = {1.0, 1.0, 5.0};
  SpBuffers buf;
  CHECK(std::abs(shortest_path(g, cost, 0, 2, buf) - 2.0) < 1e-12);
  std::vector<int32_t> path;
  extract_path(buf, inst, 0, 2, path);
  CHECK(path.size() == 2);
  // Removing the middle arc forces the direct one.
  cost[1] = kInf;
  CHECK(std::abs(shortest_path(g, cost, 0, 2, buf) - 5.0) < 1e-12);
}

// Diamond with a binding capacity. One commodity, demand 2, from 0 to 3:
// cheap path 0->1->3 costs 2/unit but arcs cap at 1 unit; expensive path
// 0->2->3 costs 4/unit uncapacitated. LP optimum splits 1+1: cost 6.
static void test_diamond_exact() {
  Instance inst;
  inst.num_nodes = 4;
  inst.arcs = {{0, 1, 1.0, 1.0}, {1, 3, 1.0, 1.0}, {0, 2, 2.0, -1}, {2, 3, 2.0, -1}};
  inst.commodities = {{0, 3, 2.0}};
  SolveOptions opt;
  opt.max_iters = 300;
  opt.gap_tol = 1e-3;
  opt.threads = 2;
  opt.verbose = false;
  SolveResult res = solve(inst, opt);
  CHECK(res.best_ub <= 6.0 + 1e-9);   // heuristic finds the exact optimum
  CHECK(res.best_lb <= 6.0 + 1e-9);   // lower bound stays valid
  CHECK(res.best_lb >= 6.0 - 0.05);   // dual converges to the LP value
}

// With no capacities the relaxation is exact at lambda = 0: LB = UB = the
// demand-weighted sum of shortest paths, reached on the first iteration.
static void test_uncapacitated_zero_gap() {
  GeneratorParams p;
  p.width = 8;
  p.height = 8;
  p.time_steps = 6;
  p.commodities = 5;
  p.capacity = 0.0;  // uncapacitated
  p.seed = 7;
  Instance inst = generate(p);
  SolveOptions opt;
  opt.max_iters = 5;
  opt.gap_tol = 1e-9;
  opt.threads = 2;
  opt.verbose = false;
  SolveResult res = solve(inst, opt);
  CHECK(res.gap < 1e-9);
  CHECK(res.iters == 1);
}

// Generator invariants: arc counts, layer structure, commodity reachability.
static void test_generator_shape() {
  GeneratorParams p;
  p.width = 5;
  p.height = 4;
  p.time_steps = 7;
  p.commodities = 9;
  p.seed = 3;
  Instance inst = generate(p);
  const int n_static = p.width * p.height;
  CHECK(inst.num_nodes == n_static * p.time_steps);
  // Directed 4-neighbor arcs: 2 * ((W-1)*H horizontal + W*(H-1) vertical).
  const size_t street = 2u * (4u * 4u + 5u * 3u);
  CHECK(inst.arcs.size() ==
        (street + static_cast<size_t>(n_static)) * static_cast<size_t>(p.time_steps - 1));
  for (const Arc& a : inst.arcs) {
    CHECK(a.head / n_static == a.tail / n_static + 1);  // arcs advance one layer
    CHECK(a.cost >= 0.0);
  }
  for (const Commodity& k : inst.commodities) {
    CHECK(k.src / n_static == 0);
    CHECK(k.dst / n_static == p.time_steps - 1);
    CHECK(k.demand >= 1.0);
  }
}

// Street-graph import + time expansion: a 4-node path graph a-b-c-d expands
// into T layers with per-layer movement arcs and uncapacitated waiting arcs;
// commodities are reachable within T-1 hops and solve to a certified gap.
static void test_street_graph_expansion() {
  const char* path = "test_street.graph";
  {
    std::ofstream f(path);
    f << "c tiny path graph, lengths in meters\n";
    f << "g 4 6\n";
    f << "v 0 -122.40 37.70\nv 1 -122.41 37.71\nv 2 -122.42 37.72\nv 3 -122.43 37.73\n";
    f << "e 0 1 100\ne 1 0 100\ne 1 2 50\ne 2 1 50\ne 2 3 200\ne 3 2 200\n";
  }
  StreetGraph sg = StreetGraph::load(path);
  CHECK(sg.num_nodes == 4);
  CHECK(sg.arcs.size() == 6);
  GeneratorParams p;
  p.time_steps = 5;
  p.commodities = 4;
  p.capacity = 2.0;
  p.hubs = 1;
  p.seed = 5;
  Instance inst = generate_from_street(sg, p);
  CHECK(inst.num_nodes == 4 * 5);
  CHECK(inst.arcs.size() == (6u + 4u) * 4u);  // (street + wait) per transition
  for (const Arc& a : inst.arcs) CHECK(a.head / 4 == a.tail / 4 + 1);
  for (const Commodity& k : inst.commodities) {
    CHECK(k.src / 4 == 0);
    CHECK(k.dst / 4 == p.time_steps - 1);
  }
  SolveOptions opt;
  opt.max_iters = 200;
  opt.threads = 2;
  opt.verbose = false;
  SolveResult res = solve(inst, opt);
  CHECK(res.best_ub < kInf);
  CHECK(res.best_lb <= res.best_ub + 1e-9);
  std::remove(path);
}

// Hand-checkable hub design. Streets 0-1-2 (100 m each); POIs 3 and 4 drop
// to nodes 0 and 2 with 10 m drops. Any single hub uses all 220 m of cable:
// total = 20000 + 10 * 220 = 22200. Two hubs would cost 40000 + 200: worse.
static void test_design_single_hub() {
  StreetGraph g;
  g.num_nodes = 5;
  g.lon = {0, 0.001, 0.002, 0, 0.002};
  g.lat = {37.77, 37.77, 37.77, 37.7701, 37.7701};
  auto both = [&](int32_t u, int32_t v, double w) {
    g.arcs.push_back({u, v, w});
    g.arcs.push_back({v, u, w});
  };
  both(0, 1, 100);
  both(1, 2, 100);
  both(3, 0, 10);
  both(4, 2, 10);
  DesignParams p;
  p.edge_cap = 500;
  p.hub_cost = 20000;
  p.cable_cost_per_m = 10;
  p.verbose = false;
  DesignResult res = design(g, {3, 4}, p);
  CHECK(res.feasible);
  CHECK(res.hubs == 1);
  CHECK(std::abs(res.total_cost - 22200.0) < 1e-6);
}

// Capacity repair: both POIs drop to node 0, edge capacity 1. A hub at node 1
// would overload edge 0-1 (load 2), so the relief pass must end with a hub at
// node 0: cable = the two 10 m drops, total = 20000 + 10 * 20 = 20200.
static void test_design_capacity_repair() {
  StreetGraph g;
  g.num_nodes = 4;
  g.lon = {0, 0.001, 0, 0};
  g.lat = {37.77, 37.77, 37.7701, 37.7702};
  auto both = [&](int32_t u, int32_t v, double w) {
    g.arcs.push_back({u, v, w});
    g.arcs.push_back({v, u, w});
  };
  both(0, 1, 100);
  both(2, 0, 10);
  both(3, 0, 10);
  DesignParams p;
  p.edge_cap = 1;
  p.hub_cost = 20000;
  p.cable_cost_per_m = 10;
  p.verbose = false;
  DesignResult res = design(g, {2, 3}, p);
  CHECK(res.feasible);
  CHECK(res.hubs == 1);
  CHECK(std::abs(res.total_cost - 20200.0) < 1e-6);
}

// The DAG level-sweep kernel (the GPU port's exact structure) must agree with
// Dijkstra on every commodity of a time-expanded instance, for the base costs
// and for perturbed (reduced-cost-like) metrics.
static void test_dag_backend_equivalence() {
  GeneratorParams p;
  p.width = 12;
  p.height = 12;
  p.time_steps = 9;
  p.commodities = 25;
  p.capacity = 3;
  p.seed = 31;
  Instance inst = generate(p);
  DagLevels dag = DagLevels::build(inst);
  CHECK(dag.is_dag);
  CHECK(dag.num_levels == p.time_steps);  // layered by construction
  Graph g = Graph::build(inst);
  SpBuffers b1, b2;
  for (int trial = 0; trial < 3; ++trial) {
    std::vector<double> cost(inst.arcs.size());
    for (size_t a = 0; a < cost.size(); ++a)
      cost[a] = inst.arcs[a].cost + trial * 0.37 * static_cast<double>(a % 7);
    for (const Commodity& k : inst.commodities) {
      const double d1 = shortest_path(g, cost, k.src, k.dst, b1);
      const double d2 = dag_shortest_path(dag, inst, cost, k.src, k.dst, b2);
      CHECK(std::abs(d1 - d2) < 1e-9);
    }
  }
  // Full solves through both backends land on the same bounds.
  SolveOptions opt;
  opt.max_iters = 60;
  opt.threads = 2;
  opt.verbose = false;
  SolveResult r1 = solve(inst, opt);
  opt.sp_backend = "dag";
  SolveResult r2 = solve(inst, opt);
  CHECK(std::abs(r1.best_lb - r2.best_lb) < 1e-6 * std::max(1.0, r1.best_lb));
  CHECK(std::abs(r1.best_ub - r2.best_ub) < 1e-6 * std::max(1.0, r1.best_ub));
}

// Hub serving capacity. Streets 0-1-2 (100 m each); POIs 3,4 drop to node 0
// and POIs 5,6 drop to node 2 (10 m drops). With hub_cap = 2 one hub cannot
// serve all four units, so the design must open exactly two hubs; the
// optimum places them at nodes 0 and 2: 2 * 20000 + 10 * 40 m = 40400.
static void test_design_hub_capacity() {
  StreetGraph g;
  g.num_nodes = 7;
  g.lon = {0, 0.001, 0.002, 0, 0, 0.002, 0.002};
  g.lat = {37.77, 37.77, 37.77, 37.7701, 37.7702, 37.7701, 37.7702};
  auto both = [&](int32_t u, int32_t v, double w) {
    g.arcs.push_back({u, v, w});
    g.arcs.push_back({v, u, w});
  };
  both(0, 1, 100);
  both(1, 2, 100);
  both(3, 0, 10);
  both(4, 0, 10);
  both(5, 2, 10);
  both(6, 2, 10);
  DesignParams p;
  p.edge_cap = 0;  // uncapacitated edges: the hub cap must drive the split
  p.hub_cap = 2;
  p.hub_cost = 20000;
  p.cable_cost_per_m = 10;
  p.verbose = false;
  DesignResult res = design(g, {3, 4, 5, 6}, p);
  CHECK(res.feasible);
  CHECK(res.hubs == 2);
  CHECK(res.total_cost <= 40400.0 + 1e-6);
  // Every hub's demand-weighted service must respect the cap.
  std::vector<double> served_check(7, 0.0);
  for (size_t i = 0; i < res.poi_hub.size(); ++i)
    served_check[static_cast<size_t>(res.poi_hub[i])] += 1.0;
  for (double s : served_check) CHECK(s <= 2.0 + 1e-9);
}

// T-shape / hub-slide: a Y junction J with POI arms to A and B, plus a
// dead-end stub J-S whose node S sits exactly at the demand centroid -- so
// both k-means seeding and recentering place the hub at S, leaving a 30 m
// stub whose only purpose is to reach the hub. The slide pass must move the
// hub onto the junction splice at J and drop the stub:
// cable = 100 + 100 + 10 + 10 = 220, total = 20000 + 2200 = 22200 (not 22500).
static void test_design_hub_slide() {
  StreetGraph g;
  g.num_nodes = 6;  // 0=J 1=A 2=B 3=S 4=poiA 5=poiB
  g.lon = {-122.400, -122.401, -122.399, -122.400, -122.4011, -122.3989};
  g.lat = {37.7704, 37.7700, 37.7700, 37.7700, 37.7700, 37.7700};
  auto both = [&](int32_t u, int32_t v, double w) {
    g.arcs.push_back({u, v, w});
    g.arcs.push_back({v, u, w});
  };
  both(0, 1, 100);  // J-A
  both(0, 2, 100);  // J-B
  both(0, 3, 30);   // J-S (dead-end stub at the centroid)
  both(4, 1, 10);   // drop at A
  both(5, 2, 10);   // drop at B
  DesignParams p;
  p.edge_cap = 0;
  p.hub_cost = 20000;
  p.cable_cost_per_m = 10;
  p.verbose = false;
  DesignResult res = design(g, {4, 5}, p);
  CHECK(res.feasible);
  CHECK(res.hubs == 1);
  CHECK(std::abs(res.total_cost - 22200.0) < 1e-6);
  CHECK(res.hub_nodes[0] == 0);  // slid onto the junction splice
}

int main() {
  test_dijkstra();
  test_diamond_exact();
  test_uncapacitated_zero_gap();
  test_generator_shape();
  test_street_graph_expansion();
  test_design_single_hub();
  test_design_capacity_repair();
  test_design_hub_capacity();
  test_design_hub_slide();
  test_dag_backend_equivalence();
  if (failures == 0) {
    std::printf("all tests passed\n");
    return 0;
  }
  std::printf("%d check(s) failed\n", failures);
  return 1;
}
