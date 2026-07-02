// Assert-style unit tests for the Stage 1 combinatorial core. Each check has a
// known-correct expected value (hand-computed optimum or a structural
// identity), so a pass is a correctness proof for that scenario.
#include <cmath>
#include <cstdio>

#include "arcedge/dijkstra.hpp"
#include "arcedge/generator.hpp"
#include "arcedge/graph.hpp"
#include "arcedge/lagrangian.hpp"

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

int main() {
  test_dijkstra();
  test_diamond_exact();
  test_uncapacitated_zero_gap();
  test_generator_shape();
  if (failures == 0) {
    std::printf("all tests passed\n");
    return 0;
  }
  std::printf("%d check(s) failed\n", failures);
  return 1;
}
