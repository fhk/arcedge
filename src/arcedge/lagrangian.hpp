#pragma once

#include <string>
#include <vector>

#include "instance.hpp"

namespace arcedge {

struct SolveOptions {
  int max_iters = 400;
  double gap_tol = 0.005;   // stop when (ub - lb) / ub <= gap_tol
  int threads = 4;
  int primal_every = 10;    // refresh the primal upper bound every N iterations
  double alpha0 = 1.5;      // initial Polyak step scale, halved on stalls
  int stall_iters = 15;     // halve alpha after this many non-improving iters
  bool verbose = true;
  // Shortest-path subproblem backend: "auto" (dijkstra), "dijkstra"
  // (per-commodity binary-heap, CPU threads), "dag" (topological level sweep,
  // CPU threads -- the GPU kernel's structure, exact on time-expanded DAGs),
  // or "cuda" (GPU batched level sweep; needs -DARCEDGE_CUDA=ON and a device).
  std::string sp_backend = "auto";
};

struct SolveResult {
  double best_lb = 0.0;
  double best_ub = 0.0;
  double gap = 0.0;  // (best_ub - best_lb) / best_ub
  int iters = 0;
  double millis = 0.0;
};

// Lagrangian relaxation of capacitated MCF: capacity constraints are dualized,
// the relaxed problem decomposes into one shortest-path problem per commodity
// (solved in parallel), and the dual is maximized by projected subgradient
// ascent with Polyak step sizes. A residual-routing primal heuristic guided by
// the multipliers supplies the upper bound.
SolveResult solve(const Instance& inst, const SolveOptions& opt);

}  // namespace arcedge
