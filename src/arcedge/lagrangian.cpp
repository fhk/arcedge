#include "lagrangian.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>

#include "dijkstra.hpp"
#include "graph.hpp"
#include "primal.hpp"

namespace arcedge {

namespace {

// Solves the shortest-path subproblem for every commodity, distributing
// commodities across threads. Fills per-commodity distances and path arc ids.
void batched_shortest_paths(const Instance& inst, const Graph& g,
                            const std::vector<double>& cost, int threads,
                            std::vector<double>& dists,
                            std::vector<std::vector<int32_t>>& paths) {
  const size_t nk = inst.commodities.size();
  std::atomic<size_t> next{0};
  auto worker = [&]() {
    SpBuffers buf;
    for (size_t k = next.fetch_add(1); k < nk; k = next.fetch_add(1)) {
      const Commodity& com = inst.commodities[k];
      dists[k] = shortest_path(g, cost, com.src, com.dst, buf);
      paths[k].clear();
      if (dists[k] < kInf) extract_path(buf, inst, com.src, com.dst, paths[k]);
    }
  };
  std::vector<std::thread> pool;
  const int nthreads = std::max(1, threads);
  pool.reserve(static_cast<size_t>(nthreads));
  for (int t = 0; t < nthreads; ++t) pool.emplace_back(worker);
  for (auto& th : pool) th.join();
}

}  // namespace

SolveResult solve(const Instance& inst, const SolveOptions& opt) {
  const auto t0 = std::chrono::steady_clock::now();
  const Graph g = Graph::build(inst);
  const size_t m = inst.arcs.size();
  const size_t nk = inst.commodities.size();

  std::vector<double> lambda(m, 0.0);  // stays 0 on uncapacitated arcs
  std::vector<double> reduced(m);
  std::vector<double> load(m);
  std::vector<double> grad(m);
  std::vector<double> dists(nk);
  std::vector<std::vector<int32_t>> paths(nk);

  SolveResult res;
  res.best_lb = -kInf;
  res.best_ub = primal_heuristic(inst, g, lambda);  // lambda = 0 warm start
  if (opt.verbose)
    std::printf("initial primal ub = %.4f\n", res.best_ub);

  double alpha = opt.alpha0;
  int no_improve = 0;

  for (int iter = 0; iter < opt.max_iters; ++iter) {
    res.iters = iter + 1;
    for (size_t a = 0; a < m; ++a) reduced[a] = inst.arcs[a].cost + lambda[a];
    batched_shortest_paths(inst, g, reduced, opt.threads, dists, paths);

    // L(lambda) = sum_k q_k * d_lambda(o_k, d_k) - sum_a lambda_a * u_a
    double lb = 0.0;
    for (size_t k = 0; k < nk; ++k) {
      if (dists[k] >= kInf) {
        std::fprintf(stderr, "commodity %zu has no path: instance infeasible\n", k);
        res.best_ub = kInf;
        return res;
      }
      lb += inst.commodities[k].demand * dists[k];
    }
    for (size_t a = 0; a < m; ++a)
      if (inst.arcs[a].capacitated()) lb -= lambda[a] * inst.arcs[a].cap;

    if (lb > res.best_lb + 1e-12) {
      res.best_lb = lb;
      no_improve = 0;
    } else if (++no_improve >= opt.stall_iters) {
      alpha *= 0.5;
      no_improve = 0;
    }

    std::fill(load.begin(), load.end(), 0.0);
    for (size_t k = 0; k < nk; ++k)
      for (int32_t a : paths[k])
        load[static_cast<size_t>(a)] += inst.commodities[k].demand;

    if (iter % opt.primal_every == 0)
      res.best_ub = std::min(res.best_ub, primal_heuristic(inst, g, lambda));

    res.gap = (res.best_ub > 0 && res.best_ub < kInf)
                  ? (res.best_ub - res.best_lb) / res.best_ub
                  : kInf;
    if (opt.verbose && (iter % 5 == 0 || res.gap <= opt.gap_tol))
      std::printf("iter %4d  lb %.4f  best_lb %.4f  best_ub %.4f  gap %.4f%%\n",
                  iter, lb, res.best_lb, res.best_ub, res.gap * 100.0);
    if (res.gap <= opt.gap_tol) break;

    // Projected subgradient on the dualized (capacitated) constraints only.
    // Components that would push an already-zero multiplier further negative
    // are projected out so they do not deflate the Polyak step.
    double norm2 = 0.0;
    for (size_t a = 0; a < m; ++a) {
      if (!inst.arcs[a].capacitated()) {
        grad[a] = 0.0;
        continue;
      }
      double ga = load[a] - inst.arcs[a].cap;
      if (lambda[a] <= 0.0 && ga < 0.0) ga = 0.0;
      grad[a] = ga;
      norm2 += ga * ga;
    }
    if (norm2 < 1e-18) {
      // The relaxed solution respects every capacity: it is primal optimal.
      double ub = 0.0;
      for (size_t k = 0; k < nk; ++k)
        for (int32_t a : paths[k])
          ub += inst.commodities[k].demand * inst.arcs[static_cast<size_t>(a)].cost;
      res.best_ub = std::min(res.best_ub, ub);
      res.gap = (res.best_ub - res.best_lb) / std::max(res.best_ub, 1e-12);
      break;
    }
    const double target =
        res.best_ub < kInf ? res.best_ub : std::abs(lb) * 1.1 + 10.0;
    const double step = alpha * std::max(target - lb, 1e-9) / norm2;
    for (size_t a = 0; a < m; ++a)
      lambda[a] = std::max(0.0, lambda[a] + step * grad[a]);
  }

  // Final primal refresh with the last multipliers.
  res.best_ub = std::min(res.best_ub, primal_heuristic(inst, g, lambda));
  res.gap = (res.best_ub > 0 && res.best_ub < kInf)
                ? (res.best_ub - res.best_lb) / res.best_ub
                : kInf;
  res.millis = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0)
                   .count();
  return res;
}

}  // namespace arcedge
