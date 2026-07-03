#include "lagrangian.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <thread>

#include "dijkstra.hpp"
#include "graph.hpp"
#include "primal.hpp"
#ifdef ARCEDGE_CUDA
#include "cuda_sssp.hpp"
#endif

namespace arcedge {

namespace {

// Solves the shortest-path subproblem for every commodity, distributing
// commodities across threads. Fills per-commodity distances and path arc ids.
// `dag` selects the topological level-sweep kernel instead of Dijkstra.
void batched_shortest_paths(const Instance& inst, const Graph& g,
                            const DagLevels* dag,
                            const std::vector<double>& cost, int threads,
                            std::vector<double>& dists,
                            std::vector<std::vector<int32_t>>& paths) {
  const size_t nk = inst.commodities.size();
  std::atomic<size_t> next{0};
  auto worker = [&]() {
    SpBuffers buf;
    for (size_t k = next.fetch_add(1); k < nk; k = next.fetch_add(1)) {
      const Commodity& com = inst.commodities[k];
      dists[k] = dag ? dag_shortest_path(*dag, inst, cost, com.src, com.dst, buf)
                     : shortest_path(g, cost, com.src, com.dst, buf);
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

  std::string backend = opt.sp_backend == "auto" ? "dijkstra" : opt.sp_backend;
  DagLevels dag;
  if (backend == "dag" || backend == "cuda") {
    dag = DagLevels::build(inst);
    if (!dag.is_dag)
      throw std::runtime_error("--sp-backend " + backend +
                               " requires a DAG (time-expanded) instance");
  }
#ifdef ARCEDGE_CUDA
  std::unique_ptr<CudaBatchSssp> cuda_engine;
  if (backend == "cuda") {
    if (!CudaBatchSssp::available())
      throw std::runtime_error("no CUDA device available");
    cuda_engine.reset(new CudaBatchSssp(inst, dag));
  }
#else
  if (backend == "cuda")
    throw std::runtime_error("arcedge was built without -DARCEDGE_CUDA=ON");
#endif
  if (backend != "dijkstra" && backend != "dag" && backend != "cuda")
    throw std::runtime_error("unknown --sp-backend: " + backend);
  if (opt.verbose) std::printf("sp backend: %s\n", backend.c_str());

  std::vector<double> lambda(m, 0.0);  // stays 0 on uncapacitated arcs
  std::vector<double> lambda_at_best_lb;  // kept for FP64 re-certification
  std::vector<double> reduced(m);
  std::vector<double> load(m);
  std::vector<double> grad(m);
  std::vector<double> dists(nk);
  std::vector<std::vector<int32_t>> paths(nk);

  SolveResult res;
  res.best_lb = -kInf;
  res.best_ub = kInf;
  // Runs the primal heuristic and keeps the flow behind every UB improvement.
  // The pure-cost companion pass only runs while lambda is cold (first
  // attempt); warm-lambda guided routing wins in practice.
  bool primal_attempted = false;
  int primal_backoff = 1;   // doubles after fully-failed attempts
  int next_attempt_iter = 0;
  auto refresh_ub = [&](const std::vector<double>& lam, bool cold_lambda) {
    std::vector<double> flow;
    const double ub =
        primal_heuristic(inst, g, lam, opt.threads, cold_lambda, &flow);
    primal_attempted = true;
    if (ub < res.best_ub) {
      res.best_ub = ub;
      res.flow = std::move(flow);
    }
    // A failed attempt on a hard instance predicts more failures: back off
    // exponentially instead of paying for routing that cannot succeed yet.
    if (ub >= kInf && res.best_ub >= kInf) primal_backoff *= 2;
    else primal_backoff = 1;
  };

  double alpha = opt.alpha0;
  int no_improve = 0;
  double lb_free_flow = 0.0;  // LB at lambda = 0, set on the first iteration

  for (int iter = 0; iter < opt.max_iters; ++iter) {
    res.iters = iter + 1;
    for (size_t a = 0; a < m; ++a) reduced[a] = inst.arcs[a].cost + lambda[a];
    bool loads_from_engine = false;
#ifdef ARCEDGE_CUDA
    if (cuda_engine) {
      cuda_engine->solve(reduced, inst.commodities, dists, load);
      loads_from_engine = true;  // demand-weighted flow accumulated on device
    } else
#endif
      batched_shortest_paths(inst, g, backend == "dag" ? &dag : nullptr,
                             reduced, opt.threads, dists, paths);

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

    if (iter == 0) lb_free_flow = lb;
    if (lb > res.best_lb + 1e-12) {
      res.best_lb = lb;
      if (loads_from_engine) lambda_at_best_lb = lambda;  // FP32 run: recheck later
      no_improve = 0;
    } else if (++no_improve >= opt.stall_iters) {
      alpha *= 0.5;
      no_improve = 0;
    }

    // An unbounded Lagrangian dual certifies primal infeasibility. If no
    // feasible flow has been found and the bound has blown far past the
    // free-flow cost, stop and say so instead of stepping forever.
    if (opt.primal && res.best_ub >= kInf && iter >= 30 &&
        res.best_lb > 20.0 * std::abs(lb_free_flow) + 1.0) {
      std::fprintf(stderr,
                   "dual bound diverging (%.4g vs free-flow %.4g) with no feasible "
                   "primal: instance is likely infeasible -- raise capacities, "
                   "add time steps, or spread demand\n",
                   res.best_lb, lb_free_flow);
      break;
    }

    if (!loads_from_engine) {
      std::fill(load.begin(), load.end(), 0.0);
      for (size_t k = 0; k < nk; ++k)
        for (int32_t a : paths[k])
          load[static_cast<size_t>(a)] += inst.commodities[k].demand;
    }

    // Cold-lambda routing on a congested instance is measured waste (the
    // passes fail outright), so the first attempt waits until either the
    // congestion is mild or the multipliers have had a chance to warm up.
    if (opt.primal && iter % opt.primal_every == 0 && iter >= next_attempt_iter) {
      double overload = 0.0;
      for (size_t a = 0; a < m; ++a)
        if (inst.arcs[a].capacitated() && inst.arcs[a].cap > 0.0)
          overload = std::max(overload, load[a] / inst.arcs[a].cap);
      if (primal_attempted || iter > 0 || overload <= 3.0) {
        refresh_ub(lambda, iter == 0);
        next_attempt_iter = iter + opt.primal_every * primal_backoff;
      }
    }

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
      // load is the demand-weighted flow, so its true cost is the UB.
      double ub = 0.0;
      for (size_t a = 0; a < m; ++a) ub += load[a] * inst.arcs[a].cost;
      if (ub < res.best_ub) {
        res.best_ub = ub;
        res.flow = load;
      }
      res.gap = (res.best_ub - res.best_lb) / std::max(res.best_ub, 1e-12);
      break;
    }
    const double target =
        res.best_ub < kInf ? res.best_ub : std::abs(lb) * 1.1 + 10.0;
    const double step = alpha * std::max(target - lb, 1e-9) / norm2;
    for (size_t a = 0; a < m; ++a)
      lambda[a] = std::max(0.0, lambda[a] + step * grad[a]);
  }

#ifdef ARCEDGE_CUDA
  // The GPU computes distances in FP32, so the running best_lb is not a
  // certified bound (rounding could overestimate a distance sum). Re-evaluate
  // L(lambda) at the best multipliers in FP64 on the CPU: any lambda >= 0
  // gives a valid bound, so the certified value replaces the FP32 one.
  // Skipped in --no-primal benchmark mode: it costs one CPU-batch iteration,
  // which would distort pure SSSP throughput comparisons.
  if (opt.primal && cuda_engine && !lambda_at_best_lb.empty()) {
    for (size_t a = 0; a < m; ++a)
      reduced[a] = inst.arcs[a].cost + lambda_at_best_lb[a];
    batched_shortest_paths(inst, g, &dag, reduced, opt.threads, dists, paths);
    double lb64 = 0.0;
    for (size_t k = 0; k < nk; ++k)
      lb64 += inst.commodities[k].demand * dists[k];
    for (size_t a = 0; a < m; ++a)
      if (inst.arcs[a].capacitated()) lb64 -= lambda_at_best_lb[a] * inst.arcs[a].cap;
    if (opt.verbose)
      std::printf("fp64 certification: fp32 lb %.6f -> certified lb %.6f\n",
                  res.best_lb, lb64);
    res.best_lb = lb64;
  }
#endif

  // Final primal refresh with the last multipliers.
  if (opt.primal && res.gap > opt.gap_tol) refresh_ub(lambda, false);
  res.gap = (res.best_ub > 0 && res.best_ub < kInf)
                ? (res.best_ub - res.best_lb) / res.best_ub
                : kInf;
  res.millis = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0)
                   .count();
  return res;
}

}  // namespace arcedge
