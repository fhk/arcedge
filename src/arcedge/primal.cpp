#include "primal.hpp"

#include <algorithm>
#include <atomic>
#include <numeric>
#include <thread>

namespace arcedge {

namespace {

// Routes all commodities in the residual network with the given arc costs.
// Wave structure: parallel shortest paths on a frozen cost snapshot, then a
// sequential commit pass in `order`. Returns the true (unpenalized) cost of
// the routing, or kInf on failure. `flow` (if non-null) receives per-arc flow.
double route_all(const Instance& inst, const Graph& g, std::vector<double> cost,
                 const std::vector<size_t>& order, int threads,
                 std::vector<double>* flow) {
  const size_t m = inst.arcs.size();
  const size_t nk = inst.commodities.size();
  std::vector<double> residual(m);
  for (size_t a = 0; a < m; ++a)
    residual[a] = inst.arcs[a].capacitated() ? inst.arcs[a].cap : kInf;
  if (flow) flow->assign(m, 0.0);

  std::vector<double> rem(nk);
  for (size_t k = 0; k < nk; ++k) rem[k] = inst.commodities[k].demand;
  std::vector<size_t> unrouted = order;
  std::vector<std::vector<int32_t>> path(nk);
  double total = 0.0;

  auto commit = [&](size_t k) {
    // Pushes as much of commodity k's remaining demand as its precomputed
    // path allows against CURRENT residuals. Returns the amount pushed.
    double push = rem[k];
    for (int32_t a : path[k])
      if (inst.arcs[static_cast<size_t>(a)].capacitated())
        push = std::min(push, residual[static_cast<size_t>(a)]);
    if (push <= 1e-9) return 0.0;
    for (int32_t a : path[k]) {
      const Arc& arc = inst.arcs[static_cast<size_t>(a)];
      total += push * arc.cost;
      if (flow) (*flow)[static_cast<size_t>(a)] += push;
      if (arc.capacitated()) {
        residual[static_cast<size_t>(a)] -= push;
        if (residual[static_cast<size_t>(a)] <= 1e-9)
          cost[static_cast<size_t>(a)] = kInf;  // saturated: drop the arc
      }
    }
    rem[k] -= push;
    return push;
  };

  const int max_waves = 40;
  for (int wave = 0; wave < max_waves && !unrouted.empty(); ++wave) {
    // Parallel SSSP for every unrouted commodity on the frozen snapshot.
    std::atomic<size_t> next{0};
    std::atomic<bool> unreachable{false};
    auto worker = [&]() {
      SpBuffers buf;
      for (size_t i = next.fetch_add(1); i < unrouted.size();
           i = next.fetch_add(1)) {
        const size_t k = unrouted[i];
        const Commodity& com = inst.commodities[k];
        path[k].clear();
        if (shortest_path(g, cost, com.src, com.dst, buf) >= kInf) {
          unreachable.store(true);
          continue;
        }
        extract_path(buf, inst, com.src, com.dst, path[k]);
      }
    };
    const int nthreads =
        std::max(1, std::min<int>(threads, static_cast<int>(unrouted.size())));
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(nthreads));
    for (int t = 0; t < nthreads; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
    // Residuals only shrink, so a commodity unreachable now stays that way.
    if (unreachable.load()) return kInf;

    // Sequential, order-fixed commit pass.
    bool progress = false;
    std::vector<size_t> still;
    for (size_t k : unrouted) {
      if (commit(k) > 0.0) progress = true;
      if (rem[k] > 1e-9) still.push_back(k);
    }
    unrouted = std::move(still);
    if (unrouted.empty()) return total;

    if (!progress) {
      // Every precomputed path saturated before commit: finish the tail with
      // exact sequential successive shortest paths (the pre-wave algorithm).
      SpBuffers buf;
      for (size_t k : unrouted) {
        const Commodity& com = inst.commodities[k];
        int augment = 0;
        while (rem[k] > 1e-9) {
          if (++augment > 512) return kInf;
          if (shortest_path(g, cost, com.src, com.dst, buf) >= kInf) return kInf;
          path[k].clear();
          extract_path(buf, inst, com.src, com.dst, path[k]);
          if (commit(k) <= 0.0) return kInf;  // cannot happen on a fresh path
        }
      }
      return total;
    }
  }
  return unrouted.empty() ? total : kInf;
}

}  // namespace

double primal_heuristic(const Instance& inst, const Graph& g,
                        const std::vector<double>& lambda, int threads,
                        bool include_pure, std::vector<double>* flow_out) {
  const size_t m = inst.arcs.size();
  const size_t nk = inst.commodities.size();

  // Largest demands first: they are the hardest to fit once arcs saturate.
  std::vector<size_t> order(nk);
  std::iota(order.begin(), order.end(), size_t{0});
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    if (inst.commodities[a].demand != inst.commodities[b].demand)
      return inst.commodities[a].demand > inst.commodities[b].demand;
    return a < b;
  });

  // Guided routing with escalating congestion pricing: if routing fails at
  // beta = 1 (heavily congested instances), amplified multipliers push early
  // commodities off the contested corridors so later ones still fit. Only
  // failures pay for the escalation.
  std::vector<double> flow_guided;
  double ub_guided = kInf;
  for (double beta : {1.0, 3.0, 10.0}) {
    std::vector<double> guided(m);
    for (size_t a = 0; a < m; ++a)
      guided[a] = inst.arcs[a].cost + beta * lambda[a];
    ub_guided = route_all(inst, g, std::move(guided), order, threads,
                          flow_out ? &flow_guided : nullptr);
    if (ub_guided < kInf) break;
  }
  double ub_pure = kInf;
  std::vector<double> flow_pure;
  if (include_pure) {
    std::vector<double> pure(m);
    for (size_t a = 0; a < m; ++a) pure[a] = inst.arcs[a].cost;
    ub_pure = route_all(inst, g, std::move(pure), order, threads,
                        flow_out ? &flow_pure : nullptr);
  }
  if (flow_out) {
    if (ub_guided <= ub_pure && ub_guided < kInf) *flow_out = std::move(flow_guided);
    else if (ub_pure < kInf) *flow_out = std::move(flow_pure);
    else flow_out->clear();
  }
  return std::min(ub_guided, ub_pure);
}

}  // namespace arcedge
