#include "primal.hpp"

#include <algorithm>
#include <numeric>

namespace arcedge {

namespace {

// Routes all commodities in the residual network with the given arc costs.
// Returns the true (unpenalized) cost of the routing, or kInf on failure.
double route_all(const Instance& inst, const Graph& g, std::vector<double> cost,
                 const std::vector<size_t>& order) {
  const size_t m = inst.arcs.size();
  std::vector<double> residual(m);
  for (size_t a = 0; a < m; ++a)
    residual[a] = inst.arcs[a].capacitated() ? inst.arcs[a].cap : kInf;

  SpBuffers buf;
  std::vector<int32_t> path;
  double total = 0.0;
  // Each augmentation saturates an arc or finishes a commodity, so this bound
  // is generous; it only guards against numerical stalls.
  const int max_augment = 512;
  for (size_t k : order) {
    const Commodity& com = inst.commodities[k];
    double rem = com.demand;
    int augment = 0;
    while (rem > 1e-9) {
      if (++augment > max_augment) return kInf;
      const double d = shortest_path(g, cost, com.src, com.dst, buf);
      if (d >= kInf) return kInf;
      path.clear();
      extract_path(buf, inst, com.src, com.dst, path);
      double push = rem;
      for (int32_t a : path) push = std::min(push, residual[static_cast<size_t>(a)]);
      for (int32_t a : path) {
        const Arc& arc = inst.arcs[static_cast<size_t>(a)];
        total += push * arc.cost;
        if (arc.capacitated()) {
          residual[static_cast<size_t>(a)] -= push;
          if (residual[static_cast<size_t>(a)] <= 1e-9)
            cost[static_cast<size_t>(a)] = kInf;  // drop saturated arc
        }
      }
      rem -= push;
    }
  }
  return total;
}

}  // namespace

double primal_heuristic(const Instance& inst, const Graph& g,
                        const std::vector<double>& lambda) {
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

  std::vector<double> guided(m), pure(m);
  for (size_t a = 0; a < m; ++a) {
    pure[a] = inst.arcs[a].cost;
    guided[a] = inst.arcs[a].cost + lambda[a];
  }
  const double ub_guided = route_all(inst, g, std::move(guided), order);
  const double ub_pure = route_all(inst, g, std::move(pure), order);
  return std::min(ub_guided, ub_pure);
}

}  // namespace arcedge
