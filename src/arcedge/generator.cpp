#include "generator.hpp"

#include <cstdlib>
#include <random>
#include <stdexcept>

namespace arcedge {

Instance generate(const GeneratorParams& p) {
  const int W = p.width, H = p.height, T = p.time_steps;
  if (W < 2 || H < 2 || T < 2) throw std::runtime_error("grid/time too small");
  const int n_static = W * H;
  std::mt19937 rng(p.seed);
  std::uniform_int_distribution<int> cost_dist(1, 10);
  std::uniform_int_distribution<int> demand_dist(1, 3);
  std::uniform_int_distribution<int> node_dist(0, n_static - 1);
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  // Static street arcs: 4-neighborhood, both directions, random integer costs.
  struct StreetArc {
    int32_t u, v;
    double cost;
  };
  std::vector<StreetArc> street;
  auto add_street = [&](int a, int b) {
    street.push_back({a, b, static_cast<double>(cost_dist(rng))});
    street.push_back({b, a, static_cast<double>(cost_dist(rng))});
  };
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const int v = y * W + x;
      if (x + 1 < W) add_street(v, v + 1);
      if (y + 1 < H) add_street(v, v + W);
    }

  Instance inst;
  inst.num_nodes = n_static * T;
  inst.arcs.reserve((street.size() + static_cast<size_t>(n_static)) *
                    static_cast<size_t>(T - 1));
  const double cap = p.capacity > 0 ? p.capacity : -1.0;
  for (int t = 0; t + 1 < T; ++t) {
    const int32_t base = t * n_static, next = (t + 1) * n_static;
    for (const StreetArc& s : street)
      inst.arcs.push_back({base + s.u, next + s.v, s.cost, cap});
    for (int v = 0; v < n_static; ++v)
      inst.arcs.push_back({base + v, next + v, p.wait_cost, -1.0});
  }

  // Hub destinations shared by many commodities create binding capacities.
  std::vector<int> hub_nodes;
  for (int h = 0; h < p.hubs; ++h) hub_nodes.push_back(node_dist(rng));
  auto manhattan = [&](int a, int b) {
    return std::abs(a % W - b % W) + std::abs(a / W - b / W);
  };
  for (int k = 0; k < p.commodities; ++k) {
    int dst = (!hub_nodes.empty() && unit(rng) < p.hub_frac)
                  ? hub_nodes[static_cast<size_t>(k) % hub_nodes.size()]
                  : node_dist(rng);
    int src = node_dist(rng);
    // Destination must be reachable within T-1 movement/wait steps.
    int guard = 0;
    while ((src == dst || manhattan(src, dst) > T - 1) && ++guard < 10000)
      src = node_dist(rng);
    if (guard >= 10000)
      throw std::runtime_error("cannot sample reachable commodity; increase time steps");
    inst.commodities.push_back(
        {src, (T - 1) * n_static + dst, static_cast<double>(demand_dist(rng))});
  }
  return inst;
}

}  // namespace arcedge
