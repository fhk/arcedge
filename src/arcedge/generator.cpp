#include "generator.hpp"

#include <algorithm>
#include <cstdlib>
#include <queue>
#include <random>
#include <stdexcept>

namespace arcedge {

namespace {

// BFS hop distances from src over the street graph's directed arcs. The
// graph is bidirectional, so distances from a node equal distances to it.
std::vector<int32_t> bfs_hops(const std::vector<std::vector<int32_t>>& adj,
                              int32_t src) {
  std::vector<int32_t> hops(adj.size(), -1);
  std::queue<int32_t> q;
  hops[static_cast<size_t>(src)] = 0;
  q.push(src);
  while (!q.empty()) {
    const int32_t u = q.front();
    q.pop();
    for (int32_t v : adj[static_cast<size_t>(u)])
      if (hops[static_cast<size_t>(v)] < 0) {
        hops[static_cast<size_t>(v)] = hops[static_cast<size_t>(u)] + 1;
        q.push(v);
      }
  }
  return hops;
}

}  // namespace

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

Instance generate_from_street(const StreetGraph& street, const GeneratorParams& p) {
  const int T = p.time_steps;
  const int32_t n_static = street.num_nodes;
  if (n_static < 2 || street.arcs.empty() || T < 2)
    throw std::runtime_error("street graph or time horizon too small");
  std::mt19937 rng(p.seed);
  std::uniform_int_distribution<int> demand_dist(1, 3);
  std::uniform_int_distribution<int32_t> node_dist(0, n_static - 1);
  std::uniform_real_distribution<double> unit(0.0, 1.0);

  Instance inst;
  inst.num_nodes = n_static * T;
  inst.arcs.reserve((street.arcs.size() + static_cast<size_t>(n_static)) *
                    static_cast<size_t>(T - 1));
  const double cap = p.capacity > 0 ? p.capacity : -1.0;
  for (int t = 0; t + 1 < T; ++t) {
    const int32_t base = t * n_static, next = (t + 1) * n_static;
    for (const StreetGraph::SArc& s : street.arcs)
      inst.arcs.push_back({base + s.tail, next + s.head, s.cost, cap});
    for (int32_t v = 0; v < n_static; ++v)
      inst.arcs.push_back({base + v, next + v, p.wait_cost, -1.0});
  }

  std::vector<std::vector<int32_t>> adj(static_cast<size_t>(n_static));
  for (const StreetGraph::SArc& s : street.arcs)
    adj[static_cast<size_t>(s.tail)].push_back(s.head);

  // Sources for a destination are sampled from the nodes BFS says can reach
  // it within T-1 movement steps (waiting absorbs the remaining layers).
  auto sample_sources = [&](int32_t dst) {
    std::vector<int32_t> candidates;
    const std::vector<int32_t> hops = bfs_hops(adj, dst);
    for (int32_t v = 0; v < n_static; ++v)
      if (v != dst && hops[static_cast<size_t>(v)] >= 1 &&
          hops[static_cast<size_t>(v)] <= T - 1)
        candidates.push_back(v);
    return candidates;
  };

  // Hubs stuck in tiny pockets (few reachable sources) are rejected; the
  // threshold adapts down for very small graphs.
  const size_t min_hub_sources =
      std::min<size_t>(8, static_cast<size_t>(n_static) / 2);
  std::vector<int32_t> hub_nodes;
  std::vector<std::vector<int32_t>> hub_sources;
  for (int h = 0; h < p.hubs; ++h) {
    int guard = 0;
    while (++guard < 1000) {
      const int32_t hub = node_dist(rng);
      std::vector<int32_t> cands = sample_sources(hub);
      if (cands.size() >= min_hub_sources) {
        hub_nodes.push_back(hub);
        hub_sources.push_back(std::move(cands));
        break;
      }
    }
  }
  if (hub_nodes.empty() && p.hubs > 0)
    throw std::runtime_error("no viable hub found; increase time steps");

  for (int k = 0; k < p.commodities; ++k) {
    int32_t dst;
    std::vector<int32_t>* cands;
    std::vector<int32_t> local;
    if (!hub_nodes.empty() && unit(rng) < p.hub_frac) {
      const size_t h = static_cast<size_t>(k) % hub_nodes.size();
      dst = hub_nodes[h];
      cands = &hub_sources[h];
    } else {
      int guard = 0;
      do {
        dst = node_dist(rng);
        local = sample_sources(dst);
      } while (local.size() < 2 && ++guard < 1000);
      if (local.size() < 2)
        throw std::runtime_error("cannot sample reachable commodity; increase time steps");
      cands = &local;
    }
    std::uniform_int_distribution<size_t> pick(0, cands->size() - 1);
    const int32_t src = (*cands)[pick(rng)];
    inst.commodities.push_back(
        {src, (T - 1) * n_static + dst, static_cast<double>(demand_dist(rng))});
  }
  return inst;
}

}  // namespace arcedge
