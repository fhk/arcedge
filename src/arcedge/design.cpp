#include "design.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>

namespace arcedge {

namespace {

constexpr double kInfD = std::numeric_limits<double>::infinity();

// Undirected view of the street graph plus POI markers.
struct Net {
  int32_t n = 0;
  std::vector<int32_t> eu, ev;   // undirected edge endpoints
  std::vector<double> elen;      // undirected edge lengths (m)
  std::vector<int32_t> adj_off;  // CSR over nodes
  std::vector<int32_t> adj_nbr;
  std::vector<int32_t> adj_edge;
  std::vector<char> is_poi;
  std::vector<double> x, y;  // projected coords (m)
};

Net build_net(const StreetGraph& g, const std::vector<int32_t>& pois) {
  Net net;
  net.n = g.num_nodes;
  std::set<std::pair<int32_t, int32_t>> seen;
  for (const StreetGraph::SArc& a : g.arcs) {
    const auto key = std::minmax(a.tail, a.head);
    if (seen.insert(key).second) {
      net.eu.push_back(key.first);
      net.ev.push_back(key.second);
      net.elen.push_back(a.cost);
    }
  }
  const size_t m = net.eu.size();
  net.adj_off.assign(static_cast<size_t>(net.n) + 1, 0);
  for (size_t e = 0; e < m; ++e) {
    net.adj_off[static_cast<size_t>(net.eu[e]) + 1]++;
    net.adj_off[static_cast<size_t>(net.ev[e]) + 1]++;
  }
  for (size_t v = 0; v < static_cast<size_t>(net.n); ++v)
    net.adj_off[v + 1] += net.adj_off[v];
  net.adj_nbr.resize(2 * m);
  net.adj_edge.resize(2 * m);
  std::vector<int32_t> cur(net.adj_off.begin(), net.adj_off.end() - 1);
  for (size_t e = 0; e < m; ++e) {
    net.adj_nbr[static_cast<size_t>(cur[static_cast<size_t>(net.eu[e])])] = net.ev[e];
    net.adj_edge[static_cast<size_t>(cur[static_cast<size_t>(net.eu[e])]++)] =
        static_cast<int32_t>(e);
    net.adj_nbr[static_cast<size_t>(cur[static_cast<size_t>(net.ev[e])])] = net.eu[e];
    net.adj_edge[static_cast<size_t>(cur[static_cast<size_t>(net.ev[e])]++)] =
        static_cast<int32_t>(e);
  }
  net.is_poi.assign(static_cast<size_t>(net.n), 0);
  for (int32_t p : pois) net.is_poi[static_cast<size_t>(p)] = 1;
  // Equirectangular projection, consistent with scripts/connect_pois.py.
  const double kx = 111320.0 * std::cos(37.77 * M_PI / 180.0), ky = 110540.0;
  net.x.resize(static_cast<size_t>(net.n));
  net.y.resize(static_cast<size_t>(net.n));
  for (int32_t v = 0; v < net.n; ++v) {
    net.x[static_cast<size_t>(v)] = g.lon[static_cast<size_t>(v)] * kx;
    net.y[static_cast<size_t>(v)] = g.lat[static_cast<size_t>(v)] * ky;
  }
  return net;
}

struct Forest {
  std::vector<double> dist;
  std::vector<int32_t> parent_edge, parent_node, hub_of;
};

// Multi-source Dijkstra from the hub set; POI nodes are never used as transit
// (they are drop leaves, flow must not cut through a customer premises).
// `edge_cost` overrides the metric (used for consolidation rounds where
// already-paid edges are discounted); pass nullptr for true lengths.
void grow_forest(const Net& net, const std::vector<int32_t>& hubs, Forest& f,
                 const std::vector<double>* edge_cost = nullptr) {
  f.dist.assign(static_cast<size_t>(net.n), kInfD);
  f.parent_edge.assign(static_cast<size_t>(net.n), -1);
  f.parent_node.assign(static_cast<size_t>(net.n), -1);
  f.hub_of.assign(static_cast<size_t>(net.n), -1);
  using Item = std::pair<double, int32_t>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
  for (int32_t h : hubs) {
    f.dist[static_cast<size_t>(h)] = 0.0;
    f.hub_of[static_cast<size_t>(h)] = h;
    pq.emplace(0.0, h);
  }
  while (!pq.empty()) {
    auto [d, u] = pq.top();
    pq.pop();
    if (d > f.dist[static_cast<size_t>(u)]) continue;
    if (net.is_poi[static_cast<size_t>(u)]) continue;  // leaf: no transit
    for (int32_t i = net.adj_off[static_cast<size_t>(u)];
         i < net.adj_off[static_cast<size_t>(u) + 1]; ++i) {
      const int32_t v = net.adj_nbr[static_cast<size_t>(i)];
      const int32_t e = net.adj_edge[static_cast<size_t>(i)];
      const double w = edge_cost ? (*edge_cost)[static_cast<size_t>(e)]
                                 : net.elen[static_cast<size_t>(e)];
      const double nd = d + w;
      if (nd < f.dist[static_cast<size_t>(v)]) {
        f.dist[static_cast<size_t>(v)] = nd;
        f.parent_edge[static_cast<size_t>(v)] = e;
        f.parent_node[static_cast<size_t>(v)] = u;
        f.hub_of[static_cast<size_t>(v)] = f.hub_of[static_cast<size_t>(u)];
        pq.emplace(nd, v);
      }
    }
  }
}

struct Eval {
  double cost = kInfD;
  double cable_m = 0.0;
  std::vector<int32_t> hubs;
  bool feasible = false;
};

// Routes all POIs on the forest metric until capacity holds, opening relief
// hubs at the congested side of overloaded edges. Fills `load`; returns false
// if a POI is disconnected or the repair does not converge.
bool route_repair(const Net& net, const std::vector<int32_t>& pois,
                  std::vector<int32_t>& hubs, const DesignParams& p, Forest& f,
                  const std::vector<double>* edge_cost, std::vector<double>& load) {
  for (int repair = 0; repair < 800; ++repair) {
    grow_forest(net, hubs, f, edge_cost);
    std::fill(load.begin(), load.end(), 0.0);
    for (int32_t poi : pois) {
      if (f.hub_of[static_cast<size_t>(poi)] < 0) return false;  // disconnected
      for (int32_t v = poi; f.parent_edge[static_cast<size_t>(v)] >= 0;
           v = f.parent_node[static_cast<size_t>(v)])
        load[static_cast<size_t>(f.parent_edge[static_cast<size_t>(v)])] += 1.0;
    }
    int32_t worst_edge = -1;
    double worst_load = p.edge_cap;
    for (size_t e = 0; e < load.size(); ++e)
      if (load[e] > worst_load) {
        worst_load = load[e];
        worst_edge = static_cast<int32_t>(e);
      }
    if (worst_edge < 0) return true;
    // The endpoint whose parent edge is the overloaded one is on the far
    // side of the funnel: opening a hub there absorbs the whole subtree.
    const int32_t u = net.eu[static_cast<size_t>(worst_edge)];
    const int32_t v = net.ev[static_cast<size_t>(worst_edge)];
    const int32_t relief =
        (f.parent_edge[static_cast<size_t>(u)] == worst_edge) ? u : v;
    hubs.push_back(relief);
  }
  return false;
}

// Steiner consolidation via the sequential shortest-path heuristic, run per
// hub cluster. POIs attach one at a time: edges already in the cluster tree
// cost nothing (their fixed charge is sunk) unless saturated (blocked), new
// edges cost their length. Capacity is exact -- every edge on the attachment
// path, tree or new, gains one unit of load. Returns total cable meters, or
// -1 if some POI cannot be routed within its cluster's capacity.
double sph_consolidate(const Net& net, const std::vector<int32_t>& pois,
                       const std::vector<int32_t>& hubs, const Forest& f,
                       const DesignParams& p) {
  const size_t m = net.elen.size();
  std::vector<double> load(m, 0.0);
  std::vector<char> tree_edge(m, 0);
  std::vector<double> dist(static_cast<size_t>(net.n));
  std::vector<int32_t> pe(static_cast<size_t>(net.n)), pn(static_cast<size_t>(net.n));
  std::vector<uint32_t> stamp(static_cast<size_t>(net.n), 0);
  uint32_t version = 0;

  // Group POIs by serving hub, nearest-first within each cluster.
  std::vector<std::pair<int32_t, int32_t>> by_hub;  // (hub, poi)
  by_hub.reserve(pois.size());
  for (int32_t poi : pois)
    by_hub.emplace_back(f.hub_of[static_cast<size_t>(poi)], poi);
  std::sort(by_hub.begin(), by_hub.end(), [&](const auto& a, const auto& b) {
    if (a.first != b.first) return a.first < b.first;
    return f.dist[static_cast<size_t>(a.second)] < f.dist[static_cast<size_t>(b.second)];
  });

  double cable = 0.0;
  using Item = std::pair<double, int32_t>;
  for (const auto& [hub, poi] : by_hub) {
    ++version;
    std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
    dist[static_cast<size_t>(poi)] = 0.0;
    pe[static_cast<size_t>(poi)] = -1;
    stamp[static_cast<size_t>(poi)] = version;
    pq.emplace(0.0, poi);
    bool reached = false;
    while (!pq.empty()) {
      auto [d, u] = pq.top();
      pq.pop();
      if (stamp[static_cast<size_t>(u)] == version && d > dist[static_cast<size_t>(u)])
        continue;
      if (u == hub) { reached = true; break; }
      if (u != poi && net.is_poi[static_cast<size_t>(u)]) continue;  // leaf
      for (int32_t i = net.adj_off[static_cast<size_t>(u)];
           i < net.adj_off[static_cast<size_t>(u) + 1]; ++i) {
        const int32_t v = net.adj_nbr[static_cast<size_t>(i)];
        const int32_t e = net.adj_edge[static_cast<size_t>(i)];
        // Stay inside this hub's cluster so per-POI searches remain local.
        if (v != hub && f.hub_of[static_cast<size_t>(v)] != hub) continue;
        double w;
        if (tree_edge[static_cast<size_t>(e)]) {
          if (load[static_cast<size_t>(e)] >= p.edge_cap - 0.5) continue;  // full
          w = 0.0;
        } else {
          w = net.elen[static_cast<size_t>(e)];
        }
        const double nd = d + w;
        if (stamp[static_cast<size_t>(v)] != version ||
            nd < dist[static_cast<size_t>(v)]) {
          stamp[static_cast<size_t>(v)] = version;
          dist[static_cast<size_t>(v)] = nd;
          pe[static_cast<size_t>(v)] = e;
          pn[static_cast<size_t>(v)] = u;
          pq.emplace(nd, v);
        }
      }
    }
    if (!reached) return -1.0;
    for (int32_t v = hub; pe[static_cast<size_t>(v)] >= 0; v = pn[static_cast<size_t>(v)]) {
      const int32_t e = pe[static_cast<size_t>(v)];
      load[static_cast<size_t>(e)] += 1.0;
      if (!tree_edge[static_cast<size_t>(e)]) {
        tree_edge[static_cast<size_t>(e)] = 1;
        cable += net.elen[static_cast<size_t>(e)];
      }
    }
  }
  return cable;
}

// Evaluates a hub set: shortest-path forest with capacity repair, then
// (optionally) SPH consolidation, keeping the cheaper feasible design.
Eval evaluate(const Net& net, const std::vector<int32_t>& pois,
              std::vector<int32_t> hubs, const DesignParams& p, Forest& f,
              bool consolidate) {
  Eval ev;
  std::vector<double> load(net.elen.size());
  if (!route_repair(net, pois, hubs, p, f, nullptr, load)) return ev;
  double cable = 0.0;
  for (size_t e = 0; e < load.size(); ++e)
    if (load[e] > 0.0) cable += net.elen[e];
  ev.cable_m = cable;
  ev.cost = p.hub_cost * static_cast<double>(hubs.size()) +
            p.cable_cost_per_m * cable;
  ev.feasible = true;
  if (consolidate) {
    const double sph_cable = sph_consolidate(net, pois, hubs, f, p);
    if (sph_cable >= 0.0 && sph_cable < ev.cable_m) {
      ev.cable_m = sph_cable;
      ev.cost = p.hub_cost * static_cast<double>(hubs.size()) +
                p.cable_cost_per_m * sph_cable;
    }
  }
  ev.hubs = std::move(hubs);
  return ev;
}

// Seeds k hubs by k-means over POI coordinates, mapping each centroid to the
// nearest non-POI node.
std::vector<int32_t> seed_hubs(const Net& net, const std::vector<int32_t>& pois,
                               int k, std::mt19937& rng) {
  std::vector<size_t> pick(static_cast<size_t>(k));
  std::uniform_int_distribution<size_t> pdist(0, pois.size() - 1);
  for (auto& s : pick) s = pdist(rng);
  std::vector<double> cx(static_cast<size_t>(k)), cy(static_cast<size_t>(k));
  for (int c = 0; c < k; ++c) {
    cx[static_cast<size_t>(c)] = net.x[static_cast<size_t>(pois[pick[static_cast<size_t>(c)]])];
    cy[static_cast<size_t>(c)] = net.y[static_cast<size_t>(pois[pick[static_cast<size_t>(c)]])];
  }
  std::vector<int32_t> assign(pois.size());
  for (int it = 0; it < 8; ++it) {
    std::vector<double> sx(static_cast<size_t>(k), 0.0), sy(static_cast<size_t>(k), 0.0);
    std::vector<int32_t> cnt(static_cast<size_t>(k), 0);
    for (size_t i = 0; i < pois.size(); ++i) {
      const double px = net.x[static_cast<size_t>(pois[i])];
      const double py = net.y[static_cast<size_t>(pois[i])];
      int best = 0;
      double bd = kInfD;
      for (int c = 0; c < k; ++c) {
        const double dx = px - cx[static_cast<size_t>(c)], dy = py - cy[static_cast<size_t>(c)];
        const double d = dx * dx + dy * dy;
        if (d < bd) { bd = d; best = c; }
      }
      assign[i] = best;
      sx[static_cast<size_t>(best)] += px;
      sy[static_cast<size_t>(best)] += py;
      cnt[static_cast<size_t>(best)]++;
    }
    for (int c = 0; c < k; ++c)
      if (cnt[static_cast<size_t>(c)] > 0) {
        cx[static_cast<size_t>(c)] = sx[static_cast<size_t>(c)] / cnt[static_cast<size_t>(c)];
        cy[static_cast<size_t>(c)] = sy[static_cast<size_t>(c)] / cnt[static_cast<size_t>(c)];
      }
  }
  // Nearest non-POI node per centroid (deduplicated).
  std::set<int32_t> hubset;
  for (int c = 0; c < k; ++c) {
    int32_t best = -1;
    double bd = kInfD;
    for (int32_t v = 0; v < net.n; ++v) {
      if (net.is_poi[static_cast<size_t>(v)]) continue;
      const double dx = net.x[static_cast<size_t>(v)] - cx[static_cast<size_t>(c)];
      const double dy = net.y[static_cast<size_t>(v)] - cy[static_cast<size_t>(c)];
      const double d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = v; }
    }
    if (best >= 0) hubset.insert(best);
  }
  return std::vector<int32_t>(hubset.begin(), hubset.end());
}

// One Lloyd round on the network: re-center every hub at the non-POI node
// closest to the coordinate centroid of the POIs it currently serves.
std::vector<int32_t> recenter(const Net& net, const std::vector<int32_t>& pois,
                              const std::vector<int32_t>& hubs, const Forest& f) {
  std::vector<double> sx, sy;
  std::vector<int32_t> cnt;
  std::vector<int32_t> idx(static_cast<size_t>(net.n), -1);
  for (size_t h = 0; h < hubs.size(); ++h) idx[static_cast<size_t>(hubs[h])] =
      static_cast<int32_t>(h);
  sx.assign(hubs.size(), 0.0);
  sy.assign(hubs.size(), 0.0);
  cnt.assign(hubs.size(), 0);
  for (int32_t poi : pois) {
    const int32_t h = f.hub_of[static_cast<size_t>(poi)];
    if (h < 0 || idx[static_cast<size_t>(h)] < 0) continue;
    const size_t c = static_cast<size_t>(idx[static_cast<size_t>(h)]);
    sx[c] += net.x[static_cast<size_t>(poi)];
    sy[c] += net.y[static_cast<size_t>(poi)];
    cnt[c]++;
  }
  std::set<int32_t> out;
  for (size_t c = 0; c < hubs.size(); ++c) {
    if (cnt[c] == 0) continue;  // hub serves nobody: drop it
    const double gx = sx[c] / cnt[c], gy = sy[c] / cnt[c];
    int32_t best = hubs[c];
    double bd = kInfD;
    for (int32_t v = 0; v < net.n; ++v) {
      if (net.is_poi[static_cast<size_t>(v)]) continue;
      const double dx = net.x[static_cast<size_t>(v)] - gx;
      const double dy = net.y[static_cast<size_t>(v)] - gy;
      const double d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = v; }
    }
    out.insert(best);
  }
  return std::vector<int32_t>(out.begin(), out.end());
}

}  // namespace

DesignResult design(const StreetGraph& g, const std::vector<int32_t>& pois,
                    const DesignParams& p) {
  if (pois.empty()) throw std::runtime_error("no POIs to serve");
  const Net net = build_net(g, pois);
  std::mt19937 rng(p.seed);
  Forest f;

  const int n_pois = static_cast<int>(pois.size());
  // The k-means seeding is O(k * pois), so the sweep is capped; the capacity
  // repair can still push the realized hub count above max_k when needed.
  const int max_k = p.max_k > 0 ? p.max_k : std::min(1024, std::max(1, n_pois / 8));

  Eval best;
  auto try_k = [&](int k, bool consolidate) {
    if (k < 1 || k > max_k) return;
    std::vector<int32_t> hubs = seed_hubs(net, pois, k, rng);
    Eval ev = evaluate(net, pois, hubs, p, f, consolidate);
    for (int round = 0; round < 2 && ev.feasible; ++round) {
      std::vector<int32_t> moved = recenter(net, pois, ev.hubs, f);
      Eval ev2 = evaluate(net, pois, std::move(moved), p, f, consolidate);
      if (ev2.feasible && ev2.cost < ev.cost) ev = std::move(ev2);
      else break;
    }
    if (p.verbose)
      std::printf("k %4d%s -> hubs %4zu  cable %.0f m  cost %.0f%s\n", k,
                  consolidate ? " (sph)" : "      ",
                  ev.feasible ? ev.hubs.size() : 0, ev.cable_m, ev.cost,
                  ev.feasible ? "" : "  (infeasible)");
    if (ev.feasible && ev.cost < best.cost) best = std::move(ev);
  };

  // Coarse geometric sweep with the cheap forest evaluation, then refine
  // around the best k with SPH consolidation (which only lowers cable cost,
  // so the cheap sweep is a sound way to locate the k neighborhood).
  for (int k = 1; k <= max_k; k = std::max(k + 1, k * 2)) try_k(k, false);
  if (best.feasible) {
    const int kb = static_cast<int>(best.hubs.size());
    // Re-evaluate the winning region with consolidation enabled.
    best = Eval();
    for (int k : {kb, kb - kb / 4, kb - kb / 8, kb - 1, kb + 1, kb + kb / 8, kb + kb / 4})
      try_k(k, true);
    const int kb2 = best.feasible ? static_cast<int>(best.hubs.size()) : kb;
    for (int k : {kb2 - 2, kb2 + 2, kb2 - kb2 / 16, kb2 + kb2 / 16})
      if (k != kb2) try_k(k, true);
  }

  DesignResult res;
  if (!best.feasible) return res;
  res.feasible = true;
  res.hubs = static_cast<int>(best.hubs.size());
  res.cable_m = best.cable_m;
  res.hub_cost = p.hub_cost * res.hubs;
  res.cable_cost = p.cable_cost_per_m * best.cable_m;
  res.total_cost = best.cost;
  res.hub_nodes = best.hubs;
  return res;
}

}  // namespace arcedge
