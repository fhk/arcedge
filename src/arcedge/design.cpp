#include "design.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>

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

Net build_net(const StreetGraph& g, const std::vector<int32_t>& pois,
              bool mark_leaves) {
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
  if (mark_leaves)
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
  std::vector<DesignUsedEdge> used;  // edges with load > 0
  std::vector<int32_t> poi_hub;      // serving hub per POI index
  bool feasible = false;
};

std::vector<DesignUsedEdge> used_from_loads(const Net& net,
                                            const std::vector<double>& load) {
  std::vector<DesignUsedEdge> used;
  for (size_t e = 0; e < load.size(); ++e)
    if (load[e] > 0.0)
      used.push_back({net.eu[e], net.ev[e], load[e]});
  return used;
}

// Routes all POIs on the forest metric until capacity holds, opening relief
// hubs at the congested side of overloaded edges. Fills `load`; returns false
// if a POI is disconnected or the repair does not converge.
bool route_repair(const Net& net, const std::vector<int32_t>& pois,
                  const std::vector<double>& dem, std::vector<int32_t>& hubs,
                  const DesignParams& p, Forest& f,
                  const std::vector<double>* edge_cost, std::vector<double>& load) {
  std::set<int32_t> hub_set(hubs.begin(), hubs.end());
  std::vector<double> served(static_cast<size_t>(net.n), 0.0);
  // Batched relief: a fraction of the overloaded funnels get relief hubs
  // per round (worst first), so rounds scale roughly logarithmically in the
  // relief count instead of linearly (one-per-round was the measured
  // full-SF bottleneck: hundreds of forest rebuilds). Relieving only a
  // fraction per round lets the forest re-balance between additions, which
  // keeps the realized hub count lean.
  for (int repair = 0; repair < 200; ++repair) {
    grow_forest(net, hubs, f, edge_cost);
    std::fill(load.begin(), load.end(), 0.0);
    std::fill(served.begin(), served.end(), 0.0);
    for (size_t i = 0; i < pois.size(); ++i) {
      const int32_t poi = pois[i];
      if (f.hub_of[static_cast<size_t>(poi)] < 0) return false;  // disconnected
      served[static_cast<size_t>(f.hub_of[static_cast<size_t>(poi)])] += dem[i];
      for (int32_t v = poi; f.parent_edge[static_cast<size_t>(v)] >= 0;
           v = f.parent_node[static_cast<size_t>(v)])
        load[static_cast<size_t>(f.parent_edge[static_cast<size_t>(v)])] += dem[i];
    }
    std::vector<std::pair<double, int32_t>> overloaded;  // (load, edge)
    std::vector<char> is_over(load.size(), 0);
    if (p.edge_cap > 0.0)
      for (size_t e = 0; e < load.size(); ++e)
        if (load[e] > p.edge_cap) {
          overloaded.emplace_back(load[e], static_cast<int32_t>(e));
          is_over[e] = 1;
        }
    // Over-capacity HUBS also demand relief: shed their heaviest child
    // subtree by opening a hub at its head.
    std::vector<std::pair<double, int32_t>> hub_relief;  // (served, relief node)
    if (p.hub_cap > 0.0)
      for (int32_t h : hubs)
        if (served[static_cast<size_t>(h)] > p.hub_cap) {
          double best_ld = 0.0;
          int32_t relief = -1;
          for (int32_t i2 = net.adj_off[static_cast<size_t>(h)];
               i2 < net.adj_off[static_cast<size_t>(h) + 1]; ++i2) {
            const int32_t w = net.adj_nbr[static_cast<size_t>(i2)];
            const int32_t e = net.adj_edge[static_cast<size_t>(i2)];
            if (f.parent_edge[static_cast<size_t>(w)] == e &&
                load[static_cast<size_t>(e)] > best_ld &&
                !net.is_poi[static_cast<size_t>(w)]) {
              best_ld = load[static_cast<size_t>(e)];
              relief = w;
            }
          }
          if (relief >= 0)
            hub_relief.emplace_back(served[static_cast<size_t>(h)], relief);
        }
    if (overloaded.empty() && hub_relief.empty()) return true;
    std::sort(overloaded.rbegin(), overloaded.rend());
    std::sort(hub_relief.rbegin(), hub_relief.rend());
    // A quarter of the candidates per round tempers overshoot; the absolute
    // cap scales up for high-density tiers (thousands of small facilities)
    // where 64/round cannot converge within the round limit.
    const int n_cand =
        static_cast<int>(overloaded.size() + hub_relief.size());
    const int per_round = std::max(1, std::min(1024, n_cand / 4));
    int added = 0;
    for (const auto& [sv, relief] : hub_relief) {
      if (added >= per_round) break;
      if (hub_set.insert(relief).second) {
        hubs.push_back(relief);
        ++added;
      }
    }
    for (const auto& [ld, e] : overloaded) {
      if (added >= per_round) break;
      const int32_t u = net.eu[static_cast<size_t>(e)];
      const int32_t v = net.ev[static_cast<size_t>(e)];
      const bool u_far = f.parent_edge[static_cast<size_t>(u)] == e;
      const int32_t relief = u_far ? u : v;
      const int32_t near = u_far ? v : u;
      // Only relieve overload-MAXIMAL edges: if an ancestor edge on the way
      // to the hub is itself overloaded, relieving the ancestor absorbs this
      // subtree too -- opening both wastes a hub.
      bool has_over_ancestor = false;
      for (int32_t w = near; f.parent_edge[static_cast<size_t>(w)] >= 0;
           w = f.parent_node[static_cast<size_t>(w)])
        if (is_over[static_cast<size_t>(f.parent_edge[static_cast<size_t>(w)])]) {
          has_over_ancestor = true;
          break;
        }
      if (has_over_ancestor) continue;
      if (hub_set.insert(relief).second) {
        hubs.push_back(relief);
        ++added;
      }
    }
    if (added == 0) return false;  // overloaded but no new relief site
  }
  return false;
}

// Steiner consolidation via the sequential shortest-path heuristic, run per
// hub cluster. POIs attach one at a time: edges already in the cluster tree
// cost nothing (their fixed charge is sunk) unless saturated (blocked), new
// edges cost their length. Capacity is exact -- every edge on the attachment
// path, tree or new, gains one unit of load. Returns total cable meters, or
// -1 if some POI cannot be routed within its cluster's capacity.
// SPH for one cluster (all POIs served by `hub`, nearest-first). Clusters
// are edge-disjoint by construction (searches never leave the hub's region),
// so concurrent clusters may share `load`/`tree_edge` without locks -- they
// touch disjoint indices. Returns the cluster's cable meters or -1.
double sph_cluster(const Net& net, const Forest& f, const DesignParams& p,
                   int32_t hub,
                   const std::vector<std::pair<int32_t, double>>& cluster_pois,
                   std::vector<double>& load, std::vector<char>& tree_edge,
                   std::vector<double>& dist, std::vector<int32_t>& pe,
                   std::vector<int32_t>& pn, std::vector<uint32_t>& stamp,
                   uint32_t& version) {
  double cable = 0.0;
  using Item = std::pair<double, int32_t>;
  for (const auto& [poi, q] : cluster_pois) {
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
          if (p.edge_cap > 0.0 &&
              load[static_cast<size_t>(e)] + q > p.edge_cap + 1e-9)
            continue;  // full for this demand
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
      load[static_cast<size_t>(e)] += q;
      if (!tree_edge[static_cast<size_t>(e)]) {
        tree_edge[static_cast<size_t>(e)] = 1;
        cable += net.elen[static_cast<size_t>(e)];
      }
    }
  }
  return cable;
}

double sph_consolidate(const Net& net, const std::vector<int32_t>& pois,
                       const std::vector<double>& dem,
                       const std::vector<int32_t>& hubs, const Forest& f,
                       const DesignParams& p, std::vector<double>& load) {
  (void)hubs;
  const size_t m = net.elen.size();
  load.assign(m, 0.0);
  std::vector<char> tree_edge(m, 0);

  // Group POIs by serving hub, nearest-first within each cluster.
  struct Item3 { int32_t hub, poi; double q; };
  std::vector<Item3> by_hub;
  by_hub.reserve(pois.size());
  for (size_t i = 0; i < pois.size(); ++i)
    by_hub.push_back({f.hub_of[static_cast<size_t>(pois[i])], pois[i], dem[i]});
  std::sort(by_hub.begin(), by_hub.end(), [&](const Item3& a, const Item3& b) {
    if (a.hub != b.hub) return a.hub < b.hub;
    return f.dist[static_cast<size_t>(a.poi)] < f.dist[static_cast<size_t>(b.poi)];
  });
  std::vector<std::pair<int32_t, std::vector<std::pair<int32_t, double>>>> clusters;
  for (const Item3& it : by_hub) {
    if (clusters.empty() || clusters.back().first != it.hub)
      clusters.push_back({it.hub, {}});
    clusters.back().second.emplace_back(it.poi, it.q);
  }

  // Clusters run in parallel; per-worker scratch, lock-free shared state
  // (disjoint edge indices), deterministic per-cluster POI order.
  std::atomic<size_t> next{0};
  std::atomic<bool> failed{false};
  const int nthreads = std::max(
      1, std::min<int>(p.threads > 0 ? p.threads
                                     : static_cast<int>(std::thread::hardware_concurrency()),
                       static_cast<int>(clusters.size())));
  std::vector<double> cable_per_thread(static_cast<size_t>(nthreads), 0.0);
  auto worker = [&](int tid) {
    std::vector<double> dist(static_cast<size_t>(net.n));
    std::vector<int32_t> pe(static_cast<size_t>(net.n)), pn(static_cast<size_t>(net.n));
    std::vector<uint32_t> stamp(static_cast<size_t>(net.n), 0);
    uint32_t version = 0;
    for (size_t c = next.fetch_add(1); c < clusters.size() && !failed.load();
         c = next.fetch_add(1)) {
      const double cable =
          sph_cluster(net, f, p, clusters[c].first, clusters[c].second, load,
                      tree_edge, dist, pe, pn, stamp, version);
      if (cable < 0.0) failed.store(true);
      else cable_per_thread[static_cast<size_t>(tid)] += cable;
    }
  };
  std::vector<std::thread> pool;
  pool.reserve(static_cast<size_t>(nthreads));
  for (int t = 0; t < nthreads; ++t) pool.emplace_back(worker, t);
  for (auto& th : pool) th.join();
  if (failed.load()) return -1.0;
  double cable = 0.0;
  for (double c : cable_per_thread) cable += c;
  return cable;
}

// Evaluates a hub set: shortest-path forest with capacity repair, then
// (optionally) SPH consolidation, keeping the cheaper feasible design.
Eval evaluate(const Net& net, const std::vector<int32_t>& pois,
              const std::vector<double>& dem, std::vector<int32_t> hubs,
              const DesignParams& p, Forest& f, bool consolidate) {
  Eval ev;
  std::vector<double> load(net.elen.size());
  if (!route_repair(net, pois, dem, hubs, p, f, nullptr, load)) return ev;
  double cable = 0.0;
  for (size_t e = 0; e < load.size(); ++e)
    if (load[e] > 0.0) cable += net.elen[e];
  ev.cable_m = cable;
  ev.cost = p.hub_cost * static_cast<double>(hubs.size()) +
            p.cable_cost_per_m * cable;
  ev.feasible = true;
  ev.used = used_from_loads(net, load);
  ev.poi_hub.resize(pois.size());
  for (size_t i = 0; i < pois.size(); ++i)
    ev.poi_hub[i] = f.hub_of[static_cast<size_t>(pois[i])];
  if (consolidate) {
    std::vector<double> sph_load;
    const double sph_cable =
        sph_consolidate(net, pois, dem, hubs, f, p, sph_load);
    if (sph_cable >= 0.0 && sph_cable < ev.cable_m) {
      ev.cable_m = sph_cable;
      ev.cost = p.hub_cost * static_cast<double>(hubs.size()) +
                p.cable_cost_per_m * sph_cable;
      ev.used = used_from_loads(net, sph_load);
    }
  }
  ev.hubs = std::move(hubs);
  return ev;
}

// Seeds k hubs by k-means over POI coordinates, mapping each centroid to the
// nearest non-POI node. For large k the O(k * n) k-means/mapping cost is not
// worth it (repair, recentering and pruning dominate placement anyway), so
// hubs seed O(k) at the street-side neighbors of random distinct POIs.
std::vector<int32_t> seed_hubs(const Net& net, const std::vector<int32_t>& pois,
                               int k, std::mt19937& rng) {
  if (k > 512) {
    std::set<int32_t> hubset;
    std::uniform_int_distribution<size_t> pdist(0, pois.size() - 1);
    for (int guard = 0; static_cast<int>(hubset.size()) < k && guard < 8 * k;
         ++guard) {
      const int32_t poi = pois[pdist(rng)];
      // A POI's first neighbor is its drop foot (a street node); if demand
      // sits directly on street nodes, the node itself serves.
      int32_t site = poi;
      if (net.is_poi[static_cast<size_t>(poi)]) {
        const int32_t deg_off = net.adj_off[static_cast<size_t>(poi)];
        if (deg_off < net.adj_off[static_cast<size_t>(poi) + 1])
          site = net.adj_nbr[static_cast<size_t>(deg_off)];
        else
          continue;
      }
      if (!net.is_poi[static_cast<size_t>(site)]) hubset.insert(site);
    }
    return std::vector<int32_t>(hubset.begin(), hubset.end());
  }
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
                    const DesignParams& p, const std::vector<double>& demands) {
  if (pois.empty()) throw std::runtime_error("no POIs to serve");
  const auto t0 = std::chrono::steady_clock::now();
  const Net net = build_net(g, pois, !p.demand_transit);
  std::vector<double> dem = demands;
  if (dem.empty()) dem.assign(pois.size(), 1.0);
  if (dem.size() != pois.size())
    throw std::runtime_error("demands size mismatch");
  if (p.hub_cap > 0.0)
    for (double q : dem)
      if (q > p.hub_cap)
        throw std::runtime_error("a single demand exceeds hub_cap");

  const int n_pois = static_cast<int>(pois.size());
  double total_demand = 0.0;
  for (double q : dem) total_demand += q;
  // The k-means seeding is O(k * pois), so the COARSE sweep is capped -- but
  // never below what the hub capacity provably requires. Refinement phases
  // may probe any k up to the POI count (realized hub counts routinely
  // exceed the sweep cap once capacity repair kicks in).
  int max_k = p.max_k > 0 ? p.max_k : std::min(1024, std::max(1, n_pois / 8));
  if (p.hub_cap > 0.0)
    max_k = std::max(max_k,
                     std::min(1024, static_cast<int>(
                                        2.0 * total_demand / p.hub_cap) + 1));
  const int nthreads = p.threads > 0
                           ? p.threads
                           : std::max(1u, std::thread::hardware_concurrency());

  std::mutex best_mu;
  Eval best;
  // Candidate k's are independent; each gets its own deterministic RNG
  // (seeded by k) so results do not depend on thread scheduling.
  auto try_k = [&](int k, bool consolidate) {
    if (k < 1 || k > n_pois) return;
    std::mt19937 rng(p.seed * 2654435761u + static_cast<unsigned>(k));
    Forest f;
    std::vector<int32_t> hubs = seed_hubs(net, pois, k, rng);
    Eval ev = evaluate(net, pois, dem, hubs, p, f, consolidate);
    // Recentering is O(hubs * n); at high hub density placement is already
    // near-nodal and repair/prune dominate, so skip it there.
    for (int round = 0; round < 2 && ev.feasible && ev.hubs.size() <= 1024;
         ++round) {
      std::vector<int32_t> moved = recenter(net, pois, ev.hubs, f);
      Eval ev2 = evaluate(net, pois, dem, std::move(moved), p, f, consolidate);
      if (ev2.feasible && ev2.cost < ev.cost) ev = std::move(ev2);
      else break;
    }
    std::lock_guard<std::mutex> lock(best_mu);
    if (p.verbose)
      std::printf("k %4d%s -> hubs %4zu  cable %.0f m  cost %.0f%s\n", k,
                  consolidate ? " (sph)" : "      ",
                  ev.feasible ? ev.hubs.size() : 0, ev.cable_m, ev.cost,
                  ev.feasible ? "" : "  (infeasible)");
    if (ev.feasible && ev.cost < best.cost) best = std::move(ev);
  };
  auto run_batch = [&](std::vector<int> ks, bool consolidate) {
    std::sort(ks.begin(), ks.end());
    ks.erase(std::unique(ks.begin(), ks.end()), ks.end());
    std::atomic<size_t> next{0};
    auto worker = [&]() {
      for (size_t i = next.fetch_add(1); i < ks.size(); i = next.fetch_add(1))
        try_k(ks[i], consolidate);
    };
    // SPH consolidation is internally parallel, so consolidating batches run
    // one k at a time; the cheap forest sweep parallelizes across k's.
    const int batch_threads =
        consolidate ? 1 : std::min<int>(nthreads, static_cast<int>(ks.size()));
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(batch_threads));
    for (int t = 0; t < batch_threads; ++t) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
  };

  // Phase A: coarse geometric sweep with the cheap forest evaluation.
  {
    std::vector<int> ks;
    for (int k = 1; k <= max_k; k = std::max(k + 1, k * 2)) ks.push_back(k);
    run_batch(std::move(ks), false);
  }
  // Phase B: cheap descent on k -- relief inflates realized hub counts at
  // coarse k, so walk downhill on the cheap objective before spending the
  // (expensive) SPH evaluations anywhere.
  for (int round = 0; round < 4 && best.feasible; ++round) {
    const double prev = best.cost;
    const int kb = static_cast<int>(best.hubs.size());
    run_batch({kb, kb - kb / 3, kb - kb / 5, kb - kb / 10, kb - 1, kb + 1,
               kb + kb / 10},
              false);
    if (best.cost >= prev - 1e-9) break;
  }
  // Phase B2: greedy hub prune. Batched relief overshoots the hub count a
  // little; closing the least-loaded hubs (letting repair re-add any that
  // were actually needed) walks the count back down at cheap-eval cost.
  if (best.feasible) {
    Forest f;
    int batch = 8;
    for (int it = 0; it < 30 && batch >= 1 && best.hubs.size() > 1; ++it) {
      std::vector<double> served(best.hubs.size(), 0.0);
      std::vector<int32_t> idx_of(static_cast<size_t>(net.n), -1);
      for (size_t h = 0; h < best.hubs.size(); ++h)
        idx_of[static_cast<size_t>(best.hubs[h])] = static_cast<int32_t>(h);
      for (size_t i = 0; i < best.poi_hub.size(); ++i) {
        const int32_t hub = best.poi_hub[i];
        if (hub >= 0 && idx_of[static_cast<size_t>(hub)] >= 0)
          served[static_cast<size_t>(idx_of[static_cast<size_t>(hub)])] += dem[i];
      }
      std::vector<size_t> by_load(best.hubs.size());
      std::iota(by_load.begin(), by_load.end(), size_t{0});
      std::sort(by_load.begin(), by_load.end(),
                [&](size_t a, size_t b) { return served[a] < served[b]; });
      const size_t drop =
          std::min<size_t>(static_cast<size_t>(batch), best.hubs.size() - 1);
      std::set<size_t> dropped(by_load.begin(), by_load.begin() + drop);
      std::vector<int32_t> trial;
      for (size_t h = 0; h < best.hubs.size(); ++h)
        if (!dropped.count(h)) trial.push_back(best.hubs[h]);
      Eval ev = evaluate(net, pois, dem, std::move(trial), p, f, false);
      if (ev.feasible && ev.cost < best.cost) {
        if (p.verbose)
          std::printf("prune -%d -> hubs %4zu  cost %.0f\n", batch,
                      ev.hubs.size(), ev.cost);
        best = std::move(ev);
      } else {
        batch /= 2;
      }
    }
  }
  // Phase C: SPH consolidation on the winning neighborhood.
  if (best.feasible) {
    const int kb = static_cast<int>(best.hubs.size());
    best = Eval();
    run_batch({kb, kb - kb / 8, kb - 1, kb + 1, kb + kb / 8}, true);
    const int kb2 = best.feasible ? static_cast<int>(best.hubs.size()) : kb;
    if (kb2 != kb) run_batch({kb2 - 2, kb2 + 2}, true);
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
  res.used_edges = std::move(best.used);
  res.poi_hub = std::move(best.poi_hub);
  res.millis = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0)
                   .count();
  return res;
}

}  // namespace arcedge
