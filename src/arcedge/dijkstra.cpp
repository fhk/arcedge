#include "dijkstra.hpp"

#include <queue>
#include <utility>

namespace arcedge {

double shortest_path(const Graph& g, const std::vector<double>& cost, int32_t src,
                     int32_t dst, SpBuffers& buf) {
  buf.dist.assign(static_cast<size_t>(g.num_nodes), kInf);
  buf.parent_arc.assign(static_cast<size_t>(g.num_nodes), -1);
  using Item = std::pair<double, int32_t>;  // (dist, node)
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
  buf.dist[static_cast<size_t>(src)] = 0.0;
  pq.emplace(0.0, src);
  while (!pq.empty()) {
    auto [d, u] = pq.top();
    pq.pop();
    if (d > buf.dist[static_cast<size_t>(u)]) continue;  // stale entry
    if (u == dst) return d;
    for (int32_t e = g.offset[static_cast<size_t>(u)];
         e < g.offset[static_cast<size_t>(u) + 1]; ++e) {
      const int32_t a = g.arc_id[static_cast<size_t>(e)];
      const double w = cost[static_cast<size_t>(a)];
      if (w >= kInf) continue;
      const int32_t v = g.head[static_cast<size_t>(e)];
      const double nd = d + w;
      if (nd < buf.dist[static_cast<size_t>(v)]) {
        buf.dist[static_cast<size_t>(v)] = nd;
        buf.parent_arc[static_cast<size_t>(v)] = a;
        pq.emplace(nd, v);
      }
    }
  }
  return buf.dist[static_cast<size_t>(dst)];
}

void extract_path(const SpBuffers& buf, const Instance& inst, int32_t src, int32_t dst,
                  std::vector<int32_t>& out) {
  int32_t v = dst;
  while (v != src) {
    const int32_t a = buf.parent_arc[static_cast<size_t>(v)];
    out.push_back(a);
    v = inst.arcs[static_cast<size_t>(a)].tail;
  }
}

}  // namespace arcedge
