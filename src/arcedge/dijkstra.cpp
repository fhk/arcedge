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

DagLevels DagLevels::build(const Instance& inst) {
  DagLevels dag;
  const size_t n = static_cast<size_t>(inst.num_nodes);
  const size_t m = inst.arcs.size();
  std::vector<int32_t> indeg(n, 0);
  for (const Arc& a : inst.arcs) indeg[static_cast<size_t>(a.head)]++;

  // Kahn's algorithm with level propagation: level[v] = longest arc-count
  // distance from any source, so every arc strictly increases the level.
  std::vector<int32_t> out_off(n + 1, 0);
  for (const Arc& a : inst.arcs) out_off[static_cast<size_t>(a.tail) + 1]++;
  for (size_t v = 0; v < n; ++v) out_off[v + 1] += out_off[v];
  std::vector<int32_t> out_arc(m);
  {
    std::vector<int32_t> cur(out_off.begin(), out_off.end() - 1);
    for (size_t a = 0; a < m; ++a)
      out_arc[static_cast<size_t>(cur[static_cast<size_t>(inst.arcs[a].tail)]++)] =
          static_cast<int32_t>(a);
  }
  dag.level_of.assign(n, 0);
  std::vector<int32_t> queue;
  queue.reserve(n);
  for (size_t v = 0; v < n; ++v)
    if (indeg[v] == 0) queue.push_back(static_cast<int32_t>(v));
  size_t processed = 0;
  for (size_t qi = 0; qi < queue.size(); ++qi) {
    const int32_t u = queue[qi];
    ++processed;
    for (int32_t i = out_off[static_cast<size_t>(u)];
         i < out_off[static_cast<size_t>(u) + 1]; ++i) {
      const Arc& a = inst.arcs[static_cast<size_t>(out_arc[static_cast<size_t>(i)])];
      const size_t h = static_cast<size_t>(a.head);
      dag.level_of[h] = std::max(dag.level_of[h],
                                 dag.level_of[static_cast<size_t>(u)] + 1);
      if (--indeg[h] == 0) queue.push_back(a.head);
    }
  }
  if (processed != n) return dag;  // cycle: is_dag stays false
  dag.is_dag = true;
  for (size_t v = 0; v < n; ++v)
    dag.num_levels = std::max(dag.num_levels, dag.level_of[v] + 1);

  dag.level_off.assign(static_cast<size_t>(dag.num_levels) + 1, 0);
  for (const Arc& a : inst.arcs)
    dag.level_off[static_cast<size_t>(dag.level_of[static_cast<size_t>(a.tail)]) + 1]++;
  for (int32_t l = 0; l < dag.num_levels; ++l)
    dag.level_off[static_cast<size_t>(l) + 1] += dag.level_off[static_cast<size_t>(l)];
  dag.arcs_by_level.resize(m);
  std::vector<int32_t> cur(dag.level_off.begin(), dag.level_off.end() - 1);
  for (size_t a = 0; a < m; ++a) {
    const int32_t l = dag.level_of[static_cast<size_t>(inst.arcs[a].tail)];
    dag.arcs_by_level[static_cast<size_t>(cur[static_cast<size_t>(l)]++)] =
        static_cast<int32_t>(a);
  }
  return dag;
}

double dag_shortest_path(const DagLevels& dag, const Instance& inst,
                         const std::vector<double>& cost, int32_t src, int32_t dst,
                         SpBuffers& buf) {
  buf.dist.assign(static_cast<size_t>(inst.num_nodes), kInf);
  buf.parent_arc.assign(static_cast<size_t>(inst.num_nodes), -1);
  buf.dist[static_cast<size_t>(src)] = 0.0;
  const int32_t l_src = dag.level_of[static_cast<size_t>(src)];
  const int32_t l_dst = dag.level_of[static_cast<size_t>(dst)];
  // Levels strictly increase along arcs, so only buckets in [l_src, l_dst)
  // can lie on a src -> dst path.
  for (int32_t l = l_src; l < l_dst; ++l) {
    for (int32_t i = dag.level_off[static_cast<size_t>(l)];
         i < dag.level_off[static_cast<size_t>(l) + 1]; ++i) {
      const int32_t a = dag.arcs_by_level[static_cast<size_t>(i)];
      const double dt = buf.dist[static_cast<size_t>(inst.arcs[static_cast<size_t>(a)].tail)];
      if (dt >= kInf) continue;
      const double w = cost[static_cast<size_t>(a)];
      if (w >= kInf) continue;
      const size_t h = static_cast<size_t>(inst.arcs[static_cast<size_t>(a)].head);
      if (dt + w < buf.dist[h]) {
        buf.dist[h] = dt + w;
        buf.parent_arc[h] = a;
      }
    }
  }
  return buf.dist[static_cast<size_t>(dst)];
}

}  // namespace arcedge
