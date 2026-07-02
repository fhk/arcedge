#include "graph.hpp"

namespace arcedge {

Graph Graph::build(const Instance& inst) {
  Graph g;
  g.num_nodes = inst.num_nodes;
  const size_t m = inst.arcs.size();
  g.offset.assign(static_cast<size_t>(g.num_nodes) + 1, 0);
  for (const Arc& a : inst.arcs) g.offset[static_cast<size_t>(a.tail) + 1]++;
  for (size_t v = 0; v < static_cast<size_t>(g.num_nodes); ++v)
    g.offset[v + 1] += g.offset[v];
  g.head.resize(m);
  g.arc_id.resize(m);
  std::vector<int32_t> cursor(g.offset.begin(), g.offset.end() - 1);
  for (size_t i = 0; i < m; ++i) {
    const Arc& a = inst.arcs[i];
    int32_t slot = cursor[static_cast<size_t>(a.tail)]++;
    g.head[static_cast<size_t>(slot)] = a.head;
    g.arc_id[static_cast<size_t>(slot)] = static_cast<int32_t>(i);
  }
  return g;
}

}  // namespace arcedge
