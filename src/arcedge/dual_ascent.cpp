#include "dual_ascent.hpp"

#include <limits>
#include <queue>

namespace arcedge {

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();
}

// Wong dual ascent on the cut formulation of the directed Steiner
// arborescence problem: every node set W that contains a terminal but not
// the root must be entered by at least one arc. The ascent keeps reduced
// costs red[a] = cost[a] - sum of dual raises on cuts containing a
// (nonnegative throughout), and repeatedly picks an unsatisfied terminal t,
// grows W = { v : v reaches t through zero-reduced arcs } by reverse BFS,
// and raises the dual on delta^-(W) by the minimum reduced cost in the cut.
// Each raise zeroes at least one arc and W only ever grows, so the number
// of raises is at most m + |terminals|. LB = sum of raises.
DualAscentResult steiner_dual_ascent(int32_t n, const std::vector<int32_t>& tail,
                                     const std::vector<int32_t>& head,
                                     const std::vector<double>& cost,
                                     int32_t root,
                                     const std::vector<int32_t>& terminals) {
  DualAscentResult res;
  const size_t m = tail.size();
  std::vector<double> red(cost);

  // In-arc CSR (arcs entering each node) for the reverse BFS.
  std::vector<int32_t> in_off(static_cast<size_t>(n) + 1, 0);
  for (size_t a = 0; a < m; ++a) in_off[static_cast<size_t>(head[a]) + 1]++;
  for (size_t v = 0; v < static_cast<size_t>(n); ++v) in_off[v + 1] += in_off[v];
  std::vector<int32_t> in_arc(m);
  {
    std::vector<int32_t> cur(in_off.begin(), in_off.end() - 1);
    for (size_t a = 0; a < m; ++a)
      in_arc[static_cast<size_t>(cur[static_cast<size_t>(head[a])]++)] =
          static_cast<int32_t>(a);
  }

  std::vector<char> active(static_cast<size_t>(n), 0);
  std::vector<int32_t> todo;
  for (int32_t t : terminals)
    if (t != root && !active[static_cast<size_t>(t)]) {
      active[static_cast<size_t>(t)] = 1;
      todo.push_back(t);
    }

  std::vector<uint32_t> stamp(static_cast<size_t>(n), 0);
  uint32_t version = 0;
  std::vector<int32_t> bfs, wset;
  // Raises are bounded by m + |terminals|; the guard is belt-and-braces.
  size_t raises_left = 2 * m + todo.size() + 16;
  for (size_t ti = 0; ti < todo.size(); ++ti) {
    const int32_t t = todo[ti];
    while (active[static_cast<size_t>(t)]) {
      // W = nodes that reach t via zero-reduced arcs (reverse BFS from t).
      ++version;
      bfs.clear();
      wset.clear();
      bfs.push_back(t);
      stamp[static_cast<size_t>(t)] = version;
      bool hit_root = false;
      for (size_t qh = 0; qh < bfs.size() && !hit_root; ++qh) {
        const int32_t w = bfs[qh];
        wset.push_back(w);
        for (int32_t i = in_off[static_cast<size_t>(w)];
             i < in_off[static_cast<size_t>(w) + 1]; ++i) {
          const int32_t a = in_arc[static_cast<size_t>(i)];
          if (red[static_cast<size_t>(a)] > 0.0) continue;
          const int32_t u = tail[static_cast<size_t>(a)];
          if (stamp[static_cast<size_t>(u)] == version) continue;
          stamp[static_cast<size_t>(u)] = version;
          if (u == root) { hit_root = true; break; }
          bfs.push_back(u);
        }
      }
      if (hit_root) {
        active[static_cast<size_t>(t)] = 0;
        break;
      }
      // Raise the dual on the cut delta^-(W) by the minimum reduced cost.
      double delta = kInf;
      for (int32_t w : wset)
        for (int32_t i = in_off[static_cast<size_t>(w)];
             i < in_off[static_cast<size_t>(w) + 1]; ++i) {
          const int32_t a = in_arc[static_cast<size_t>(i)];
          const double r = red[static_cast<size_t>(a)];
          if (r > 0.0 && stamp[static_cast<size_t>(tail[static_cast<size_t>(a)])] != version &&
              r < delta)
            delta = r;
        }
      if (delta == kInf) {
        // Terminal cannot be reached at all: bound stays valid, no tree.
        active[static_cast<size_t>(t)] = 0;
        res.lower_bound = kInf;
        return res;
      }
      res.lower_bound += delta;
      for (int32_t w : wset)
        for (int32_t i = in_off[static_cast<size_t>(w)];
             i < in_off[static_cast<size_t>(w) + 1]; ++i) {
          const int32_t a = in_arc[static_cast<size_t>(i)];
          if (red[static_cast<size_t>(a)] > 0.0 &&
              stamp[static_cast<size_t>(tail[static_cast<size_t>(a)])] != version)
            red[static_cast<size_t>(a)] -= delta;
        }
      if (--raises_left == 0) break;  // never expected; keep the LB, no tree
    }
    if (raises_left == 0) return res;
  }

  // Primal extraction: shortest paths from the root under reduced costs
  // (every terminal is reachable at reduced distance 0 after the ascent);
  // the union of parent paths to the terminals is the tree. Union of
  // shortest-path-tree paths never has non-terminal leaves, so no pruning
  // pass is needed.
  std::vector<int32_t> out_off(static_cast<size_t>(n) + 1, 0);
  for (size_t a = 0; a < m; ++a) out_off[static_cast<size_t>(tail[a]) + 1]++;
  for (size_t v = 0; v < static_cast<size_t>(n); ++v) out_off[v + 1] += out_off[v];
  std::vector<int32_t> out_arc(m);
  {
    std::vector<int32_t> cur(out_off.begin(), out_off.end() - 1);
    for (size_t a = 0; a < m; ++a)
      out_arc[static_cast<size_t>(cur[static_cast<size_t>(tail[a])]++)] =
          static_cast<int32_t>(a);
  }
  std::vector<double> dist(static_cast<size_t>(n), kInf);
  std::vector<int32_t> parc(static_cast<size_t>(n), -1);
  using Item = std::pair<double, int32_t>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> pq;
  dist[static_cast<size_t>(root)] = 0.0;
  pq.emplace(0.0, root);
  while (!pq.empty()) {
    auto [d, u] = pq.top();
    pq.pop();
    if (d > dist[static_cast<size_t>(u)]) continue;
    for (int32_t i = out_off[static_cast<size_t>(u)];
         i < out_off[static_cast<size_t>(u) + 1]; ++i) {
      const int32_t a = out_arc[static_cast<size_t>(i)];
      const int32_t v = head[static_cast<size_t>(a)];
      const double nd = d + red[static_cast<size_t>(a)];
      if (nd < dist[static_cast<size_t>(v)]) {
        dist[static_cast<size_t>(v)] = nd;
        parc[static_cast<size_t>(v)] = a;
        pq.emplace(nd, v);
      }
    }
  }
  std::vector<char> in_tree(m, 0);
  for (int32_t t : terminals) {
    if (t == root) continue;
    if (dist[static_cast<size_t>(t)] == kInf) return res;  // no tree
    for (int32_t v = t; v != root;) {
      const int32_t a = parc[static_cast<size_t>(v)];
      if (in_tree[static_cast<size_t>(a)]) break;  // joined an earlier path
      in_tree[static_cast<size_t>(a)] = 1;
      v = tail[static_cast<size_t>(a)];
    }
  }
  for (size_t a = 0; a < m; ++a)
    if (in_tree[a]) res.tree_arcs.push_back(static_cast<int32_t>(a));
  res.feasible = true;
  return res;
}

}  // namespace arcedge
