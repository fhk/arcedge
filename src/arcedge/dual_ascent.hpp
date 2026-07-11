#pragma once

#include <cstdint>
#include <vector>

namespace arcedge {

struct DualAscentResult {
  // Valid lower bound on the cost of any arborescence rooted at `root`
  // reaching every terminal in the given directed graph (Wong 1984 dual
  // ascent on the cut formulation). Valid even when no tree is returned.
  double lower_bound = 0.0;
  // Arc indices (into the input arrays) of a primal arborescence extracted
  // from the reduced-cost graph: root-to-terminal shortest paths under
  // reduced costs, union over terminals. Empty when infeasible.
  std::vector<int32_t> tree_arcs;
  bool feasible = false;
};

// Wong dual ascent for the directed Steiner arborescence problem.
// Arcs are (tail[i] -> head[i]) with nonnegative cost[i]. Terminals equal to
// the root are ignored. Self-contained; no arcedge graph types.
DualAscentResult steiner_dual_ascent(int32_t n, const std::vector<int32_t>& tail,
                                     const std::vector<int32_t>& head,
                                     const std::vector<double>& cost,
                                     int32_t root,
                                     const std::vector<int32_t>& terminals);

}  // namespace arcedge
