#pragma once

#include <vector>

#include "dijkstra.hpp"
#include "graph.hpp"
#include "instance.hpp"

namespace arcedge {

// Constructs a feasible fractional flow by routing commodities sequentially
// through the residual network (successive shortest paths, saturated arcs
// removed). Runs two passes -- one guided by the Lagrangian multipliers
// (reduced costs steer flow away from congested arcs) and one with pure costs
// -- and returns the cheaper feasible solution's true cost, or kInf if
// neither pass finds a feasible routing. If `flow_out` is given, it receives
// the winning pass's per-arc flow (empty when infeasible).
double primal_heuristic(const Instance& inst, const Graph& g,
                        const std::vector<double>& lambda,
                        std::vector<double>* flow_out = nullptr);

}  // namespace arcedge
