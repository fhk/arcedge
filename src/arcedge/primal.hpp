#pragma once

#include <vector>

#include "dijkstra.hpp"
#include "graph.hpp"
#include "instance.hpp"

namespace arcedge {

// Constructs a feasible fractional flow with wave-parallel successive
// shortest paths: each wave batch-solves the still-unrouted commodities in
// parallel against a frozen residual snapshot, then commits their paths in a
// fixed order (partial pushes allowed; commodities whose path saturated
// retry next wave), so the result is deterministic regardless of threading.
//
// `include_pure` additionally runs a pure-cost pass and keeps the cheaper
// feasible routing -- only worth it when lambda is cold (all zero); with
// warm multipliers the guided pass wins in practice and the pure pass is
// skipped to halve the cost. Returns the true cost, or kInf if infeasible.
// If `flow_out` is given it receives the winning per-arc flow.
double primal_heuristic(const Instance& inst, const Graph& g,
                        const std::vector<double>& lambda, int threads,
                        bool include_pure,
                        std::vector<double>* flow_out = nullptr);

}  // namespace arcedge
