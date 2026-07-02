#pragma once

#include <limits>
#include <vector>

#include "graph.hpp"

namespace arcedge {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Reusable per-thread scratch space for shortest-path calls.
struct SpBuffers {
  std::vector<double> dist;
  std::vector<int32_t> parent_arc;  // arc used to reach node, -1 if none
};

// Single-source single-target Dijkstra with early exit. `cost` is indexed by
// arc id; arcs with cost >= kInf are treated as absent. Costs must be
// non-negative. Returns dist(src, dst), or kInf if dst is unreachable.
double shortest_path(const Graph& g, const std::vector<double>& cost, int32_t src,
                     int32_t dst, SpBuffers& buf);

// Walks parent_arc pointers back from dst and appends the arc ids of the
// src->dst path to `out` (in reverse order, which callers here never rely on).
void extract_path(const SpBuffers& buf, const Instance& inst, int32_t src, int32_t dst,
                  std::vector<int32_t>& out);

}  // namespace arcedge
