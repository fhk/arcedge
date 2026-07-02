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

// Topological levels of a DAG instance (time-expanded graphs are layered
// DAGs). Arcs are bucketed by tail level; relaxing buckets in ascending level
// order computes exact shortest paths in one O(m) sweep with no priority
// queue -- the same bulk-relaxation structure the CUDA backend executes, one
// kernel launch per level.
struct DagLevels {
  bool is_dag = false;
  int32_t num_levels = 0;
  std::vector<int32_t> level_of;       // per node
  std::vector<int32_t> arcs_by_level;  // arc ids grouped by tail level
  std::vector<int32_t> level_off;      // size num_levels + 1 into arcs_by_level

  static DagLevels build(const Instance& inst);
};

// Exact SSSP on a DAG via the level sweep. Semantics match shortest_path():
// same cost vector convention (>= kInf means absent), same buffers.
double dag_shortest_path(const DagLevels& dag, const Instance& inst,
                         const std::vector<double>& cost, int32_t src, int32_t dst,
                         SpBuffers& buf);

}  // namespace arcedge
