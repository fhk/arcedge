#pragma once

#include <vector>

#include "instance.hpp"

namespace arcedge {

// CSR out-adjacency view over an Instance's arcs.
struct Graph {
  int32_t num_nodes = 0;
  std::vector<int32_t> offset;  // size num_nodes + 1
  std::vector<int32_t> head;    // size arcs
  std::vector<int32_t> arc_id;  // size arcs, index into Instance::arcs

  static Graph build(const Instance& inst);
};

}  // namespace arcedge
