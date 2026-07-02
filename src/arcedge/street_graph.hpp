#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace arcedge {

// A static bidirectional street graph (e.g. imported from Overture Maps via
// scripts/overture_to_graph.py). Arc costs are lengths in meters; both
// directions of every street segment are present as separate arcs.
struct StreetGraph {
  struct SArc {
    int32_t tail = 0;
    int32_t head = 0;
    double cost = 0.0;
  };
  int32_t num_nodes = 0;
  std::vector<SArc> arcs;
  std::vector<double> lon, lat;  // per node, for provenance/debugging

  static StreetGraph load(const std::string& path);
};

}  // namespace arcedge
