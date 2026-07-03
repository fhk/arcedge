#pragma once

#include <cstdint>
#include <vector>

#include "street_graph.hpp"

namespace arcedge {

struct DesignParams {
  double edge_cap = 500.0;         // max POIs sharing one street edge
  double hub_cost = 20000.0;       // fixed cost per opened hub
  double cable_cost_per_m = 10.0;  // fixed cost per meter of edge used
  unsigned seed = 1;
  int max_k = 0;                   // 0 = derive from POI count
  bool verbose = true;
};

struct DesignUsedEdge {
  int32_t u = 0, v = 0;  // node ids in the input street graph
  double load = 0.0;     // demand units carried (POI count)
};

struct DesignResult {
  int hubs = 0;
  double cable_m = 0.0;
  double hub_cost = 0.0;
  double cable_cost = 0.0;
  double total_cost = 0.0;
  bool feasible = false;
  std::vector<int32_t> hub_nodes;
  std::vector<DesignUsedEdge> used_edges;  // every edge carrying flow
  std::vector<int32_t> poi_hub;            // serving hub per input POI
};

// Capacitated hub-location / fixed-charge access network design: open hubs
// (each hub_cost) and route every POI's unit demand to some hub so that no
// edge carries more than edge_cap units; every edge that carries flow costs
// cable_cost_per_m * length once, regardless of load. Solved with a
// clustering matheuristic: k-means seeding on POI coordinates, multi-source
// Dijkstra assignment (shortest-path forest), subtree-load capacity repair
// that opens relief hubs, network re-centering (Lloyd rounds), and a search
// over k. Hubs may sit on any non-POI node.
DesignResult design(const StreetGraph& g, const std::vector<int32_t>& pois,
                    const DesignParams& p);

}  // namespace arcedge
