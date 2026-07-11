#pragma once

#include <cstdint>
#include <vector>

#include "street_graph.hpp"

namespace arcedge {

struct DesignParams {
  double edge_cap = 500.0;         // max demand units per edge; <= 0 = uncap
  double hub_cap = 0.0;            // max demand units served per hub; <= 0 = uncap
  double hub_cost = 20000.0;       // fixed cost per opened hub
  double cable_cost_per_m = 10.0;  // fixed cost per meter of edge used
  // Splice charging: every extra branch at a non-hub tree node of degree
  // >= 3 costs splice_cost (a degree-d node carries d-2 branch splices); a
  // branch at a mid-block node (see StreetGraph::midblock) pays the
  // surcharge on top. Hubs are exempt -- a hub IS a splice cabinet.
  double splice_cost = 0.0;
  double splice_midblock_surcharge = 0.0;
  unsigned seed = 1;
  int max_k = 0;                   // 0 = derive from POI count
  int threads = 0;                 // 0 = hardware concurrency
  // Demand nodes are customer-premises leaves by default (no transit).
  // Upper facility tiers put demand at street cabinets, which ARE transit.
  bool demand_transit = false;
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
  double splice_cost = 0.0;   // total splice charges (0 when disabled)
  int64_t splices = 0;        // branch splices charged (non-hub, degree>=3)
  int64_t splices_midblock = 0;  // of which at mid-block nodes
  double total_cost = 0.0;
  // Sum of per-cluster Wong dual-ascent lower bounds on cable meters,
  // conditional on the final clustering (hub set + POI assignment): it
  // measures tree quality, not global optimality. 0 when not computed.
  double cable_lb = 0.0;
  bool feasible = false;
  double millis = 0.0;
  std::vector<int32_t> hub_nodes;
  std::vector<DesignUsedEdge> used_edges;  // every edge carrying flow
  std::vector<int32_t> poi_hub;            // serving hub per input POI
};

// Capacitated hub-location / fixed-charge access network design: open hubs
// (each hub_cost) and route every POI's demand to some hub so that no edge
// carries more than edge_cap units and no hub serves more than hub_cap
// units; every edge that carries flow costs cable_cost_per_m * length once,
// regardless of load. Solved with a clustering matheuristic: k-means seeding
// on POI coordinates, multi-source Dijkstra assignment (shortest-path
// forest), subtree-load capacity repair that opens relief hubs (for
// overloaded edges AND over-capacity hubs), network re-centering, a greedy
// hub prune, and a search over k. Hubs may sit on any non-POI node.
// `demands` gives per-POI units (empty = all 1).
DesignResult design(const StreetGraph& g, const std::vector<int32_t>& pois,
                    const DesignParams& p,
                    const std::vector<double>& demands = {});

}  // namespace arcedge
