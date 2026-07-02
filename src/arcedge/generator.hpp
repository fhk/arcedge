#pragma once

#include "instance.hpp"
#include "street_graph.hpp"

namespace arcedge {

struct GeneratorParams {
  int width = 20;         // street grid width
  int height = 20;        // street grid height
  int time_steps = 10;    // time-expansion layers
  int commodities = 20;
  double capacity = 4.0;  // per movement arc per time step; <= 0 => uncapacitated
  int hubs = 3;           // shared destinations that create congestion
  double hub_frac = 0.6;  // fraction of commodities destined for a hub
  double wait_cost = 0.01;
  unsigned seed = 1;
};

// Builds a time-expanded capacitated MCF instance over a 4-neighbor street
// grid. Node (t, v) has id t * width * height + v. Movement arcs advance one
// time layer and carry the street capacity; waiting arcs are uncapacitated.
// Each commodity starts at layer 0 and must reach its destination node at the
// final layer (waiting arcs absorb slack time).
Instance generate(const GeneratorParams& p);

// Same time expansion over a real street graph (e.g. an Overture Maps
// import): node (t, v) has id t * street.num_nodes + v, each street arc
// becomes a capacitated movement arc per layer, waiting arcs are
// uncapacitated. Commodity endpoints are sampled with a BFS hop-distance
// check so every origin can reach its destination within time_steps - 1
// moves. Grid fields (width/height) of GeneratorParams are ignored;
// wait_cost is in the street graph's cost unit (meters).
Instance generate_from_street(const StreetGraph& street, const GeneratorParams& p);

}  // namespace arcedge
