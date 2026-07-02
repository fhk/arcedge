#pragma once

#include "instance.hpp"

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

}  // namespace arcedge
