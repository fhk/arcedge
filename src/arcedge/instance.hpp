#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace arcedge {

// cap < 0 means uncapacitated.
struct Arc {
  int32_t tail = 0;
  int32_t head = 0;
  double cost = 0.0;
  double cap = -1.0;
  bool capacitated() const { return cap >= 0.0; }
};

struct Commodity {
  int32_t src = 0;
  int32_t dst = 0;
  double demand = 0.0;
};

// A capacitated multicommodity min-cost flow instance. Time expansion is done
// by the generator; the solver only sees a plain directed graph.
struct Instance {
  int32_t num_nodes = 0;
  std::vector<Arc> arcs;
  std::vector<Commodity> commodities;

  static Instance load(const std::string& path);
  void save(const std::string& path) const;
};

}  // namespace arcedge
