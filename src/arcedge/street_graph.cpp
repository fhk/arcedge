#include "street_graph.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace arcedge {

StreetGraph StreetGraph::load(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open street graph file: " + path);
  StreetGraph g;
  std::string line;
  int64_t declared_arcs = -1;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == 'c') continue;
    std::istringstream ss(line);
    char tag;
    ss >> tag;
    if (tag == 'g') {
      ss >> g.num_nodes >> declared_arcs;
      g.lon.assign(static_cast<size_t>(g.num_nodes), 0.0);
      g.lat.assign(static_cast<size_t>(g.num_nodes), 0.0);
      g.arcs.reserve(static_cast<size_t>(declared_arcs));
    } else if (tag == 'v') {
      int32_t id;
      double lon, lat;
      ss >> id >> lon >> lat;
      if (id < 0 || id >= g.num_nodes)
        throw std::runtime_error("node id out of range in street graph file");
      g.lon[static_cast<size_t>(id)] = lon;
      g.lat[static_cast<size_t>(id)] = lat;
    } else if (tag == 'e') {
      SArc a;
      ss >> a.tail >> a.head >> a.cost;
      if (a.tail < 0 || a.tail >= g.num_nodes || a.head < 0 || a.head >= g.num_nodes)
        throw std::runtime_error("arc endpoint out of range in street graph file");
      g.arcs.push_back(a);
    } else {
      throw std::runtime_error("unknown line tag in street graph file: " + line);
    }
    if (!ss) throw std::runtime_error("malformed line in street graph file: " + line);
  }
  if (declared_arcs >= 0 && static_cast<int64_t>(g.arcs.size()) != declared_arcs)
    throw std::runtime_error("arc count mismatch in street graph file");
  return g;
}

}  // namespace arcedge
