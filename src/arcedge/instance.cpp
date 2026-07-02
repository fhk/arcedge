#include "instance.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace arcedge {

Instance Instance::load(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open instance file: " + path);
  Instance inst;
  std::string line;
  int64_t declared_arcs = -1, declared_commodities = -1;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == 'c') continue;
    std::istringstream ss(line);
    char tag;
    ss >> tag;
    if (tag == 'p') {
      ss >> inst.num_nodes >> declared_arcs >> declared_commodities;
      inst.arcs.reserve(static_cast<size_t>(declared_arcs));
      inst.commodities.reserve(static_cast<size_t>(declared_commodities));
    } else if (tag == 'a') {
      Arc a;
      ss >> a.tail >> a.head >> a.cost >> a.cap;
      inst.arcs.push_back(a);
    } else if (tag == 'k') {
      Commodity k;
      ss >> k.src >> k.dst >> k.demand;
      inst.commodities.push_back(k);
    } else {
      throw std::runtime_error("unknown line tag in instance file: " + line);
    }
    if (!ss) throw std::runtime_error("malformed line in instance file: " + line);
  }
  if (declared_arcs >= 0 && static_cast<int64_t>(inst.arcs.size()) != declared_arcs)
    throw std::runtime_error("arc count mismatch in instance file");
  if (declared_commodities >= 0 &&
      static_cast<int64_t>(inst.commodities.size()) != declared_commodities)
    throw std::runtime_error("commodity count mismatch in instance file");
  return inst;
}

void Instance::save(const std::string& path) const {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write instance file: " + path);
  out << "c arcedge capacitated MCF instance (cap < 0 => uncapacitated)\n";
  out << "p " << num_nodes << ' ' << arcs.size() << ' ' << commodities.size() << '\n';
  for (const Arc& a : arcs)
    out << "a " << a.tail << ' ' << a.head << ' ' << a.cost << ' ' << a.cap << '\n';
  for (const Commodity& k : commodities)
    out << "k " << k.src << ' ' << k.dst << ' ' << k.demand << '\n';
}

}  // namespace arcedge
