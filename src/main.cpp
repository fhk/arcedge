#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>

#include "arcedge/dijkstra.hpp"
#include "arcedge/generator.hpp"
#include "arcedge/instance.hpp"
#include "arcedge/lagrangian.hpp"
#include "arcedge/street_graph.hpp"

namespace {

void usage() {
  std::printf(
      "arcedge - Stage 1 combinatorial core PoC (Lagrangian capacitated MCF)\n\n"
      "usage:\n"
      "  arcedge gen --out FILE [--street GRAPH | --width N --height N]\n"
      "              [--time N] [--commodities N] [--cap X (<=0 = uncapacitated)]\n"
      "              [--hubs N] [--hub-frac X] [--wait-cost X] [--seed N]\n"
      "  arcedge solve INSTANCE [--iters N] [--tol X] [--threads N]\n"
      "              [--primal-every N] [--result FILE] [--quiet]\n");
}

bool arg_match(int argc, char** argv, int& i, const char* name, std::string& out) {
  if (std::strcmp(argv[i], name) != 0) return false;
  if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
  out = argv[++i];
  return true;
}

int run_gen(int argc, char** argv) {
  arcedge::GeneratorParams p;
  std::string out_path, street_path, v;
  for (int i = 2; i < argc; ++i) {
    if (arg_match(argc, argv, i, "--out", v)) out_path = v;
    else if (arg_match(argc, argv, i, "--street", v)) street_path = v;
    else if (arg_match(argc, argv, i, "--wait-cost", v)) p.wait_cost = std::stod(v);
    else if (arg_match(argc, argv, i, "--width", v)) p.width = std::stoi(v);
    else if (arg_match(argc, argv, i, "--height", v)) p.height = std::stoi(v);
    else if (arg_match(argc, argv, i, "--time", v)) p.time_steps = std::stoi(v);
    else if (arg_match(argc, argv, i, "--commodities", v)) p.commodities = std::stoi(v);
    else if (arg_match(argc, argv, i, "--cap", v)) p.capacity = std::stod(v);
    else if (arg_match(argc, argv, i, "--hubs", v)) p.hubs = std::stoi(v);
    else if (arg_match(argc, argv, i, "--hub-frac", v)) p.hub_frac = std::stod(v);
    else if (arg_match(argc, argv, i, "--seed", v)) p.seed = static_cast<unsigned>(std::stoul(v));
    else { std::fprintf(stderr, "unknown option: %s\n", argv[i]); return 2; }
  }
  if (out_path.empty()) { usage(); return 2; }
  arcedge::Instance inst;
  if (!street_path.empty()) {
    const arcedge::StreetGraph street = arcedge::StreetGraph::load(street_path);
    std::printf("loaded street graph %s: %d nodes, %zu directed arcs\n",
                street_path.c_str(), street.num_nodes, street.arcs.size());
    inst = arcedge::generate_from_street(street, p);
  } else {
    inst = arcedge::generate(p);
  }
  inst.save(out_path);
  std::printf("wrote %s: %d nodes, %zu arcs, %zu commodities\n", out_path.c_str(),
              inst.num_nodes, inst.arcs.size(), inst.commodities.size());
  return 0;
}

int run_solve(int argc, char** argv) {
  if (argc < 3) { usage(); return 2; }
  const std::string inst_path = argv[2];
  arcedge::SolveOptions opt;
  opt.threads = static_cast<int>(std::thread::hardware_concurrency());
  std::string result_path, v;
  for (int i = 3; i < argc; ++i) {
    if (arg_match(argc, argv, i, "--iters", v)) opt.max_iters = std::stoi(v);
    else if (arg_match(argc, argv, i, "--tol", v)) opt.gap_tol = std::stod(v);
    else if (arg_match(argc, argv, i, "--threads", v)) opt.threads = std::stoi(v);
    else if (arg_match(argc, argv, i, "--primal-every", v)) opt.primal_every = std::stoi(v);
    else if (arg_match(argc, argv, i, "--result", v)) result_path = v;
    else if (std::strcmp(argv[i], "--quiet") == 0) opt.verbose = false;
    else { std::fprintf(stderr, "unknown option: %s\n", argv[i]); return 2; }
  }
  const arcedge::Instance inst = arcedge::Instance::load(inst_path);
  std::printf("loaded %s: %d nodes, %zu arcs, %zu commodities\n", inst_path.c_str(),
              inst.num_nodes, inst.arcs.size(), inst.commodities.size());
  const arcedge::SolveResult res = arcedge::solve(inst, opt);
  std::printf(
      "result: lb %.6f  ub %.6f  gap %.4f%%  iters %d  time %.1f ms\n",
      res.best_lb, res.best_ub, res.gap * 100.0, res.iters, res.millis);
  if (!result_path.empty()) {
    std::ofstream out(result_path);
    out << "lb " << res.best_lb << "\nub " << res.best_ub << "\ngap " << res.gap
        << "\niters " << res.iters << "\nms " << res.millis << "\n";
  }
  return res.best_ub < arcedge::kInf ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc >= 2 && std::strcmp(argv[1], "gen") == 0) return run_gen(argc, argv);
    if (argc >= 2 && std::strcmp(argv[1], "solve") == 0) return run_solve(argc, argv);
    usage();
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
