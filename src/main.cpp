#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <string>
#include <thread>

#include "arcedge/design.hpp"
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
      "              [--primal-every N] [--sp-backend auto|dijkstra|dag|cuda]\n"
      "              [--result FILE] [--flow FILE] [--no-primal] [--quiet]\n"
      "  arcedge design --graph FILE --pois FILE [--cap X] [--hub-cost X]\n"
      "              [--cable-cost X] [--max-k N] [--seed N] [--result FILE]\n"
      "              [--solution FILE] [--quiet]\n");
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
  std::string result_path, flow_path, v;
  for (int i = 3; i < argc; ++i) {
    if (arg_match(argc, argv, i, "--iters", v)) opt.max_iters = std::stoi(v);
    else if (arg_match(argc, argv, i, "--tol", v)) opt.gap_tol = std::stod(v);
    else if (arg_match(argc, argv, i, "--threads", v)) opt.threads = std::stoi(v);
    else if (arg_match(argc, argv, i, "--primal-every", v)) opt.primal_every = std::stoi(v);
    else if (arg_match(argc, argv, i, "--sp-backend", v)) opt.sp_backend = v;
    else if (arg_match(argc, argv, i, "--result", v)) result_path = v;
    else if (arg_match(argc, argv, i, "--flow", v)) flow_path = v;
    else if (std::strcmp(argv[i], "--quiet") == 0) opt.verbose = false;
    else if (std::strcmp(argv[i], "--no-primal") == 0) opt.primal = false;
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
    out << std::setprecision(15);  // bounds must not round across the LP optimum
    out << "lb " << res.best_lb << "\nub " << res.best_ub << "\ngap " << res.gap
        << "\niters " << res.iters << "\nms " << res.millis << "\n";
  }
  if (!flow_path.empty()) {
    if (res.flow.empty()) {
      std::fprintf(stderr, "no feasible flow to write to %s\n", flow_path.c_str());
    } else {
      std::ofstream out(flow_path);
      out << std::setprecision(15);
      out << "c arcedge flow: per-arc flow of the best_ub solution\n";
      size_t nz = 0;
      for (double f : res.flow) nz += f > 1e-9;
      out << "f " << res.flow.size() << ' ' << nz << '\n';
      for (size_t a = 0; a < res.flow.size(); ++a)
        if (res.flow[a] > 1e-9) out << "a " << a << ' ' << res.flow[a] << '\n';
      std::printf("wrote %s: %zu arcs with flow\n", flow_path.c_str(), nz);
    }
  }
  return res.best_ub < arcedge::kInf ? 0 : 1;
}

int run_design(int argc, char** argv) {
  arcedge::DesignParams p;
  std::string graph_path, pois_path, result_path, solution_path, v;
  for (int i = 2; i < argc; ++i) {
    if (arg_match(argc, argv, i, "--graph", v)) graph_path = v;
    else if (arg_match(argc, argv, i, "--solution", v)) solution_path = v;
    else if (arg_match(argc, argv, i, "--pois", v)) pois_path = v;
    else if (arg_match(argc, argv, i, "--cap", v)) p.edge_cap = std::stod(v);
    else if (arg_match(argc, argv, i, "--hub-cost", v)) p.hub_cost = std::stod(v);
    else if (arg_match(argc, argv, i, "--cable-cost", v)) p.cable_cost_per_m = std::stod(v);
    else if (arg_match(argc, argv, i, "--max-k", v)) p.max_k = std::stoi(v);
    else if (arg_match(argc, argv, i, "--seed", v)) p.seed = static_cast<unsigned>(std::stoul(v));
    else if (arg_match(argc, argv, i, "--result", v)) result_path = v;
    else if (std::strcmp(argv[i], "--quiet") == 0) p.verbose = false;
    else { std::fprintf(stderr, "unknown option: %s\n", argv[i]); return 2; }
  }
  if (graph_path.empty() || pois_path.empty()) { usage(); return 2; }
  const arcedge::StreetGraph g = arcedge::StreetGraph::load(graph_path);
  std::vector<int32_t> pois;
  {
    std::ifstream in(pois_path);
    if (!in) { std::fprintf(stderr, "cannot open pois file\n"); return 1; }
    int32_t id;
    while (in >> id) pois.push_back(id);
  }
  std::printf("design: %d nodes, %zu directed arcs, %zu POIs; cap %.0f, "
              "hub cost %.0f, cable cost %.2f/m\n",
              g.num_nodes, g.arcs.size(), pois.size(), p.edge_cap, p.hub_cost,
              p.cable_cost_per_m);
  const arcedge::DesignResult res = arcedge::design(g, pois, p);
  if (!res.feasible) {
    std::fprintf(stderr, "design infeasible: POIs disconnected or capacity too tight\n");
    return 1;
  }
  std::printf("design result: hubs %d (cost %.0f) + cable %.0f m (cost %.0f)\n"
              "TOTAL COST %.0f\n",
              res.hubs, res.hub_cost, res.cable_m, res.cable_cost, res.total_cost);
  if (!result_path.empty()) {
    std::ofstream out(result_path);
    out << std::setprecision(15);
    out << "hubs " << res.hubs << "\ncable_m " << res.cable_m << "\nhub_cost "
        << res.hub_cost << "\ncable_cost " << res.cable_cost << "\ntotal_cost "
        << res.total_cost << "\n";
    out << "hub_nodes";
    for (int32_t h : res.hub_nodes) out << ' ' << h;
    out << "\n";
  }
  if (!solution_path.empty()) {
    std::ofstream out(solution_path);
    out << std::setprecision(15);
    out << "c arcedge design solution: h <hub-node>; e <u> <v> <load>; "
           "p <poi-node> <serving-hub-node>\n";
    for (int32_t h : res.hub_nodes) out << "h " << h << '\n';
    for (const arcedge::DesignUsedEdge& e : res.used_edges)
      out << "e " << e.u << ' ' << e.v << ' ' << e.load << '\n';
    for (size_t i = 0; i < pois.size(); ++i)
      out << "p " << pois[i] << ' ' << res.poi_hub[i] << '\n';
    std::printf("wrote %s: %d hubs, %zu used edges, %zu POI assignments\n",
                solution_path.c_str(), res.hubs, res.used_edges.size(),
                pois.size());
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc >= 2 && std::strcmp(argv[1], "gen") == 0) return run_gen(argc, argv);
    if (argc >= 2 && std::strcmp(argv[1], "solve") == 0) return run_solve(argc, argv);
    if (argc >= 2 && std::strcmp(argv[1], "design") == 0) return run_design(argc, argv);
    usage();
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
