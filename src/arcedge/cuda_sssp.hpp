#pragma once

#include <memory>
#include <vector>

#include "dijkstra.hpp"
#include "instance.hpp"

namespace arcedge {

// GPU batched SSSP over a layered DAG (time-expanded graph): one bulk
// relaxation kernel per topological level, all commodities in the batch
// relaxed concurrently. Distances are FP32 on device with the (dist, parent
// arc) pair packed into a 64-bit word so a single atomicMin keeps them
// consistent; non-negative IEEE floats compare correctly as unsigned bits.
//
// Graph topology and level buckets stay resident on the device across
// subgradient iterations; only the reduced-cost vector is re-uploaded.
//
// Only compiled when ARCEDGE_CUDA is defined (cmake -DARCEDGE_CUDA=ON).
class CudaBatchSssp {
 public:
  CudaBatchSssp(const Instance& inst, const DagLevels& dag);
  ~CudaBatchSssp();

  static bool available();  // a CUDA device is present

  // Solves SSSP for every commodity under `cost` (converted to FP32).
  // Fills per-commodity distances and the demand-weighted per-arc flow
  // (`loads`), both accumulated on the device -- only ~4 bytes per arc plus
  // one distance per commodity cross the PCIe bus, not the full distance
  // matrix. Paths never materialize on the host.
  void solve(const std::vector<double>& cost,
             const std::vector<Commodity>& commodities,
             std::vector<double>& dists, std::vector<double>& loads);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace arcedge
