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
  // Fills per-commodity distances and path arc ids, matching the semantics
  // of the CPU backends (paths in reverse order).
  void solve(const std::vector<double>& cost,
             const std::vector<Commodity>& commodities,
             std::vector<double>& dists,
             std::vector<std::vector<int32_t>>& paths);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace arcedge
