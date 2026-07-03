// CUDA backend for batched SSSP on layered DAGs. See cuda_sssp.hpp.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include "cuda_sssp.hpp"

namespace arcedge {

namespace {

#define ARCEDGE_CUDA_CHECK(call)                                              \
  do {                                                                        \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess)                                                 \
      throw std::runtime_error(std::string("CUDA error: ") +                  \
                               cudaGetErrorString(err__) + " at " __FILE__    \
                               ":" + std::to_string(__LINE__));               \
  } while (0)

constexpr unsigned kNoParent = 0xFFFFFFFFu;
constexpr unsigned kInfBits = 0x7F800000u;  // +inf as IEEE-754 bits
constexpr unsigned long long kInfPacked =
    (static_cast<unsigned long long>(kInfBits) << 32) | kNoParent;

__global__ void init_packed(unsigned long long* packed, size_t total) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
       i < total; i += static_cast<size_t>(gridDim.x) * blockDim.x)
    packed[i] = kInfPacked;
}

__global__ void set_sources(unsigned long long* packed, const int32_t* srcs,
                            int k_count, int n) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < k_count)
    packed[static_cast<size_t>(k) * n + srcs[k]] =
        static_cast<unsigned long long>(kNoParent);  // dist bits 0 | no parent
}

// Relaxes every arc in one topological level for every commodity. dist bits
// live in the high word so the packed 64-bit atomicMin orders by distance
// (ties broken toward the smaller arc id, deterministically).
__global__ void relax_level(const int32_t* tail, const int32_t* head,
                            const float* cost, const int32_t* bucket,
                            long long bucket_size, int k_count, int n,
                            unsigned long long* packed) {
  const long long total = bucket_size * k_count;
  for (long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
       idx < total; idx += static_cast<long long>(gridDim.x) * blockDim.x) {
    const int k = static_cast<int>(idx / bucket_size);
    const int32_t a = bucket[idx % bucket_size];
    const unsigned long long pt =
        packed[static_cast<size_t>(k) * n + tail[a]];
    const unsigned dbits = static_cast<unsigned>(pt >> 32);
    if (dbits >= kInfBits) continue;  // unreached tail (or blocked)
    const float nd = __uint_as_float(dbits) + cost[a];
    const unsigned long long np =
        (static_cast<unsigned long long>(__float_as_uint(nd)) << 32) |
        static_cast<unsigned>(a);
    atomicMin(&packed[static_cast<size_t>(k) * n + head[a]], np);
  }
}

// One thread per commodity: read the destination distance, then walk the
// parent chain accumulating demand onto every arc of the path. Loads stay on
// the device; only they and the K distances are copied back.
__global__ void gather_paths(const unsigned long long* packed,
                             const int32_t* tail, const int32_t* srcs,
                             const int32_t* dsts, const float* demand,
                             int k_count, int n, float* dist_out,
                             float* loads) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= k_count) return;
  const unsigned long long* row = packed + static_cast<size_t>(k) * n;
  const unsigned dbits = static_cast<unsigned>(row[dsts[k]] >> 32);
  dist_out[k] = __uint_as_float(dbits);
  if (dbits >= kInfBits) return;  // unreachable: caller sees +inf
  const float q = demand[k];
  int32_t v = dsts[k];
  for (int guard = 0; v != srcs[k] && guard < n; ++guard) {
    const unsigned a = static_cast<unsigned>(row[v]);
    atomicAdd(&loads[a], q);
    v = tail[a];
  }
}

int grid_for(long long work, int block = 256) {
  const long long g = (work + block - 1) / block;
  return static_cast<int>(g > 4096 ? 4096 : (g < 1 ? 1 : g));
}

}  // namespace

struct CudaBatchSssp::Impl {
  int n = 0;
  int32_t num_levels = 0;
  std::vector<int32_t> level_off;    // host copy of bucket offsets
  std::vector<int32_t> level_of;     // host copy for the sweep bound
  size_t m = 0;
  int32_t* d_tail = nullptr;
  int32_t* d_head = nullptr;
  float* d_cost = nullptr;
  int32_t* d_bucket = nullptr;
  int32_t* d_srcs = nullptr;
  int32_t* d_dsts = nullptr;
  float* d_demand = nullptr;
  float* d_dist_out = nullptr;
  float* d_loads = nullptr;
  unsigned long long* d_packed = nullptr;
  size_t packed_cap = 0;  // commodities currently allocated for
  std::vector<float> h_cost;
  std::vector<float> h_small;  // reused staging for K-sized transfers
  std::vector<float> h_loads;
  std::vector<int32_t> h_ids;

  ~Impl() {
    cudaFree(d_tail);
    cudaFree(d_head);
    cudaFree(d_cost);
    cudaFree(d_bucket);
    cudaFree(d_srcs);
    cudaFree(d_dsts);
    cudaFree(d_demand);
    cudaFree(d_dist_out);
    cudaFree(d_loads);
    cudaFree(d_packed);
  }
};

bool CudaBatchSssp::available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

CudaBatchSssp::CudaBatchSssp(const Instance& inst, const DagLevels& dag)
    : impl_(new Impl) {
  if (!dag.is_dag)
    throw std::runtime_error("CUDA SSSP backend requires a DAG instance");
  Impl& im = *impl_;
  im.n = inst.num_nodes;
  im.num_levels = dag.num_levels;
  im.level_off = dag.level_off;
  im.level_of = dag.level_of;
  const size_t m = inst.arcs.size();
  std::vector<int32_t> tail(m), head(m);
  for (size_t a = 0; a < m; ++a) {
    tail[a] = inst.arcs[a].tail;
    head[a] = inst.arcs[a].head;
  }
  ARCEDGE_CUDA_CHECK(cudaMalloc(&im.d_tail, m * sizeof(int32_t)));
  ARCEDGE_CUDA_CHECK(cudaMalloc(&im.d_head, m * sizeof(int32_t)));
  ARCEDGE_CUDA_CHECK(cudaMalloc(&im.d_cost, m * sizeof(float)));
  ARCEDGE_CUDA_CHECK(cudaMalloc(&im.d_bucket, m * sizeof(int32_t)));
  ARCEDGE_CUDA_CHECK(cudaMemcpy(im.d_tail, tail.data(), m * sizeof(int32_t),
                                cudaMemcpyHostToDevice));
  ARCEDGE_CUDA_CHECK(cudaMemcpy(im.d_head, head.data(), m * sizeof(int32_t),
                                cudaMemcpyHostToDevice));
  ARCEDGE_CUDA_CHECK(cudaMemcpy(im.d_bucket, dag.arcs_by_level.data(),
                                m * sizeof(int32_t), cudaMemcpyHostToDevice));
  ARCEDGE_CUDA_CHECK(cudaMalloc(&im.d_loads, m * sizeof(float)));
  im.m = m;
  im.h_cost.resize(m);
  im.h_loads.resize(m);
}

CudaBatchSssp::~CudaBatchSssp() = default;

void CudaBatchSssp::solve(const std::vector<double>& cost,
                          const std::vector<Commodity>& commodities,
                          std::vector<double>& dists, std::vector<double>& loads) {
  Impl& im = *impl_;
  const size_t m = cost.size();
  const size_t k_count = commodities.size();
  const size_t total = k_count * static_cast<size_t>(im.n);

  for (size_t a = 0; a < m; ++a)
    im.h_cost[a] = cost[a] >= kInf ? std::numeric_limits<float>::infinity()
                                   : static_cast<float>(cost[a]);
  ARCEDGE_CUDA_CHECK(cudaMemcpy(im.d_cost, im.h_cost.data(), m * sizeof(float),
                                cudaMemcpyHostToDevice));

  if (im.packed_cap < k_count) {
    cudaFree(im.d_packed);
    cudaFree(im.d_srcs);
    cudaFree(im.d_dsts);
    cudaFree(im.d_demand);
    cudaFree(im.d_dist_out);
    ARCEDGE_CUDA_CHECK(
        cudaMalloc(&im.d_packed, total * sizeof(unsigned long long)));
    ARCEDGE_CUDA_CHECK(cudaMalloc(&im.d_srcs, k_count * sizeof(int32_t)));
    ARCEDGE_CUDA_CHECK(cudaMalloc(&im.d_dsts, k_count * sizeof(int32_t)));
    ARCEDGE_CUDA_CHECK(cudaMalloc(&im.d_demand, k_count * sizeof(float)));
    ARCEDGE_CUDA_CHECK(cudaMalloc(&im.d_dist_out, k_count * sizeof(float)));
    im.packed_cap = k_count;
  }
  im.h_ids.resize(k_count);
  im.h_small.resize(k_count);
  int32_t max_dst_level = 0;
  for (size_t k = 0; k < k_count; ++k) {
    im.h_ids[k] = commodities[k].src;
    im.h_small[k] = static_cast<float>(commodities[k].demand);
    max_dst_level = std::max(
        max_dst_level, im.level_of[static_cast<size_t>(commodities[k].dst)]);
  }
  ARCEDGE_CUDA_CHECK(cudaMemcpy(im.d_srcs, im.h_ids.data(),
                                k_count * sizeof(int32_t),
                                cudaMemcpyHostToDevice));
  ARCEDGE_CUDA_CHECK(cudaMemcpy(im.d_demand, im.h_small.data(),
                                k_count * sizeof(float),
                                cudaMemcpyHostToDevice));
  for (size_t k = 0; k < k_count; ++k) im.h_ids[k] = commodities[k].dst;
  ARCEDGE_CUDA_CHECK(cudaMemcpy(im.d_dsts, im.h_ids.data(),
                                k_count * sizeof(int32_t),
                                cudaMemcpyHostToDevice));

  init_packed<<<grid_for(static_cast<long long>(total)), 256>>>(im.d_packed,
                                                                total);
  set_sources<<<grid_for(static_cast<long long>(k_count)), 256>>>(
      im.d_packed, im.d_srcs, static_cast<int>(k_count), im.n);

  // The level sweep: arcs whose tail sits at level >= max destination level
  // cannot lie on any wanted path.
  for (int32_t l = 0; l < std::min(im.num_levels, max_dst_level); ++l) {
    const long long bucket_size =
        im.level_off[static_cast<size_t>(l) + 1] - im.level_off[static_cast<size_t>(l)];
    if (bucket_size == 0) continue;
    relax_level<<<grid_for(bucket_size * static_cast<long long>(k_count)), 256>>>(
        im.d_tail, im.d_head, im.d_cost,
        im.d_bucket + im.level_off[static_cast<size_t>(l)], bucket_size,
        static_cast<int>(k_count), im.n, im.d_packed);
  }

  // Device-side path walk: loads and distances come back over PCIe instead
  // of the K x N packed matrix (6.7 MB vs ~0.5 GB on the SF instance).
  ARCEDGE_CUDA_CHECK(cudaMemset(im.d_loads, 0, m * sizeof(float)));
  gather_paths<<<grid_for(static_cast<long long>(k_count)), 256>>>(
      im.d_packed, im.d_tail, im.d_srcs, im.d_dsts, im.d_demand,
      static_cast<int>(k_count), im.n, im.d_dist_out, im.d_loads);
  ARCEDGE_CUDA_CHECK(cudaGetLastError());

  ARCEDGE_CUDA_CHECK(cudaMemcpy(im.h_small.data(), im.d_dist_out,
                                k_count * sizeof(float),
                                cudaMemcpyDeviceToHost));
  ARCEDGE_CUDA_CHECK(cudaMemcpy(im.h_loads.data(), im.d_loads,
                                m * sizeof(float), cudaMemcpyDeviceToHost));

  dists.assign(k_count, kInf);
  for (size_t k = 0; k < k_count; ++k)
    if (im.h_small[k] < std::numeric_limits<float>::infinity())
      dists[k] = static_cast<double>(im.h_small[k]);
  loads.assign(m, 0.0);
  for (size_t a = 0; a < m; ++a) loads[a] = static_cast<double>(im.h_loads[a]);
}

}  // namespace arcedge
