#pragma once
#include "cu_ocb/camellia.cuh"
#include "cu_ocb/cuda_mem.cuh"
#include "cu_ocb/cuda_utils.cuh"

#include <deque>
#include <stdexcept>
#include <utility>

namespace cu_ocb
{
__global__ void block_integral_128xor(u32* data, u32* base);

__global__ void apply_128xor(const u32* base, u32* blk);

__global__ void fill_L_reduce(size_t pos, const u32* L, u32* blk, u32* base);

__global__ void fill_L(size_t pos, const u32* L, u32* dst);

class OcbOffsetCalc
{
 public:
  explicit OcbOffsetCalc(size_t cuda_block_size = 64,
                         size_t cuda_block_size_for_buffer = 0)
      : block_size_{cuda_block_size},
        block_size_for_buffer_{cuda_block_size_for_buffer}
  {
    if (block_size_ < 4) throw std::invalid_argument("block size too small");
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }

  ~OcbOffsetCalc()
  {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  u32* compute(__uint128_t& last_offset, size_t pos128, size_t count128,
               const u32* L_table);

  float exec_time_{};

 private:
  u32* ensureBuffer(size_t buf_idx, size_t buf_size);
  bool computeIntegral(size_t buf_idx, size_t base_size);

  size_t block_size_;
  size_t block_size_for_buffer_;
  std::deque<std::pair<CudaMem<u32>, size_t>> buffers_;
  cudaEvent_t start_;
  cudaEvent_t stop_;
};

}  // namespace cu_ocb
