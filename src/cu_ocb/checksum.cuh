#pragma once

#include "camellia.cuh"
#include "cuda_mem.cuh"
#include "cuda_utils.cuh"

#include <stdexcept>

namespace cu_ocb
{
__global__ void reduce_128xor(const u32* src, size_t count, u32* dst);

class Checksum128Calc
{
  // Computes xor of array of 128-bit elements.
 public:
  explicit Checksum128Calc(size_t cuda_block_size)
      : block_size_{cuda_block_size}
  {
    if (!block_size_) throw std::invalid_argument("block size cannot be zero");
  }

  bool compute(const u32* cu_data, size_t count128, u32* host_result);

 private:
  size_t ensureBuffer(size_t elem_count, CudaMem<u32>& buf, size_t& buf_count);

  void ensureBuffers(size_t size)
  {
    auto req = ensureBuffer(size, buf1_, buf1_count_);
    ensureBuffer(req, buf2_, buf2_count_);
  }

  size_t block_size_;
  CudaMem<u32> buf1_{};
  size_t buf1_count_{};
  CudaMem<u32> buf2_{};
  size_t buf2_count_{};
};

}  // namespace cu_ocb
