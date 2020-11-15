#include "checksum.cuh"

#include <algorithm>

namespace cu_ocb
{
__global__ void reduce_128xor(const u32* src, size_t count, u32* dst)
{
  constexpr int elem_size = kU32ElemSize;
  auto idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  auto tid = threadIdx.x;

  extern __shared__ u32 s_red[];
  u32 tmp[4];
  for (int e = 0; e < elem_size; ++e)
    s_red[tid * elem_size + e] = tmp[e] =
        (idx < count ? src[idx * elem_size + e] : 0) ^
        (idx + blockDim.x < count ? src[(idx + blockDim.x) * elem_size + e]
                                  : 0);
  __syncthreads();

  for (int i = blockDim.x / 2; i; i >>= 1)
    {
      if (tid < i)
        {
          for (int e = 0; e < elem_size; ++e)
            s_red[tid * elem_size + e] = tmp[e] =
                tmp[e] ^ s_red[(tid + i) * elem_size + e];
        }
      __syncthreads();
    }

  if (tid == 0)
    for (int e = 0; e < elem_size; ++e)
      dst[blockIdx.x * elem_size + e] = tmp[e];
}

size_t Checksum128Calc::ensureBuffer(size_t elem_count, CudaMem<u32>& buf,
                                     size_t& buf_count)
{
  if (!elem_count) { return 0; }
  auto req = (elem_count - 1) / block_size_ + 1;
  if (buf_count < req)
    {
      buf = allocateDevice(req * CAMELLIA_BLOCK_SIZE);
      buf_count = req;
    }
  return req;
}

bool Checksum128Calc::compute(const u32* cu_data, size_t count128,
                              u32* host_result)
{
  if (!count128) throw std::invalid_argument("count128");
  ensureBuffers(count128);
  if (!buf1_) throw std::runtime_error("buf1_");
  if (!buf2_) throw std::runtime_error("buf2_");
  u32* src = buf1_;
  u32* dst = buf2_;
  bool first{true};
  while (count128 != 1)
    {
      std::swap(src, dst);
      ;
      auto grid_size = (count128 - 1) / (block_size_ * 2) + 1;
      reduce_128xor<<<grid_size, block_size_,
                      block_size_ * CAMELLIA_BLOCK_SIZE>>>(
          first ? cu_data : src, count128, dst);
      // cudaDeviceSynchronize();
      if (!checkError(__FILE__, __LINE__))
        throw std::runtime_error("reduce_128xor");
      assert(grid_size < count128);
      count128 = grid_size;
      first = false;
    }
  cudaMemcpy(host_result, first ? cu_data : dst, CAMELLIA_BLOCK_SIZE,
             cudaMemcpyDeviceToHost);
  // cudaDeviceSynchronize();
  if (!checkError(__FILE__, __LINE__)) { throw std::runtime_error(""); }
  return checkError(__FILE__, __LINE__);
}

}  // namespace cu_ocb
