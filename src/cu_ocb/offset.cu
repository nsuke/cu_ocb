#include "offset.cuh"

#include <cassert>

namespace cu_ocb
{
__global__ void block_integral_128xor(u32* data, u32* base)
{
  constexpr int elem_size = kU32ElemSize;
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  auto tid = threadIdx.x;

  extern __shared__ u32 s_blk[];
  u32 tmp[4];
  for (int e = 0; e < elem_size; ++e)
    s_blk[tid * elem_size + e] = tmp[e] = data[idx * elem_size + e];
  __syncthreads();

  for (int i = blockDim.x / 2; i; i >>= 1)
    {
      if (i <= tid)
        for (int e = 0; e < elem_size; ++e)
          tmp[e] = tmp[e] ^ s_blk[(tid - i) * elem_size + e];
      __syncthreads();

      if (i <= tid)
        for (int e = 0; e < elem_size; ++e) s_blk[tid * elem_size + e] = tmp[e];
      __syncthreads();
    }

  for (int e = 0; e < elem_size; ++e) data[idx * elem_size + e] = tmp[e];
  if (tid + 1 == blockDim.x)
    {
      for (int e = 0; e < elem_size; ++e)
        base[(blockIdx.x + 1) * elem_size + e] = tmp[e];
    }
}

__global__ void apply_128xor(const u32* base, u32* blk)
{
  constexpr int elem_size = kU32ElemSize;
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int e = 0; e < elem_size; ++e)
    blk[idx * elem_size + e] ^= base[blockIdx.x * elem_size + e];
}

__global__ void fill_L_reduce(size_t pos, const u32* L, u32* blk, u32* base)
{
  constexpr int elem_size = kU32ElemSize;
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  auto tid = threadIdx.x;
  auto l_idx = __clzll(__brevll(pos + idx + 1));
  assert(l_idx < 34);

  extern __shared__ u32 a_blk[];
  u32 tmp[4];
  for (int e = 0; e < elem_size; ++e)
    a_blk[tid * elem_size + e] = tmp[e] = L[l_idx * elem_size + e];
  __syncthreads();

  for (int i = blockDim.x / 2; i; i >>= 1)
    {
      if (i <= tid)
        for (int e = 0; e < elem_size; ++e)
          tmp[e] ^= a_blk[(tid - i) * elem_size + e];
      __syncthreads();

      if (i <= tid)
        for (int e = 0; e < elem_size; ++e) a_blk[tid * elem_size + e] = tmp[e];
      __syncthreads();
    }

  for (int e = 0; e < elem_size; ++e) blk[idx * elem_size + e] = tmp[e];
  if (tid + 1 == blockDim.x)
    {
      for (int e = 0; e < elem_size; ++e)
        base[(blockIdx.x + 1) * elem_size + e] = tmp[e];
    }
}

__global__ void fill_L(size_t pos, const u32* L, u32* dst)
{
  constexpr int elem_size = kU32ElemSize;
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  auto l_idx = __clzll(__brevll(pos + idx + 1));
  for (int e = 0; e < elem_size; ++e)
    dst[idx * elem_size + e] = L[l_idx * elem_size + e];
}

u32* OcbOffsetCalc::compute(__uint128_t& last_offset, size_t pos128,
                            size_t count128, const u32* L_table)
{
  assert(count128);
  auto grid_size = (count128 - 1) / block_size_ + 1;
  assert(grid_size);
  auto grid_size_for_buffer = block_size_for_buffer_
                                  ? (count128 - 1) / block_size_for_buffer_ + 1
                                  : grid_size;
  auto buf_size = std::max(grid_size * block_size_,
                           grid_size_for_buffer * block_size_for_buffer_) *
                  CAMELLIA_BLOCK_SIZE;
  auto next_grid_size = grid_size / block_size_ + 1;
  constexpr auto buf_idx = 0;
  if (u32* buf0 = ensureBuffer(buf_idx, buf_size))
    {
      if (u32* buf1 = ensureBuffer(
              buf_idx + 1, next_grid_size * block_size_ * CAMELLIA_BLOCK_SIZE))
        {
          cudaDeviceSynchronize();
          if (!checkError(__FILE__, __LINE__)) throw std::runtime_error("");

          cudaMemcpy(buf1, &last_offset, CAMELLIA_BLOCK_SIZE,
                     cudaMemcpyHostToDevice);
          cudaDeviceSynchronize();
          if (!checkError(__FILE__, __LINE__))
            throw std::runtime_error("cudaMemcpy -> buf1");
          fill_L_reduce<<<grid_size, block_size_,
                          block_size_ * CAMELLIA_BLOCK_SIZE>>>(pos128, L_table,
                                                               buf0, buf1);
          cudaDeviceSynchronize();
          if (!checkError(__FILE__, __LINE__))
            throw std::runtime_error("fill_L_reduce");
          if (!computeIntegral(buf_idx, grid_size))
            throw std::runtime_error("computeIntegral");
          cudaMemcpy(&last_offset, buf0 + (count128 - 1) * kU32ElemSize,
                     CAMELLIA_BLOCK_SIZE, cudaMemcpyDeviceToHost);
          cudaDeviceSynchronize();
          if (!checkError(__FILE__, __LINE__))
            throw std::runtime_error("cudaMemcpy ->last_offset");
          return buf0;
        }
      else
        {
          throw std::runtime_error("buf1");
        }
    }
  else
    {
      throw std::runtime_error("buf0");
    }
  return nullptr;
}

u32* OcbOffsetCalc::ensureBuffer(size_t buf_idx, size_t buf_size)
{
  // std::cerr << __func__ << " " << buf_idx << " " << buf_size << std::endl;
  if (!buf_size) throw std::invalid_argument("buf_size");
  if (buffers_.size() < buf_idx)
    throw std::invalid_argument("buf_idx too large");
  if (buffers_.size() == buf_idx)
    return buffers_.emplace_back(allocateDevice(buf_size), buf_size).first;
  auto&& [buf, size] = buffers_[buf_idx];
  if (size < buf_size)
    {
      buf = allocateDevice(buf_size);
      size = buf_size;
    }
  if (!buf) throw std::logic_error("buf");
  return buf;
}

bool OcbOffsetCalc::computeIntegral(size_t buf_idx, size_t base_size)
{
  // std::cerr << __func__ << " " << buf_idx << " " << base_size << std::endl;
  if (!base_size) throw std::invalid_argument("base_size");
  u32* buf0 = buffers_[buf_idx].first;
  u32* buf1 = buffers_[buf_idx + 1].first;
  if (base_size != 1)
    {
      ++buf_idx;
      auto grid_size = base_size / block_size_ + 1;
      auto next_grid_size = grid_size / block_size_ + 1;
      assert(grid_size < base_size);
      u32* buf2 = ensureBuffer(
          buf_idx + 1, next_grid_size * block_size_ * CAMELLIA_BLOCK_SIZE);
      if (!buf2) throw std::runtime_error("buf2");
      cudaDeviceSynchronize();
      block_integral_128xor<<<grid_size, block_size_,
                              block_size_ * CAMELLIA_BLOCK_SIZE>>>(buf1, buf2);
      cudaDeviceSynchronize();
      if (!checkError(__FILE__, __LINE__))
        throw std::runtime_error("block_integral_128xor");
      char zeros[CAMELLIA_BLOCK_SIZE]{};
      cudaMemcpy(buf2, zeros, CAMELLIA_BLOCK_SIZE, cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();
      if (!checkError(__FILE__, __LINE__)) throw std::runtime_error("");
      if (!computeIntegral(buf_idx, grid_size))
        throw std::runtime_error("recur");
    }
  cudaDeviceSynchronize();
  apply_128xor<<<base_size, block_size_>>>(buf1, buf0);
  cudaDeviceSynchronize();
  if (!checkError(__FILE__, __LINE__)) throw std::runtime_error("apply_128xor");
  return checkError(__FILE__, __LINE__);
}

}  // namespace cu_ocb
