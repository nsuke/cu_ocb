#pragma once

#include "cu_ocb/constants.h"
#include "cu_ocb/cuda_utils.cuh"

#include <cassert>
#include <functional>
#include <utility>

namespace cu_ocb
{
using Allocator = cudaError_t (*)(void** ptr, size_t byte_count);
using Deleter = std::function<cudaError_t(void* ptr)>;

template <typename T>
struct CudaMem
{
  T* ptr_;
  Deleter deleter_;

  CudaMem(T* ptr = nullptr, Deleter deleter = cudaFree)
      : ptr_{ptr}, deleter_{deleter}
  {
  }

  ~CudaMem() { reset(); }

  CudaMem(const CudaMem&) = delete;
  CudaMem& operator=(const CudaMem&) = delete;

  CudaMem(CudaMem&& other) : CudaMem() { *this = std::move(other); }
  CudaMem& operator=(CudaMem&& other)
  {
    reset();
    std::swap(ptr_, other.ptr_);
    deleter_ = other.deleter_;
    return *this;
  }

  void reset()
  {
    if (ptr_)
      {
        deleter_(ptr_);
        checkError(__FILE__, __LINE__);
        ptr_ = nullptr;
      }
  }

  operator T*() { return ptr_; }
};

template <typename T = u32>
CudaMem<T> allocateManaged(size_t byte_count)
{
  // std::cerr << __func__ << " - " << byte_count << std::endl;
  assert(byte_count);
  assert(byte_count < 20'000'000'000);
  T* ptr;
  cudaMallocManaged(reinterpret_cast<void**>(&ptr), byte_count);
  if (!checkError(__FILE__, __LINE__))
    {
      throw std::runtime_error("failed to allocate device memory");
    }
  CudaMem<T> mem{ptr};
  return mem;
}

template <typename T = u32>
CudaMem<T> allocateDevice(size_t byte_count, const T* initial_data = nullptr,
                          Allocator allocator = &cudaMalloc)
{
  // std::cerr << __func__ << " - " << byte_count << std::endl;
  assert(byte_count);
  assert(byte_count < 20'000'000'000);
  T* ptr;
  allocator(reinterpret_cast<void**>(&ptr), byte_count);
  if (!checkError(__FILE__, __LINE__))
    {
      throw std::runtime_error("failed to allocate device memory");
    }
  CudaMem<T> mem{ptr};
  if (initial_data)
    {
      cudaMemcpy(mem, initial_data, byte_count, cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();
      if (!checkError(__FILE__, __LINE__))
        {
          throw std::runtime_error("failed to copy to device memory");
        }
    }
  return mem;
}

template <typename T = u32>
CudaMem<T> allocateHost(size_t byte_count, unsigned int flag = 0)
{
  // std::cerr << __func__ << " - " << byte_count << std::endl;
  assert(byte_count);
  assert(byte_count < 20'000'000'000);
  T* ptr;
  cudaHostAlloc(reinterpret_cast<void**>(&ptr), byte_count, flag);
  if (!checkError(__FILE__, __LINE__))
    throw std::runtime_error("Failed to allocate host memory");
  return {ptr, cudaFreeHost};
}

template <typename T>
CudaMem<T> registerCuda(T* ptr, size_t byte_count, unsigned int flag)
{
  // std::cerr << __func__ << " - " << byte_count << std::endl;
  cudaHostRegister(&ptr, byte_count, flag);
  if (!checkError(__FILE__, __LINE__))
    {
      throw std::runtime_error("Failed to allocate device memory");
    }
  return {ptr, cudaHostUnregister};
}

}  // namespace cu_ocb
