#pragma once

#include "cu_ocb/checksum.cuh"
#include "cu_ocb/cuda_mem.cuh"
#include "cu_ocb/cuda_utils.cuh"
#include "cu_ocb/ocb_camellia.h"
#include "cu_ocb/offset.cuh"

#include <array>
#include <string_view>

namespace cu_ocb
{
class OcbCamelliaImpl
{
 public:
  using Block = __uint128_t;

  explicit OcbCamelliaImpl(OcbConfig config);

  ~OcbCamelliaImpl();

  void generateKeytable(std::string_view key);

  void setKeytable(void* keytable)
  {
    if (config_.camellia_on_gpu)
      cudaMemcpy(keytable_, keytable, CAMELLIA_TABLE_BYTE_LEN,
                 cudaMemcpyHostToDevice);
    else
      {
        auto k = reinterpret_cast<const char*>(keytable);
        std::copy(k, k + CAMELLIA_TABLE_BYTE_LEN,
                  std::back_inserter(cpu_keytable_));
      }
  }

  size_t encrypt(std::string_view data, size_t index, std::string_view L,
                 Block& check_sum, Block& last_offset, void* result,
                 bool encrypt);

  const GpuTimeSpent* gpuTimeSpent() const
  {
    if (!config_.measure_gpu_time) { return nullptr; }
    time_.checksum_ = chk_.exec_time_;
    time_.ocb_offset_ = off_.exec_time_;
    return config_.measure_gpu_time ? &time_ : nullptr;
  }

  // TODO: not updated
  bool hasError() const { return has_error_; }

 private:
  void ensureBuffers(size_t req)
  {
    if (buf_size_ < req)
      {
        if (config_.camellia_on_gpu)
        {
          in_buf_ = allocateDevice(req);
          out_buf_ = allocateDevice(req);
        }
        if (config_.offset_mode == OffsetComputation::ApplyOnCpu ||
            (config_.offset_mode == OffsetComputation::Gpu &&
             config_.verify_with_cpu_result))
          {
            offsets_host_ = allocateHost(req);
            offsets_cpu_buf_.resize(req / sizeof(Block));
          }
        if (config_.offset_mode == OffsetComputation::ComputeOnCpu)
          {
            offsets_host_ = allocateHost(req, cudaHostAllocWriteCombined);
            offsets_cpu_ = allocateDevice(req);
          }
        buf_size_ = req;
      }
  }

  void applyOffsets(Block* data, size_t size);
  /*
  void applyOffsets(Block& last_offset, std::string_view L, size_t index,
                    Block* data, size_t size);
                    */

  u32* fillOffsets(Block& last_offset, size_t index, std::string_view L,
                   size_t size);

  void computeChecksum(Block& checksum, const u32* cu_data, const Block* data,
                       size_t size);

  OcbConfig config_;
  mutable GpuTimeSpent time_{};
  bool has_error_{};
  CudaMem<u32> keytable_;
  std::vector<char> cpu_keytable_;
  CudaMem<u32> L_;
  CudaMem<u32> in_buf_;
  CudaMem<u32> out_buf_;
  CudaMem<u32> offsets_cpu_;
  CudaMem<u32> offsets_host_;
  CudaMem<u32> checksum_host_;
  std::vector<Block> offsets_cpu_buf_;
  size_t buf_size_{};
  OcbOffsetCalc off_{config_.offset_threads, config_.encrypt_threads};
  Checksum128Calc chk_{config_.checksum_threads};

  cudaEvent_t start_;
  cudaEvent_t stop_;
};

}  // namespace cu_ocb
