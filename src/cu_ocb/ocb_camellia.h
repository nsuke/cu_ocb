#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <string_view>

namespace cu_ocb
{
enum class OffsetComputation
{
  Gpu,
  ComputeOnCpu,
  ApplyOnCpu,
};

struct OcbConfig
{
  size_t encrypt_threads{256};
  size_t offset_threads{64};
  size_t checksum_threads{32};

  size_t minimum_blocks{1};
  bool process_incomplete_block{true};
  OffsetComputation offset_mode{OffsetComputation::Gpu};
  bool checksum_on_gpu{true};

  bool verify_with_cpu_result{};
  bool measure_gpu_time{};
  bool debug{};
};

struct GpuTimeSpent
{
  float encryption_;
  float checksum_;
  float ocb_offset_;
};

class OcbCamelliaImpl;

class OcbCamellia
{
 public:
  using Block = __uint128_t;

  explicit OcbCamellia(OcbConfig config);
  ~OcbCamellia();

  void generateKeytable(std::string_view key);

  void setKeytable(void* keytable);

  size_t encrypt(std::string_view data, size_t index, std::string_view L,
                 Block& check_sum, Block& last_offset, void* result,
                 bool encrypt);

  const GpuTimeSpent* gpuTimeSpent() const;

  bool hasError() const;

 private:
  OcbCamelliaImpl* impl_;
};
}  // namespace cu_ocb
