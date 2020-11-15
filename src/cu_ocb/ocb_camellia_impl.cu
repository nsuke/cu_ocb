#include "cu_ocb/ocb_camellia_impl.cuh"

#include <iomanip>
#include <iostream>
#include <ostream>

namespace cu_ocb
{
namespace
{
std::ostream& operator<<(std::ostream& os, __uint128_t val)
{
  auto p = reinterpret_cast<uint8_t*>(&val);
  for (size_t i = 0; i < sizeof(val); ++i)
    {
      os << std::setw(2) << std::setfill('0') << std::hex
         << static_cast<uint32_t>(p[i]) << " ";
    }
  return os;
}

struct MeasureGpuTime
{
  MeasureGpuTime(float* result, cudaEvent_t& start, cudaEvent_t& stop)
      : result_{result}, start_{start}, stop_{stop}
  {
    if (result_) cudaEventRecord(start_, 0);
  }

  ~MeasureGpuTime()
  {
    if (result_)
      {
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
        float milliseconds{};
        cudaEventElapsedTime(&milliseconds, start_, stop_);
        *result_ += milliseconds;
      }
  }

 private:
  float* result_;
  cudaEvent_t& start_;
  cudaEvent_t& stop_;
};

constexpr size_t kMaxAddressBit{34};
}  // namespace

OcbCamelliaImpl::OcbCamelliaImpl(OcbConfig config)
    : config_{std::move(config)},
      keytable_{allocateDevice(CAMELLIA_TABLE_BYTE_LEN)},
      L_{allocateDevice(kMaxAddressBit * sizeof(Block))},
      checksum_host_{allocateHost(sizeof(Block))}
{
  if (config_.measure_gpu_time)
    {
      cudaEventCreate(&start_);
      cudaEventCreate(&stop_);
    }
  has_error_ |= checkError(__FILE__, __LINE__);
}

OcbCamelliaImpl::~OcbCamelliaImpl()
{
  if (config_.measure_gpu_time)
    {
      cudaEventDestroy(start_);
      cudaEventDestroy(stop_);
    }
}

void OcbCamelliaImpl::generateKeytable(std::string_view key)
{
  auto cu_key = allocateDevice<uint8_t>(
      sizeof(key), reinterpret_cast<const uint8_t*>(key.data()));
  cudaMemcpy(cu_key, key.data(), key.size(), cudaMemcpyHostToDevice);
  switch (key.size())
    {
      case 128 / 8:
        camellia_setup128<<<1, config_.encrypt_threads>>>(cu_key, keytable_);
        break;
      case 192 / 8:
      case 256 / 8:
        std::cerr << "key length not supported (yet): " << key.size();
        break;
      default:
        std::cerr << "invalid key length: " << key.size();
        break;
    }
}

size_t OcbCamelliaImpl::encrypt(std::string_view data, size_t index,
                                std::string_view L, Block& checksum,
                                Block& last_offset, void* result, bool encrypt)
{
  assert(checkError(__FILE__, __LINE__));

  auto total_elem_count = data.size() / sizeof(Block);
  if (config_.debug)
    std::cerr << "total_elem_count=" << total_elem_count << std::endl;
  if (!total_elem_count) return 0;

  auto block_size = config_.encrypt_threads;
  assert(block_size);
  auto grid_size = config_.process_incomplete_block
                       ? (total_elem_count - 1) / config_.encrypt_threads + 1
                       : total_elem_count / config_.encrypt_threads;
  if (config_.debug) std::cerr << "grid_size=" << grid_size << std::endl;
  if (!grid_size) return 0;
  auto buf_size = grid_size * block_size * sizeof(Block);
  if (config_.debug) std::cerr << "buf_size=" << buf_size << std::endl;
  ensureBuffers(buf_size);
  auto process_size = std::min(buf_size, total_elem_count * sizeof(Block));
  if (config_.debug) std::cerr << "process_size=" << process_size << std::endl;
  auto elem_count = process_size / sizeof(Block);
  if (config_.debug) std::cerr << "elem_count=" << elem_count << std::endl;
  if (!elem_count || elem_count < config_.minimum_blocks) return 0;

  cudaMemcpy(in_buf_, data.data(), process_size, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  if (!checkError(__FILE__, __LINE__)) return 0;

  if (encrypt)
    {
      computeChecksum(checksum, in_buf_,
                      reinterpret_cast<const Block*>(data.data()),
                      process_size);
    }
  cudaDeviceSynchronize();

  auto offsets = fillOffsets(last_offset, index, L, process_size);

  {
    MeasureGpuTime m{config_.measure_gpu_time ? &time_.encryption_ : nullptr,
                     start_, stop_};
    if (encrypt)
      camellia_encrypt128<<<grid_size, block_size>>>(keytable_, offsets,
                                                     in_buf_, out_buf_);
    else
      camellia_decrypt128<<<grid_size, block_size>>>(keytable_, offsets,
                                                     in_buf_, out_buf_);
  }
  if (!checkError(__FILE__, __LINE__)) { throw std::runtime_error("encrypt"); }

  cudaMemcpy(result, out_buf_, process_size, cudaMemcpyDeviceToHost);
  if (!checkError(__FILE__, __LINE__)) return 0;

  if (!encrypt)
    {
      cudaDeviceSynchronize();
      computeChecksum(checksum, out_buf_,
                      reinterpret_cast<const Block*>(result), process_size);
    }

  return elem_count;
}

void OcbCamelliaImpl::applyOffsets(Block& last_offset, std::string_view L,
                                   size_t index, Block* data, size_t size)
{
  assert(size);
  for (size_t i = 0; i < size; ++i)
    {
      auto l_idx = __builtin_ctzll(index + i + 1);
      last_offset ^= L[l_idx];
      data[i] ^= last_offset;
    }
}

u32* OcbCamelliaImpl::fillOffsets(Block& last_offset, size_t index,
                                  std::string_view L, size_t size)
{
  Block last_offset_cpu{last_offset};
  u32* result{};

  if (config_.offset_mode == OffsetComputation::Gpu)
    {
      cudaMemcpy(L_, L.data(), L.size(), cudaMemcpyHostToDevice);
      MeasureGpuTime m{config_.measure_gpu_time ? &time_.ocb_offset_ : nullptr,
                       start_, stop_};
      result = off_.compute(last_offset, index, size / sizeof(Block), L_);
    }

  if (config_.offset_mode == OffsetComputation::ComputeOnCpu ||
      config_.verify_with_cpu_result)
    {
      auto offsets_cpu = config_.offset_mode == OffsetComputation::ComputeOnCpu
                             ? reinterpret_cast<Block*>(offsets_host_.ptr_)
                             : offsets_cpu_buf_.data();
      assert(offsets_cpu);
      auto lptr = reinterpret_cast<const Block*>(L.data());
      for (size_t i = 0; i < size / sizeof(Block); ++i)
        {
          auto l_idx = __builtin_ctzll(index + i + 1);
          offsets_cpu[i] = last_offset_cpu = last_offset_cpu ^ lptr[l_idx];
        }

      if (config_.offset_mode != OffsetComputation::ComputeOnCpu)
        {
          cudaMemcpy(offsets_host_, result, size, cudaMemcpyDeviceToHost);
          cudaDeviceSynchronize();
          auto cu_ptr = reinterpret_cast<Block*>(offsets_host_.ptr_);
          for (size_t i = 0; i < size / sizeof(Block); ++i)
            if (config_.debug && cu_ptr[i] != offsets_cpu[i])
              {
                std::cerr << " gpu_" << i << ": " << cu_ptr[i] << std::endl;
                std::cerr << " cpu_" << i << ": " << offsets_cpu[i]
                          << std::endl;
              }
          if (last_offset != last_offset_cpu)
            {
              std::cerr << " gpu_offset: " << last_offset << std::endl;
              std::cerr << " cpu_offset: " << last_offset_cpu << std::endl;
            }
          auto last_idx = size / sizeof(Block) - 1;
          assert(last_offset == cu_ptr[last_idx]);
          assert(offsets_cpu[last_idx] == last_offset_cpu);
        }
      else
        {
          assert(offsets_cpu_);
          last_offset = last_offset_cpu;
          cudaMemcpy(offsets_cpu_, offsets_cpu, size, cudaMemcpyHostToDevice);
          result = offsets_cpu_;
        }
    }
  return result;
}

void OcbCamelliaImpl::computeChecksum(Block& checksum, const u32* cu_data,
                                      const Block* data, size_t size)
{
  Block checksum_cpu = checksum;
  if (config_.checksum_on_gpu)
    {
      MeasureGpuTime m{config_.measure_gpu_time ? &time_.checksum_ : nullptr,
                       start_, stop_};
      if (!chk_.compute(cu_data, size / sizeof(Block), checksum_host_))
        throw std::runtime_error("checksum");
      checksum ^= *reinterpret_cast<Block*>(checksum_host_.ptr_);
    }
  if (!config_.checksum_on_gpu || config_.verify_with_cpu_result)
    {
      for (size_t i = 0; i < size / sizeof(Block); ++i)
        {
          std::cerr << " data[i] " <<  data[i] << std::endl;
          checksum_cpu ^= data[i];
        }
      if (config_.checksum_on_gpu)
        {
          if (checksum != checksum_cpu)
            {
              std::cerr << "gpu_chk: " << checksum << std::endl;
              std::cerr << "cpu_chk: " << checksum_cpu << std::endl;
            }
        }
      else
        {
          checksum = checksum_cpu;
        }
    }
}

}  // namespace cu_ocb
