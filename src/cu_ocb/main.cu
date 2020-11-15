#include "cu_ocb/constants.h"
#include "cu_ocb/cuda_mem.cuh"
#include "cu_ocb/cuda_utils.cuh"
#include "cu_ocb/ocb_camellia.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>

#define OCB_CAMELLIA_VERIFY_WITH_CPU 0

namespace cu_ocb
{
namespace
{
#if OCB_CAMELLIA_VERIFY_WITH_CPU

#endif

constexpr size_t L_size = 34;

template <bool encrypt>
std::string run_encrypt(size_t buffer_size, OcbConfig config,
                        std::string_view data_size, __uint128_t& offset,
                        std::string_view L)
{
  OcbCamellia enc{std::move(config)};

  constexpr uint8_t key[] = {
      0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
      0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10,
  };
  enc.generateKeytable({reinterpret_cast<const char*>(key), sizeof(key)});

  std::string src{"/tmp/"};
  src += data_size;
  src += ".blob";
  auto dst = src + (encrypt ? ".gpg" : ".decrypted");
  if (!encrypt) src += ".gpg";

  std::cout << "source file: " << src << std::endl;
  std::cout << "destination file: " << dst << std::endl;
  std::cout << "buffer size: " << buffer_size << std::endl;

  constexpr auto test_reference_input = true;
  if (test_reference_input && encrypt) {
    std::ofstream ofs{src};
    ofs.write(reinterpret_cast<const char*>(key), 16);
  }

  std::ifstream ifs;
  ifs.exceptions(std::ios::failbit | std::ios::badbit);
  ifs.open(src);
  ifs.exceptions(std::ios::badbit);

  std::ofstream ofs;
  ofs.exceptions(std::ios::failbit | std::ios::badbit);
  ofs.open(dst);

  auto in_buffer = allocateHost<char>(buffer_size);
  auto out_buffer = allocateHost<char>(buffer_size);

  __uint128_t checksum{0};
  size_t pos = 0;
  while (true)
    {
      assert(checkError(__FILE__, __LINE__));
      ifs.read(in_buffer, buffer_size);
      auto byte_count = ifs.gcount();
      if (byte_count < CAMELLIA_BLOCK_SIZE) break;
      auto elem_count = byte_count / CAMELLIA_BLOCK_SIZE;

      assert(L.size() == 544);
      enc.encrypt({in_buffer, static_cast<size_t>(byte_count)}, pos, L,
                  checksum, offset, out_buffer, encrypt);

      pos += elem_count;
      ofs.write(out_buffer, elem_count * CAMELLIA_BLOCK_SIZE);
    }

  if (auto* time = enc.gpuTimeSpent())
    {
      std::cout << " *** encryption: " << time->encryption_ << " ms"
                << std::endl;
      std::cout << " *** offset    : " << time->ocb_offset_ << " ms"
                << std::endl;
      std::cout << " *** checksum  : " << time->checksum_ << " ms" << std::endl;
    }
  return {reinterpret_cast<char*>(&checksum), CAMELLIA_BLOCK_SIZE};
}

std::ostream& print_hex(std::ostream& os, const std::string& v)
{
  for (auto c : v)
    {
      os << " " << std::hex << std::setw(2) << std::setfill('0')
         << static_cast<int>(static_cast<uint8_t>(c));
    }
  return os;
}

}  // namespace
}  // namespace cu_ocb

using namespace cu_ocb;
using namespace std::literals;

int main(int argc, char* argv[])
{
  constexpr __uint128_t nonce{0x0};
  std::array<__uint128_t, L_size> L;
  std::fill(L.begin(), L.end(), 1);
  for (size_t i = 0; i < L.size(); ++i) { L[i] <<= i; }
  std::fill(L.begin(), L.end(), 0);
  std::string_view L_view{reinterpret_cast<const char*>(L.data()),
                          L.size() * sizeof(__uint128_t)};

  auto data_size = "1G"s;
  auto offset = nonce;

  OcbConfig config{};
  size_t buffer_size = 8 * 1024 * 1024;
  for (int i = 1; i + 1 < argc; ++i)
    {
      std::string arg{argv[i]};
      if (arg == "--data-size" || arg == "-D")
        data_size = argv[i + 1];
      else if (arg == "--buffer-size" || arg == "-b")
        {
          std::string arg2{argv[i + 1]};
          int unit = 1;
          switch (arg2.back())
            {
              case 'k':
              case 'K':
                unit = 1024;
                break;
              case 'm':
              case 'M':
                unit = 1024 * 1024;
                break;
              case 'g':
              case 'G':
                unit = 1024 * 1024 * 1024;
                break;
            }
          if (int n = std::stoi(arg2)) buffer_size = n * unit;
        }
      else if (arg == "--encrypt-threads" || arg == "-e")
        {
          if (int n = std::atoi(argv[i + 1])) config.encrypt_threads = n;
        }
      else if (arg == "--offset-threads" || arg == "-o")
        {
          if (int n = std::atoi(argv[i + 1])) config.offset_threads = n;
        }
      else if (arg == "--checksum-threads" || arg == "-c")
        {
          if (int n = std::atoi(argv[i + 1])) config.checksum_threads = n;
        }
      else if (arg == "--offset-mode" || arg == "-O")
        {
          if (argv[i + 1] == "cpu"s)
            {
              std::cout << "computing offset on cpu" << std::endl;
              config.offset_mode = OffsetComputation::ComputeOnCpu;
            }
        }
      else if (arg == "--checksum-mode" || arg == "-C")
        if (argv[i + 1] == "cpu"s) config.checksum_on_gpu = false;
    }
  for (int i = 1; i < argc; ++i)
    {
      std::string arg{argv[i]};
      if (arg == "--verify" || arg == "-v")
        config.verify_with_cpu_result = true;
      else if (arg == "--debug" || arg == "-d")
        config.debug = true;
      else if (arg == "--time" || arg == "-t")
        config.measure_gpu_time = true;
    }
  print_hex(std::cout << "enc checksum:",
            run_encrypt<true>(buffer_size, config, data_size, offset, L_view))
      << std::endl;
  offset = nonce;
  print_hex(std::cout << "dec checksum:",
            run_encrypt<false>(buffer_size, config, data_size, offset, L_view))
      << std::endl;
}
