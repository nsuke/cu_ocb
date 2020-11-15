#include "cu_ocb/checksum.cuh"
#include "cu_ocb/constants.h"
#include "cu_ocb/cuda_mem.cuh"
#include "cu_ocb/cuda_utils.cuh"
#include "cu_ocb/offset.cuh"

#include <array>
#include <bitset>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

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

void test_cuda_mem_move()
{
  std::cout << "---- " << __func__ << std::endl;
  std::vector<uintptr_t> deleted;
  auto del = [&](void* ptr) -> cudaError_t {
    deleted.push_back(reinterpret_cast<uintptr_t>(ptr));
    return cudaSuccess;
  };
  uintptr_t dummy = 0x42;
  CudaMem<char> mem1{reinterpret_cast<char*>(dummy), del};
  assert(deleted.empty());

  {
    CudaMem<char> moved{std::move(mem1)};
    assert(deleted.empty());
  }

  assert(!deleted.empty());
  assert(deleted[0] == dummy);
}

void test_cuda_mem_reset()
{
  std::cout << "---- " << __func__ << std::endl;
  std::vector<uintptr_t> deleted;
  auto del = [&](void* ptr) -> cudaError_t {
    deleted.push_back(reinterpret_cast<uintptr_t>(ptr));
    return cudaSuccess;
  };
  uintptr_t dummy = 0x42;
  CudaMem<char> mem1{reinterpret_cast<char*>(dummy), del};
  assert(deleted.empty());

  mem1.reset();

  assert(!deleted.empty());
  assert(deleted[0] == dummy);
}

void test_cuda_mem()
{
  test_cuda_mem_move();
  test_cuda_mem_reset();
}

void test_reduce_128xor()
{
  std::cout << "---- " << __func__ << std::endl;

  constexpr size_t buf_count = 8;
  constexpr size_t num_elems = 7;

  auto src = allocateManaged(buf_count * CAMELLIA_BLOCK_SIZE);
  std::fill_n(src.ptr_, buf_count * kU32ElemSize, 1);
  for (size_t i = 0; i < buf_count * kU32ElemSize; ++i) src.ptr_[i] <<= i;

  constexpr size_t grid_size = 2;
  auto dst = allocateManaged(grid_size * CAMELLIA_BLOCK_SIZE);
  auto block_size = buf_count / grid_size / 2;
  reduce_128xor<<<grid_size, block_size, block_size * CAMELLIA_BLOCK_SIZE>>>(
      src, num_elems, dst);
  cudaDeviceSynchronize();
  assert(checkError(__FILE__, __LINE__));

  u32 expected[]{
      0b00000000000000000001000100010001, 0b00000000000000000010001000100010,
      0b00000000000000000100010001000100, 0b00000000000000001000100010001000,
      0b00000001000100010000000000000000, 0b00000010001000100000000000000000,
      0b00000100010001000000000000000000, 0b00001000100010000000000000000000,
  };
  for (size_t i = 0; i < grid_size * kU32ElemSize; ++i)
    {
      if (auto print = false)
        std::cout << i << ':' << std::bitset<32>{dst[i]} << std::endl;
      assert(expected[i] == dst[i]);
    }
}

template <size_t BlockSize>
void test_checksum_compute()
{
  std::cout << "---- " << __func__ << std::endl;
  Checksum128Calc calc{BlockSize};

  constexpr auto num_elems = 11;
  auto cu_data = allocateManaged(CAMELLIA_BLOCK_SIZE * num_elems);
  auto result = allocateHost(CAMELLIA_BLOCK_SIZE);
  std::fill_n(cu_data.ptr_, num_elems * kU32ElemSize, 1);
  for (size_t i = 0; i < num_elems * kU32ElemSize; ++i)
    {
      cu_data[i] <<= (i % 32);
      // std::cout << cu_data[i] << std::endl;
    }

  cudaDeviceSynchronize();
  assert(calc.compute(cu_data, num_elems, result));
  cudaDeviceSynchronize();
  assert(checkError(__FILE__, __LINE__));

  std::array<u32, num_elems> expected{
      0b00010001000100010001000000000000,
      0b00100010001000100010000000000000,
      0b01000100010001000100000000000000,
      0b10001000100010001000000000000000,
  };

  for (size_t i = 0; i < kU32ElemSize; ++i)
    {
      if (auto print = false)
        std::cout << std::bitset<32>{result[i]} << std::endl;
      assert(expected[i] == result[i]);
    }
}

void test_checksum()
{
  test_reduce_128xor();
  test_checksum_compute<1>();
  test_checksum_compute<2>();
  test_checksum_compute<4>();
  test_checksum_compute<8>();
  test_checksum_compute<16>();
  test_checksum_compute<256>();
}

template <size_t GridSize>
void test_fill_L()
{
  std::cout << "---- " << __func__ << std::endl;
  constexpr size_t num_elems = 8;
  static_assert(GridSize <= num_elems);
  std::array<__uint128_t, 34> L{
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  };
  for (int i = 0; i < 34; ++i) { L[i] <<= i; }

  auto cu_L = allocateDevice(CAMELLIA_BLOCK_SIZE * L.size());
  cudaMemcpy(cu_L, L.data(), CAMELLIA_BLOCK_SIZE * L.size(),
             cudaMemcpyHostToDevice);
  assert(checkError(__FILE__, __LINE__));
  auto dst = allocateManaged(CAMELLIA_BLOCK_SIZE * 32);

  auto block_size = num_elems / GridSize;
  fill_L<<<GridSize, block_size, block_size * CAMELLIA_BLOCK_SIZE>>>(3, cu_L,
                                                                     dst);
  cudaDeviceSynchronize();
  assert(checkError(__FILE__, __LINE__));

  auto result_ptr = reinterpret_cast<__uint128_t*>(dst.ptr_);
  std::array<__uint128_t, 8> expected{
      4, 1, 2, 1, 8, 1, 2, 1,
  };
  for (size_t i = 0; i < expected.size(); ++i)
    {
      // std::cout << i << ": " << result_ptr[i] << std::endl;
      assert(expected[i] == result_ptr[i]);
    }
}

template <size_t GridSize>
void test_fill_L_reduction()
{
  std::cout << "---- " << __func__ << std::endl;
  constexpr size_t num_elems = 8;
  static_assert(GridSize <= num_elems);
  std::array<__uint128_t, 34> L{
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
  };
  for (int i = 0; i < 34; ++i) { L[i] <<= i; }
  auto cu_L = allocateDevice(CAMELLIA_BLOCK_SIZE * L.size());
  cudaMemcpy(cu_L, L.data(), CAMELLIA_BLOCK_SIZE * L.size(),
             cudaMemcpyHostToDevice);
  assert(checkError(__FILE__, __LINE__));
  auto blk = allocateManaged(CAMELLIA_BLOCK_SIZE * 32ULL);
  auto base = allocateManaged(CAMELLIA_BLOCK_SIZE * 32ULL);

  auto block_size = num_elems / GridSize;
  fill_L_reduce<<<GridSize, block_size, block_size * CAMELLIA_BLOCK_SIZE>>>(
      3, cu_L, blk, base);
  cudaDeviceSynchronize();
  assert(checkError(__FILE__, __LINE__));
  auto blk_ptr = reinterpret_cast<__uint128_t*>(blk.ptr_);
  auto base_ptr = reinterpret_cast<__uint128_t*>(base.ptr_);

  if (auto print = false)
    {
      for (size_t i = 0; i < num_elems; ++i)
        std::cout << "blk_" << i << ": " << blk_ptr[i] << std::endl;
      for (size_t i = 0; i <= GridSize; ++i)
        std::cout << "base_" << i << ": " << base_ptr[i] << std::endl;
    }
  if constexpr (GridSize == 1)
    {
      assert(blk_ptr[0] == 0x04);
      assert(blk_ptr[1] == 0x05);
      assert(blk_ptr[2] == 0x07);
      assert(blk_ptr[3] == 0x06);
      assert(blk_ptr[4] == 0x0e);
      assert(blk_ptr[5] == 0x0f);
      assert(blk_ptr[6] == 0x0d);
      assert(blk_ptr[7] == 0x0c);
      assert(base_ptr[1] == 0x0c);
    }
  else if constexpr (GridSize == 2)
    {
      assert(blk_ptr[0] == 0x04);
      assert(blk_ptr[1] == 0x05);
      assert(blk_ptr[2] == 0x07);
      assert(blk_ptr[3] == 0x06);
      assert(blk_ptr[4] == 0x08);
      assert(blk_ptr[5] == 0x09);
      assert(blk_ptr[6] == 0x0b);
      assert(blk_ptr[7] == 0x0a);
      assert(base_ptr[1] == 0x06);
      assert(base_ptr[2] == 0x0a);
    }
  else if constexpr (GridSize == 4)
    {
      assert(blk_ptr[0] == 0x04);
      assert(blk_ptr[1] == 0x05);
      assert(blk_ptr[2] == 0x02);
      assert(blk_ptr[3] == 0x03);
      assert(blk_ptr[4] == 0x08);
      assert(blk_ptr[5] == 0x09);
      assert(blk_ptr[6] == 0x02);
      assert(blk_ptr[7] == 0x03);
      assert(base_ptr[1] == 0x05);
      assert(base_ptr[2] == 0x03);
      assert(base_ptr[3] == 0x09);
      assert(base_ptr[4] == 0x03);
    }
  else if constexpr (GridSize == 8)
    {
      assert(blk_ptr[0] == 0x04);
      assert(blk_ptr[1] == 0x01);
      assert(blk_ptr[2] == 0x02);
      assert(blk_ptr[3] == 0x01);
      assert(blk_ptr[4] == 0x08);
      assert(blk_ptr[5] == 0x01);
      assert(blk_ptr[6] == 0x02);
      assert(blk_ptr[7] == 0x01);
      assert(base_ptr[1] == 0x04);
      assert(base_ptr[2] == 0x01);
      assert(base_ptr[3] == 0x02);
      assert(base_ptr[4] == 0x01);
      assert(base_ptr[5] == 0x08);
      assert(base_ptr[6] == 0x01);
      assert(base_ptr[7] == 0x02);
      assert(base_ptr[8] == 0x01);
    }
}

void test_apply_128xor()
{
  std::cout << "---- test_apply_128xor" << std::endl;
  constexpr auto grid_size = 3;
  constexpr auto block_size = 4;
  constexpr auto data_size = grid_size * block_size;
  constexpr auto data_len = data_size * CAMELLIA_BLOCK_SIZE;
  std::array<__int128, grid_size> base_data{
      0x1000,
      0x10000,
      0x100,
  };
  std::array<__int128, data_size> blk_data{
      0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0x10, 0x11, 0x12,
  };
  auto base =
      allocateDevice(grid_size * CAMELLIA_BLOCK_SIZE, (u32*)base_data.data());
  auto blk = allocateDevice(data_len, (u32*)blk_data.data());

  apply_128xor<<<grid_size, block_size>>>(base, blk);
  cudaDeviceSynchronize();
  assert(checkError(__FILE__, __LINE__));
  auto result = allocateHost(data_size * CAMELLIA_BLOCK_SIZE);
  cudaMemcpy(result, blk, data_len, cudaMemcpyDeviceToHost);
  assert(checkError(__FILE__, __LINE__));
  auto result_ptr = reinterpret_cast<__uint128_t*>(result.ptr_);

  std::array<__uint128_t, data_size> expected{
      0x1001,  0x1002,  0x1003, 0x1004, 0x10005, 0x10006,
      0x10007, 0x10008, 0x109,  0x110,  0x111,   0x112,
  };
  for (size_t i = 0; i < blk_data.size(); ++i)
    {
      if (auto print = false)
        {
          std::cout << i << ": " << result_ptr[i] << std::endl;
          std::cout << i << ": " << expected[i] << std::endl;
        }
      assert(expected[i] == result_ptr[i]);
    }
}

void test_block_integral_128xor()
{
  std::cout << "---- " << __func__ << std::endl;
  constexpr auto grid_size = 3;
  constexpr auto block_size = 4;
  constexpr auto data_size = grid_size * block_size;
  constexpr auto byte_len = data_size * CAMELLIA_BLOCK_SIZE;

  auto data = allocateManaged(byte_len);
  auto data_ptr = reinterpret_cast<__uint128_t*>(data.ptr_);
  data_ptr[0] = 0b000000000001;
  data_ptr[1] = 0b000000000010;
  data_ptr[2] = 0b000000000100;
  data_ptr[3] = 0b000000001000;
  data_ptr[4] = 0b000000010000;
  data_ptr[5] = 0b000000100000;
  data_ptr[6] = 0b000001000000;
  data_ptr[7] = 0b000010000000;
  data_ptr[8] = 0b000100000000;
  data_ptr[9] = 0b001000000000;
  data_ptr[10] = 0b010000000000;
  data_ptr[11] = 0b100000000000;
  auto base = allocateManaged((grid_size + 1) * CAMELLIA_BLOCK_SIZE);

  block_integral_128xor<<<grid_size, block_size,
                          block_size * CAMELLIA_BLOCK_SIZE>>>(data, base);
  cudaDeviceSynchronize();
  assert(checkError(__FILE__, __LINE__));
  auto base_ptr = reinterpret_cast<__uint128_t*>(base.ptr_);

  if (auto print = false)
    {
      for (size_t i = 0; i < data_size; ++i)
        std::cout << "d_" << i << ": " << data_ptr[i] << std::endl;
      for (size_t i = 0; i <= grid_size; ++i)
        std::cout << "b_" << i << ": " << base_ptr[i] << std::endl;
    }

  assert(data_ptr[0] == 0b000000000001);
  assert(data_ptr[1] == 0b000000000011);
  assert(data_ptr[2] == 0b000000000111);
  assert(data_ptr[3] == 0b000000001111);
  assert(data_ptr[4] == 0b000000010000);
  assert(data_ptr[5] == 0b000000110000);
  assert(data_ptr[6] == 0b000001110000);
  assert(data_ptr[7] == 0b000011110000);
  assert(data_ptr[8] == 0b000100000000);
  assert(data_ptr[9] == 0b001100000000);
  assert(data_ptr[10] == 0b011100000000);
  assert(data_ptr[11] == 0b111100000000);

  assert(data_ptr[3] == base_ptr[1]);
  assert(data_ptr[7] == base_ptr[2]);
  assert(data_ptr[11] == base_ptr[3]);
}

void test_ocb_offset_size_1()
{
  std::cout << "---- " << __func__ << std::endl;
  OcbOffsetCalc calc{4};
  {
    std::array<__uint128_t, 34> L{
        1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    };
    __uint128_t last_offset{};
    auto cu_L = allocateDevice(L.size() * CAMELLIA_BLOCK_SIZE,
                               reinterpret_cast<const u32*>(L.data()));
    auto offsets = calc.compute(last_offset, 7, 1, cu_L);
    cudaDeviceSynchronize();
    assert(checkError(__FILE__, __LINE__));
    assert(offsets);

    auto result = allocateHost(CAMELLIA_BLOCK_SIZE);
    cudaMemcpy(result, offsets, CAMELLIA_BLOCK_SIZE, cudaMemcpyDeviceToHost);
    auto result_ptr = reinterpret_cast<__uint128_t*>(result.ptr_);

    if (auto print = false)
      {
        std::cout << "last: " << last_offset << std::endl;
        std::cout << "0: " << result_ptr[0] << std::endl;
      }

    assert(result_ptr[0] == last_offset);
    assert(result_ptr[0] == L[3]);
  }
}

template <size_t BlockSize, size_t ElemCount>
void test_ocb_offset_size_n()
{
  std::cout << "---- " << __func__ << std::endl;
  OcbOffsetCalc calc{BlockSize};
  {
    std::array<__uint128_t, 34> L;
    std::fill(L.begin(), L.end(), 1);
    for (int i = 0; i < 34; ++i) { L[i] <<= i; }

    __uint128_t last_offset{0b1000};
    // auto last_offset_orig = last_offset;
    auto cu_L = allocateDevice(L.size() * CAMELLIA_BLOCK_SIZE,
                               reinterpret_cast<const u32*>(L.data()));
    auto offsets = calc.compute(last_offset, 0, ElemCount, cu_L);
    cudaDeviceSynchronize();
    assert(checkError(__FILE__, __LINE__));
    assert(offsets);

    auto result = allocateHost(CAMELLIA_BLOCK_SIZE * ElemCount);
    cudaMemcpy(result, offsets, CAMELLIA_BLOCK_SIZE * ElemCount,
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    auto result_ptr = reinterpret_cast<__uint128_t*>(result.ptr_);

    if (auto print = false)
      {
        std::cout << "last: " << last_offset << std::endl;
        std::cout << "result: " << std::endl;
      }

    std::array<__uint128_t, 32> expected{
        0b0000000000001001, 0b0000000000001011, 0b0000000000001010,
        0b0000000000001110, 0b0000000000001111, 0b0000000000001101,
        0b0000000000001100, 0b0000000000000100, 0b0000000000000101,
        0b0000000000000111, 0b0000000000000110, 0b0000000000000010,
        0b0000000000000011, 0b0000000000000001, 0b0000000000000000,
        0b0000000000010000, 0b0000000000010001, 0b0000000000010011,
        0b0000000000010010, 0b0000000000010110, 0b0000000000010111,
        0b0000000000010101, 0b0000000000010100, 0b0000000000011100,
        0b0000000000011101, 0b0000000000011111, 0b0000000000011110,
        0b0000000000011010, 0b0000000000011011, 0b0000000000011001,
        0b0000000000011000, 0b0000000000111000,
    };

    assert(checkError(__FILE__, __LINE__));
    assert(result_ptr[ElemCount - 1] == last_offset);
    for (size_t i = 0; i < min(ElemCount, expected.size()); ++i)
      {
        if (auto print = false)
          std::cout << std::bitset<128>(result_ptr[i]) << std::endl;
        assert(result_ptr[i] == expected[i]);
      }
  }
}

void test_ocb_offset()
{
  test_fill_L<1>();
  test_fill_L<2>();
  test_fill_L<4>();
  test_fill_L<8>();
  test_fill_L_reduction<1>();
  test_fill_L_reduction<2>();
  test_fill_L_reduction<4>();
  test_fill_L_reduction<8>();
  test_apply_128xor();
  test_block_integral_128xor();
  test_ocb_offset_size_1();
  test_ocb_offset_size_n<8, 5>();
  test_ocb_offset_size_n<4, 5>();
  test_ocb_offset_size_n<16, 15>();
  test_ocb_offset_size_n<8, 15>();
  test_ocb_offset_size_n<4, 15>();
  test_ocb_offset_size_n<32, 25>();
  test_ocb_offset_size_n<16, 25>();
  test_ocb_offset_size_n<8, 25>();
  test_ocb_offset_size_n<4, 25>();
  test_ocb_offset_size_n<32, 32>();
  test_ocb_offset_size_n<4, 256>();
  test_ocb_offset_size_n<32, 256>();
  test_ocb_offset_size_n<128, 256>();
  test_ocb_offset_size_n<256, 256>();
}

}  // namespace
}  // namespace cu_ocb

using namespace cu_ocb;

int main()
{
  test_cuda_mem();
  test_checksum();
  test_ocb_offset();
}
