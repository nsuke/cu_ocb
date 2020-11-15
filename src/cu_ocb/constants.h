#pragma once

#include <cstdint>
#include <cstddef>

namespace cu_ocb {

constexpr int CAMELLIA_BLOCK_SIZE{16};
constexpr int CAMELLIA_TABLE_BYTE_LEN{272};

using u32 = uint32_t;

// TODO: cleanup
constexpr size_t kU32ElemSize{CAMELLIA_BLOCK_SIZE / sizeof(uint32_t)};
constexpr size_t kInnerLoopSize = 1;

}
