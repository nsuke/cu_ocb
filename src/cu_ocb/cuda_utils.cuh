#pragma once

#include <iostream>

inline bool checkError(const char* file, int line)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
    {
      std::cerr << "CUDA ERROR - " << file << ':' << line << ": ["
                << static_cast<int>(err) << "] " << cudaGetErrorString(err)
                << std::endl;
      return false;
    }
  return true;
}

