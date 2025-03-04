#pragma once

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cuda.h>

#include <iostream>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <memory>
#include <string>
#include <limits>

#include <chrono>

namespace fCUDA
{
    void getDeviceName();
    void getDeviceInfo(size_t deviceID);
    /*
    * Please note that this function is only used if you have multiple GPU in your system
    */
    void setGPUDirect();

    template <int div>
    __global__ void dGPUDiv(unsigned int size, const unsigned char* __restrict__ input, unsigned char* __restrict__ output);
    template <int div>
    __global__ void dGPUDiv_opt(const unsigned char* __restrict__ input, unsigned char* __restrict__ output);

    __global__ void dGPUTestKernel(unsigned int* data);
}

namespace fTEST
{
    void fillArray(unsigned int* data, const unsigned int size);
    void checkArray(char* dPrefix, unsigned int* data, const unsigned int size);
    double measure_allocation_time(unsigned int max_allocations);
    void p2p_alloc_bench();
}
