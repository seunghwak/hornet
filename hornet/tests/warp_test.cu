#include <cuda.h>

#include <cub/cub.cuh>
#include <gtest/gtest.h>

#include "StandardAPI.hpp"

#include "hornet_test_fixtures.h"

namespace {

__global__ void WarpScanTest() {
    int value = 1;

    __shared__ cub::WarpScan<decltype(value)>::TempStorage temp_storage;

    cub::WarpScan<decltype(value)>(temp_storage).ExclusiveSum(value, value);
}

__global__ void WarpReduceTest() {
    int value = 1;

    __shared__ cub::WarpReduce<decltype(value)>::TempStorage temp_storage;

    auto aggregate = cub::WarpReduce<decltype(value)>(temp_storage).Sum(value);
    (void)aggregate;
}

}

class WarpTest : public HornetTest {
protected:
};

TEST_F(WarpTest, WarpScanTest) {
    int num_devices = 0;

    ASSERT_EQ(cudaGetDeviceCount(&num_devices), cudaSuccess);
    std::cout << "# GPUs=" << num_devices << std::endl;

    for (size_t i = 0; i < static_cast<size_t>(num_devices); ++i) {
        int warp_size = 0;
        ASSERT_EQ(cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, i), cudaSuccess);
        ASSERT_EQ(cudaSetDevice(i), cudaSuccess);

        std::cout << "run test on GPU " << i << " warp size=" << warp_size << std::endl;
        WarpScanTest<<<1, warp_size>>>();
    }
}

TEST_F(WarpTest, WarpReduceTest) {
    int num_devices = 0;

    ASSERT_EQ(cudaGetDeviceCount(&num_devices), cudaSuccess);
    std::cout << "# GPUs=" << num_devices << std::endl;

    for (size_t i = 0; i < static_cast<size_t>(num_devices); ++i) {
        int warp_size = 0;
        ASSERT_EQ(cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, i), cudaSuccess);
        ASSERT_EQ(cudaSetDevice(i), cudaSuccess);

        std::cout << "run test on GPU " << i << " warp size=" << warp_size << std::endl;
        WarpReduceTest<<<1, warp_size>>>();
    }
}

