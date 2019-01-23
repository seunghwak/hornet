#include <gtest/gtest.h>

#include "Host/Numeric.hpp"
#include "Device/Util/DeviceProperties.cuh"
#include "Device/Util/PrintExt.cuh"
#include "Device/Util/Algorithm.cuh"
#include "Device/Primitives/BinarySearchLB.cuh"
#include "Device/Primitives/impl/BinarySearchLB2.i.cuh"
#include "Device/Primitives/MergePathLB.cuh"
#include "Device/Util/Timer.cuh"
//#include <Graph/GraphBase.hpp>
#include <Graph/GraphStd.hpp>
#include <Graph/GraphWeight.hpp>
#include <Graph/BellmanFord.hpp>
#include <Graph/Dijkstra.hpp>

#include <iostream>

#include "Device/Util/Timer.cuh"
#include "Device/DataMovement/impl/Block.i.cuh"
#include <cooperative_groups.h>
#include <random>
#include <chrono>
#include "StandardAPI.hpp"

#include "../hornet_test_fixtures.h"

using namespace graph;
using namespace timer;
using namespace hornets_nest;

template<int ITEMS_PER_BLOCK, int BLOCK_SIZE>
__global__
void MergePathTest2(const int* __restrict__ d_partitions,
                    int                     num_partitions,
                    const int* __restrict__ d_prefixsum,
                    int                     prefixsum_size,
                    int* __restrict__       d_pos,
                    int* __restrict__       d_offset) {
    __shared__ int smem[ITEMS_PER_BLOCK];

    const auto& lambda = [&](int pos, int, int index) {
                             d_pos[index] = pos;
                             //d_offset[index] = offset;
                        };
    xlib::mergePathLB<BLOCK_SIZE, ITEMS_PER_BLOCK>
        (d_partitions, num_partitions, d_prefixsum, prefixsum_size, smem, lambda);
}


__device__ int d_value;

template<int ITEMS_PER_BLOCK, int BLOCK_SIZE>
__global__
void copyKernel(const int* __restrict__ input, int num_blocks, int smem_size) {
    __shared__ int smem[ITEMS_PER_BLOCK];

    for (int i = blockIdx.x; i < num_blocks; i += gridDim.x) {
        xlib::block::StrideOp<0, ITEMS_PER_BLOCK, BLOCK_SIZE>
            ::copy(input + i * ITEMS_PER_BLOCK, smem_size, smem);

        if (threadIdx.x > 1023)
            d_value = smem[threadIdx.x];
    }
}


template<int ITEMS_PER_BLOCK, int BLOCK_SIZE>
__global__
void copyKernel2(const int* __restrict__ input, int num_blocks, int smem_size) {
    for (int i = blockIdx.x; i < num_blocks; i += gridDim.x) {
        auto smem_tmp = xlib::dyn_smem + threadIdx.x;
        auto d_tmp    = input + i * ITEMS_PER_BLOCK + threadIdx.x;

        for (int i = threadIdx.x; i < smem_size; i += BLOCK_SIZE) {
            *smem_tmp = *d_tmp;
            smem_tmp += BLOCK_SIZE;
            d_tmp    += BLOCK_SIZE;
        }

        if (threadIdx.x > 1023)
            d_value = xlib::dyn_smem[threadIdx.x];
    }
}


__global__
void noLambdaKernel(const int* __restrict__ ptr2, int* __restrict__ ptr1, int size) {
    int id     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < size; i += stride) {
        ptr1[i] = ptr2[i];
        ptr1[i + 10] = ptr2[i + 10];
        ptr1[i + 20] = ptr2[i + 20];
    }
}

template<typename Lambda>
__global__
void lambdaKernel(Lambda lambda, int size) {
    int id     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < size; i += stride)
        lambda(i);
}


template<typename Lambda, typename... TArgs>
__global__
void lambdaKernel2(Lambda lambda, int size, TArgs* __restrict__ ... args) {
    int id     = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = id; i < size; i += stride)
        lambda(i, args...);
}

struct LL {
    int*       __restrict__ ptr1;
    const int* __restrict__ ptr2;

    __device__ __forceinline__
    void operator()(int i) {
        const int* __restrict__ vv2 = ptr2;
        int*       __restrict__ vv1 = ptr1;

        vv1[i] = vv2[i];
        vv1[i + 10] = vv2[i + 10];
        vv1[i + 20] = vv2[i + 20];
    }
};

void exec(const std::string& graph_file_path) {
    using namespace graph;

    GraphStd<int, int> graph1;
    graph1.read(graph_file_path.c_str());

    graph1.print_degree_distrib();
    graph1.print_analysis();

    auto weights = new int[graph1.nV()];
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch()
                .count();
    std::mt19937 engine(seed);
    std::uniform_int_distribution<int> distrib(0, 100);
    std::generate(weights, weights + graph1.nV(),
                  [&](){ return distrib(engine); } );

    GraphWeight<int, int, int> graph_weight(graph1.csr_out_edges(), graph1.nV(),
                                            graph1.csr_out_edges(), graph1.nE(),
                                            weights);


    Timer<HOST> TM1;

    Dijkstra<int, int, int> dijkstra(graph_weight);

    TM1.start();

    for (int i = 0; i < graph1.nV(); ++i) {
        dijkstra.run(i);
        dijkstra.reset();
    }
    TM1.stop();
    TM1.print("Dijkstra");

    return;
}

namespace {
    std::vector<std::string> v_command_line_arg;
}

class LoadBalanceTest : public HornetTest {
protected:
};

TEST_F(LoadBalanceTest, LoadBalanceBinarySearchTest) {
    exec(v_command_line_arg);
}

INSTANTIATE_TEST_CASE_P(LoadBalanceTests, LoadBalanceTest, ::testing::Values("../../example/G.mtx", "../../example/rome99.gr"));

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    ASSERT_EQ();
    for (size_t i = 0; i < static_cast<size_t>(argc); ++i) {
        v_command_line_arg.emplace_back(argv[i]);
    }
    return RUN_ALL_TESTS();
}

