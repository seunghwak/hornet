/**
 * @brief Sparse Matrix-Vector multiplication
 * @file
 */

#include <iostream>
#include <chrono>

#include <gtest/gtest.h>

#include <Graph/GraphStd.hpp>

#include <StandardAPI.hpp>

#include <Util/CommandLineParam.hpp>

#include "Static/SpMV/SpMV.cuh"

#include "../../hornet/tests/hornet_test_fixtures.h"

namespace {

void exec(int argc, char* argv[]) {
    graph::GraphStd<hornets_nest::vid_t, hornets_nest::eoff_t> graph;
    hornets_nest::CommandLineParam cmd(graph, argc, argv);

    std::unique_ptr<int[]> p_vector(new int[graph.nV()]);
    std::unique_ptr<int[]> p_value(new int[graph.nE()]);
    std::fill(p_vector.get(), p_vector.get() + graph.nV(), 1);
    std::fill(p_value.get(), p_value.get() + graph.nE(), 1);

    hornets_nest::HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());
    hornet_init.insertEdgeData(p_value.get());
    hornets_nest::HornetGraph hornet_matrix(hornet_init);

    hornets_nest::SpMV spmv(hornet_matrix, p_vector.get());

    auto start = std::chrono::high_resolution_clock::now();

    spmv.run();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "Computation time: " << diff.count() * 1000/* s to ms */ << " ms" << std::endl;

    ASSERT_TRUE(spmv.validate());

    return;
}

}

class SPMVTest : public HornetTest {
public:
    static int argc;
    static char** argv;

protected:
};

int SPMVTest::argc = 0;
char** SPMVTest::argv = nullptr;

TEST_F(SPMVTest, SPMVTest) {
    ASSERT_TRUE(SPMVTest::argc >= 2);
    exec(SPMVTest::argc, SPMVTest::argv);
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    SPMVTest::argc = argc;
    SPMVTest::argv = argv;
    return RUN_ALL_TESTS();
}


