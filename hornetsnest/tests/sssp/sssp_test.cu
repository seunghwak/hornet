/**
 * @brief SSSP test program
 * @file
 */

#include <iostream>
#include <chrono>

#include <gtest/gtest.h>

#include <Graph/GraphStd.hpp>
#include <Graph/GraphWeight.hpp>

#include <StandardAPI.hpp>

#include <Util/CommandLineParam.hpp>

#include "Static/ShortestPath/SSSP.cuh"

#include "../../hornet/tests/hornet_test_fixtures.h"

namespace {

void exec(int argc, char* argv[]) {
    graph::GraphStd<hornets_nest::vid_t, hornets_nest::eoff_t> graph;
    hornets_nest::CommandLineParam cmd(graph, argc, argv,false);

    std::unique_ptr<hornets_nest::weight_t[]> p_weights(new hornets_nest::weight_t[graph.nE()]);

    hornets_nest::host::generate_randoms(p_weights.get(), graph.nE(), 0, 100);

    hornets_nest::HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());
    hornet_init.insertEdgeData(p_weights.get());
    hornets_nest::HornetGraph hornet_graph(hornet_init);

    hornets_nest::SSSP sssp(hornet_graph);

    hornets_nest::vid_t root = 0;
    sssp.set_parameters(root);

    auto start = std::chrono::high_resolution_clock::now();

    sssp.run();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "Computation time: " << diff.count() * 1000/* s to ms */ << " ms" << std::endl;

    ASSERT_TRUE(sssp.validate());

    return;
}

}

class SSSPTest : public HornetTest {
public:
    static int argc;
    static char** argv;

protected:
};

int SSSPTest::argc = 0;
char** SSSPTest::argv = nullptr;

TEST_F(SSSPTest, SSSPTest) {
    ASSERT_TRUE(SSSPTest::argc >= 2);
    exec(SSSPTest::argc, SSSPTest::argv);
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    SSSPTest::argc = argc;
    SSSPTest::argv = argv;
    return RUN_ALL_TESTS();
}

