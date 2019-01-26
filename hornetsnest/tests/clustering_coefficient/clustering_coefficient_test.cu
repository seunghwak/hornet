/**
 * @brief
 * @file
 */

#include <iostream>
#include <chrono>

#include <gtest/gtest.h>

#include <Graph/GraphStd.hpp>
#include <Graph/GraphWeight.hpp>

#include <StandardAPI.hpp>

#include <Util/CommandLineParam.hpp>

#include "Static/ClusteringCoefficient/cc.cuh"

#include "../../hornet/tests/hornet_test_fixtures.h"

namespace {

void exec(int argc, char* argv[]) {
    graph::GraphStd<hornets_nest::vid_t, hornets_nest::eoff_t> graph(graph::structure_prop::UNDIRECTED);
    graph.read(argv[1], graph::parsing_prop::SORT | graph::parsing_prop::PRINT_INFO);

    hornets_nest::HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());
    hornets_nest::HornetGraph hornet_graph(hornet_init);

    hornets_nest::ClusteringCoefficient cc(hornet_graph);

    cc.init();

    auto start = std::chrono::high_resolution_clock::now();

    cc.run();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "Computation time: " << diff.count() * 1000/* s to ms */ << " ms" << std::endl;
  
    return;
}

}

class ClusteringCoefficientTest : public HornetTest {
public:
    static int argc;
    static char** argv;

protected:
};

int ClusteringCoefficientTest::argc = 0;
char** ClusteringCoefficientTest::argv = nullptr;

TEST_F(ClusteringCoefficientTest, ClusteringCoefficientTest) {
    ASSERT_TRUE(ClusteringCoefficientTest::argc >= 2);
    exec(ClusteringCoefficientTest::argc, ClusteringCoefficientTest::argv);
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    ClusteringCoefficientTest::argc = argc;
    ClusteringCoefficientTest::argv = argv;
    return RUN_ALL_TESTS();
}

