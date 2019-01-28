/**
 * @brief Breadth-first Search Top-Down test program
 * @file
 */

#include <iostream>
#include <chrono>

#include <gtest/gtest.h>

#include <Graph/GraphStd.hpp>

#include <StandardAPI.hpp>

#include <Util/CommandLineParam.hpp>

#include "Static/BreadthFirstSearch/bfs_top_down.cuh"

#include "../../hornet/tests/hornet_test_fixtures.h"

namespace {

void exec(int argc, char* argv[]) {
    graph::GraphStd<hornets_nest::vid_t, hornets_nest::eoff_t> graph;
    hornets_nest::CommandLineParam cmd(graph, argc, argv, false);

    hornets_nest::HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());
    hornets_nest::HornetGraph hornet_graph(hornet_init);

    hornets_nest::BfsTopDown bfs_top_down(hornet_graph);

    auto root = graph.max_out_degree_id();
    bfs_top_down.set_parameters(root);

    auto start = std::chrono::high_resolution_clock::now();

    bfs_top_down.run();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "Computation time: " << diff.count() * 1000/* s to ms */ << " ms" << std::endl;

    ASSERT_TRUE(bfs_top_down.validate());

    return;
}

}

class BfsTest : public HornetTest {
public:
    static int argc;
    static char** argv;

protected:
};

int BfsTest::argc = 0;
char** BfsTest::argv = nullptr;

TEST_F(BfsTest, BfsTopDownTest) {
    ASSERT_TRUE(BfsTest::argc >= 2);
    exec(BfsTest::argc, BfsTest::argv);
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    BfsTest::argc = argc;
    BfsTest::argv = argv;
    return RUN_ALL_TESTS();
}

