/**
 * @brief Breadth-first Search Top-Down test program
 * @file
 */

#include <gtest/gtest.h>

#include <Graph/GraphStd.hpp>

#include <StandardAPI.hpp>

#include <Util/CommandLineParam.hpp>

#include "Static/BreadthFirstSearch/TopDown.cuh"
#include "Static/BreadthFirstSearch/TopDown2.cuh"

#include "../../hornet/tests/hornet_test_fixtures.h"

namespace {

template<typename TBfsAlg>
void exec(int argc, char* argv[]) {
    graph::GraphStd<hornets_nest::vid_t, hornets_nest::eoff_t> graph;
    hornets_nest::CommandLineParam cmd(graph, argc, argv, false);

    hornets_nest::HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());
    hornets_nest::HornetGraph hornet_graph(hornet_init);
    TBfsAlg bfs_top_down(hornet_graph);

    hornets_nest::vid_t root = graph.max_out_degree_id();
    bfs_top_down.set_parameters(root);

    auto start = std::chrono::high_resolution_clock::now();

    bfs_top_down.run();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "execution time:" << diff.count() * 1000 << " ms" << std::endl;

    ASSERT_TRUE(bfs_top_down.validate());

    return;
}

}

class BFSTest : public HornetTestWithParam<const char*> {
public:
    static int argc;
    static char** argv;

protected:
};

int BFSTest::argc = 0;
char** BFSTest::argv = nullptr;

TEST_F(BFSTest, BFSTopDownTest) {
    exec<hornets_nest::BfsTopDown>(BFSTest::argc, BFSTest::argv);
}

TEST_F(BFSTest, BFSTopDown2Test) {
    exec<hornets_nest::BfsTopDown2>(BFSTest::argc, BFSTest::argv);
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    BFSTest::argc = argc;
    BFSTest::argv = argv;
    return RUN_ALL_TESTS();
}

