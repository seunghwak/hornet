/**
 * @brief Breadth-first Search Top-Down test program
 * @file
 */
#include <iostream>
#include <chrono>

#include <gtest/gtest.h>

#include <Graph/GraphStd.hpp>
#include <Graph/GraphWeight.hpp>

#include <StandardAPI.hpp>

#include <Util/CommandLineParam.hpp>

#include "Static/PageRank/PageRank.cuh"

#include "../../hornet/tests/hornet_test_fixtures.h"

namespace {

void exec(const bool undirected, int argc, char* argv[]) {
    graph::StructureProp structure_flag = graph::structure_prop::NONE;
    if (undirected == true) {
        structure_flag = graph::structure_prop::UNDIRECTED;
    }
    graph::GraphStd<hornets_nest::vid_t, hornets_nest::eoff_t> graph(structure_flag);
    graph.read(argv[1], graph::parsing_prop::PRINT_INFO | graph::parsing_prop::SORT);

    hornets_nest::HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());
    hornets_nest::gpu::Hornet<std::tuple<>, std::tuple<>> hornet_graph(hornet_init);

    hornets_nest::StaticPageRank page_rank(hornet_graph, 50, 0.001, 0.85, undirected);

    auto start = std::chrono::high_resolution_clock::now();

    page_rank.run();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "Computation time: " << diff.count() * 1000/* s to ms */ << " ms" << std::endl;

    return;
}

}

class PageRankTest : public HornetTest {
public:
    static int argc;
    static char** argv;

protected:
};

int PageRankTest::argc = 0;
char** PageRankTest::argv = nullptr;

TEST_F(PageRankTest, PageRankDirectedGraphTest) {
    ASSERT_TRUE(PageRankTest::argc >= 2);
    exec(false, PageRankTest::argc, PageRankTest::argv);
}

TEST_F(PageRankTest, PageRankUndirectedGraphTest) {
    ASSERT_TRUE(PageRankTest::argc >= 2);
    exec(true, PageRankTest::argc, PageRankTest::argv);
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    PageRankTest::argc = argc;
    PageRankTest::argv = argv;
    return RUN_ALL_TESTS();
}

