/**
 * @brief Breadth-first Search Top-Down test program
 * @file
 */

#include <gtest/gtest.h>

#include <Graph/GraphStd.hpp>

#include <StandardAPI.hpp>

#include "../../hornet/tests/hornet_test_fixtures.h"

#include "Static/BreadthFirstSearch/TopDown.cuh"

namespace {

void exec(const char* p_graph_file_path) {
    graph::ParsingProp prop(graph::parsing_prop::PRINT_INFO);
    graph::GraphStd<hornets_nest::vid_t, hornets_nest::eoff_t> graph;
    graph.read(p_graph_file_path, prop);

    hornets_nest::HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());
    hornets_nest::HornetGraph hornet_graph(hornet_init);
    hornets_nest::BfsTopDown bfs_top_down(hornet_graph);

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
protected:
};

TEST_P(BFSTest, BFSTopDownTest) {
    auto p_param = GetParam();
    static_assert(std::is_same<decltype(p_param), const char*>::value, "param should be const char*");
    exec(p_param);
}

INSTANTIATE_TEST_CASE_P(BFSTests, BFSTest, ::testing::Values("../../example/G.mtx", "../../example/rome99.gr"));

