#include <chrono>
#include <iostream>
#include <random>

#include <gtest/gtest.h>

#include <Graph/Dijkstra.hpp>
#include <Graph/GraphStd.hpp>
#include <Graph/GraphWeight.hpp>

#include "../hornet_test_fixtures.h"

namespace {

void exec(int argc, char* argv[]) {
    graph::GraphStd<int, int> graph;

    graph.read(argv[1]);
    graph.print_degree_distrib();
    graph.print_analysis();

    std::unique_ptr<int[]> p_weights(new (std::nothrow) int[graph.nV()]);
    ASSERT_NE(p_weights.get(), nullptr);
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::mt19937 engine(seed);
    std::uniform_int_distribution<int> distrib(0, 100);
    std::generate(p_weights.get(), p_weights.get() + graph.nV(), [&](){ return distrib(engine); } );

    graph::GraphWeight<int, int, int> graph_weight(graph.csr_out_edges(), graph.nV(), graph.csr_out_edges(), graph.nE(), p_weights.get());

    graph::Dijkstra<int, int, int> dijkstra(graph_weight);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < graph.nV(); ++i) {
        dijkstra.run(i);
        dijkstra.reset();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Dijkstra took " << diff.count() * 1000.0/* s to ms */ << " ms." << std::endl;

    return;
}

}

class LoadBalanceTest : public HornetTest {
public:
    static int argc;
    static char** argv;

protected:
};

int LoadBalanceTest::argc = 0;
char** LoadBalanceTest::argv = nullptr;

TEST_F(LoadBalanceTest, LoadBalanceBinarySearchTest) {
    ASSERT_TRUE(LoadBalanceTest::argc >= 2);
    exec(LoadBalanceTest::argc, LoadBalanceTest::argv);
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    LoadBalanceTest::argc = argc;
    LoadBalanceTest::argv = argv;
    return RUN_ALL_TESTS();
}

