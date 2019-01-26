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

#include "Static/BetweennessCentrality/bc.cuh"
#include "Static/BetweennessCentrality/exact_bc.cuh"
#include "Static/BetweennessCentrality/approximate_bc.cuh"

#include "../../hornet/tests/hornet_test_fixtures.h"

namespace {

void exec(int argc, char* argv[]) {
    graph::GraphStd<hornets_nest::vid_t, hornets_nest::eoff_t> graph(graph::structure_prop::UNDIRECTED);
    hornets_nest::CommandLineParam cmd(graph, argc, argv,false);

    hornets_nest::HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());
    hornets_nest::HornetGraph hornet_graph(hornet_init);

    hornets_nest::BCCentrality bc(hornet_graph);

    auto root = graph.max_out_degree_id();
    bc.reset();
    bc.setRoot(root);

    auto start = std::chrono::high_resolution_clock::now();

    bc.run();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "Computation time: " << diff.count() * 1000/* s to ms */ << " ms" << std::endl;

    return;
}

void execExact(int argc, char* argv[]) {
    graph::GraphStd<hornets_nest::vid_t, hornets_nest::eoff_t> graph(graph::structure_prop::UNDIRECTED);
    hornets_nest::CommandLineParam cmd(graph, argc, argv,false);

    hornets_nest::HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());
    hornets_nest::HornetGraph hornet_graph(hornet_init);

    hornets_nest::ExactBC ebc(hornet_graph);

    ebc.reset();

    auto start = std::chrono::high_resolution_clock::now();

    ebc.run();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "Computation time: " << diff.count() * 1000/* s to ms */ << " ms" << std::endl;

    return;
}

void execApproximate(int argc, char* argv[]) {
    graph::GraphStd<hornets_nest::vid_t, hornets_nest::eoff_t> graph(graph::structure_prop::UNDIRECTED);
    hornets_nest::CommandLineParam cmd(graph, argc, argv,false);

    hornets_nest::HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());
    hornets_nest::HornetGraph hornet_graph(hornet_init);

    hornets_nest::vid_t numRoots = std::min(1000, graph.nV());
    hornets_nest::vid_t* p_roots = nullptr;
    hornets_nest::ApproximateBC::generateRandomRootsUniform(hornet_graph.nV(), numRoots, &p_roots, 1 );
    ASSERT_NE(p_roots, nullptr);

    hornets_nest::ApproximateBC abc(hornet_graph, p_roots, numRoots);

    abc.reset();

    auto start = std::chrono::high_resolution_clock::now();

    abc.run();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "Computation time: " << diff.count() * 1000/* s to ms */ << " ms" << std::endl;

    delete[] p_roots;

    return;
}

}

class BCTest : public HornetTest {
public:
    static int argc;
    static char** argv;

protected:
};

int BCTest::argc = 0;
char** BCTest::argv = nullptr;

TEST_F(BCTest, BCTest) {
    ASSERT_TRUE(BCTest::argc >= 2);
    exec(BCTest::argc, BCTest::argv);
}

TEST_F(BCTest, ExactBCTest) {
    ASSERT_TRUE(BCTest::argc >= 2);
    execExact(BCTest::argc, BCTest::argv);
}

TEST_F(BCTest, ApproximateBCTest) {
    ASSERT_TRUE(BCTest::argc >= 2);
    execApproximate(BCTest::argc, BCTest::argv);
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    BCTest::argc = argc;
    BCTest::argv = argv;
    return RUN_ALL_TESTS();
}

