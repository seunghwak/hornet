/**
 * @brief
 * @author Oded Green                                                       <br>
 *   Georgia Institute of Technology, Computational Science and Engineering <br>                   <br>
 *   ogreen@gatech.edu
 * @date August, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 Hornet. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 *
 * @file
 */

#include <iostream>
#include <chrono>

#include <gtest/gtest.h>

#include <Graph/GraphStd.hpp>
#include <Graph/GraphWeight.hpp>

#include <StandardAPI.hpp>

#include <Util/CommandLineParam.hpp>

#include "Static/KatzCentrality/Katz.cuh"
#include "Dynamic/KatzCentrality/Katz.cuh"

#include "../../hornet/tests/hornet_test_fixtures.h"

namespace {

void exec(int argc, char* argv[]) {//based on the old KatzTest.cu
    graph::GraphStd<hornets_nest::vid_t, hornets_nest::eoff_t> graph(graph::structure_prop::UNDIRECTED);
    graph.read(argv[1], graph::parsing_prop::SORT | graph::parsing_prop::PRINT_INFO);

    int max_iterations = 50;// Limit the number of iteartions for graphs with large number of vertices.
    int topK = graph.nV();

    hornets_nest::HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());
    hornets_nest::HornetGraph hornet_graph(hornet_init);
 
    hornets_nest::degree_t max_degree = hornet_graph.max_degree();// Finding largest vertex degree

    hornets_nest::KatzCentrality kcPostUpdate(hornet_graph, max_iterations, topK, max_degree);

    auto start = std::chrono::high_resolution_clock::now();

    kcPostUpdate.run();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "Computation time: " << diff.count() * 1000/* s to ms */ << " ms" << std::endl;

    auto total_time = diff.count() * 1000/* s to ms */;

    std::cout << "The number of iterations     : "
              << kcPostUpdate.get_iteration_count()
              << "\nTopK                       : " << topK 
              << "\nTotal time for KC          : " << total_time
              << "\nAverage time per iteartion : "
              << total_time /
                 static_cast<float>(kcPostUpdate.get_iteration_count())
              << "\n";

    return;
}

void execDynamic(int argc, char* argv[]) {//based on the old KatzDynamicTest.cu, but this does not use KatzDynamicCentrality, and KatzDynamicCentrality's run() is declared but no defined... Meh...
    graph::GraphStd<hornets_nest::vid_t, hornets_nest::eoff_t> graph(graph::structure_prop::UNDIRECTED | graph::structure_prop::ENABLE_INGOING);
    graph.read(argv[1], graph::parsing_prop::SORT | graph::parsing_prop::PRINT_INFO);

    int max_iterations = 1000;
    int topK = 100;

    hornets_nest::HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());
    hornets_nest::HornetGraph hornet_graph(hornet_init);
 
    hornets_nest::degree_t max_degree = hornet_graph.max_degree();// Finding largest vertex degree

    hornets_nest::KatzCentrality kcPostUpdate(hornet_graph, max_iterations, topK, max_degree);

    auto start = std::chrono::high_resolution_clock::now();

    kcPostUpdate.run();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "Computation time: " << diff.count() * 1000/* s to ms */ << " ms" << std::endl;

    auto total_time = diff.count() * 1000/* s to ms */;

    std::cout << "The number of iterations     : "
              << kcPostUpdate.get_iteration_count()
              << "\nTopK                       : " << topK 
              << "\nTotal time for KC          : " << total_time
              << "\nAverage time per iteartion : "
              << total_time /
                 static_cast<float>(kcPostUpdate.get_iteration_count())
              << "\n";

    return;
}

}

class KatzCentralityTest : public HornetTest {
public:
    static int argc;
    static char** argv;

protected:
};

int KatzCentralityTest::argc = 0;
char** KatzCentralityTest::argv = nullptr;

TEST_F(KatzCentralityTest, KatzCentralityStaticTest) {
    ASSERT_TRUE(KatzCentralityTest::argc >= 2);
    exec(KatzCentralityTest::argc, KatzCentralityTest::argv);
}

TEST_F(KatzCentralityTest, KatzCentralityDynamicTest) {
    ASSERT_TRUE(KatzCentralityTest::argc >= 2);
    execDynamic(KatzCentralityTest::argc, KatzCentralityTest::argv);
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    KatzCentralityTest::argc = argc;
    KatzCentralityTest::argv = argv;
    return RUN_ALL_TESTS();
}

