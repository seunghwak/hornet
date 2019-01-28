#include <chrono>
#include <iostream>

#include <gtest/gtest.h>

#include <Graph/GraphStd.hpp>

#include <StandardAPI.hpp>

#include "Static/TriangleCounting/triangle.cuh"

#include "../../hornet/tests/hornet_test_fixtures.h"

namespace {

hornets_nest::triangle_t hostSingleIntersection (const hornets_nest::vid_t* p_start0, const hornets_nest::vid_t* p_end0, const hornets_nest::vid_t* p_start1, const hornets_nest::vid_t* p_end1) {//assume sorted index lists
    if ((p_start0 == p_end0) || (p_start1 == p_end1) || (*(p_end0 - 1) < *p_start1) || (*(p_end1 - 1) < *p_start0)) {
        return 0;
    }
    else {
        hornets_nest::triangle_t ret = 0;
        const hornets_nest::vid_t* p_cur0 = p_start0;
        const hornets_nest::vid_t* p_cur1 = p_start1;

        while ((p_cur0 < p_end0) && (p_cur1 < p_end1)) {
            if (*p_cur0 == *p_cur1) {
                ++p_cur0;
                ++p_cur1;
                ret++;
            }
            else if (*p_cur0 < *p_cur1) {
                ++p_cur0;
            }
            else {
                assert(*p_cur1 < *p_cur0);
                ++p_cur1;
            }
        }
        return ret;
    }
}

hornets_nest::triangle_t hostCountTriangles (const hornets_nest::vid_t num_vertices, const hornets_nest::vid_t num_edges, const hornets_nest::eoff_t * p_offsets, const hornets_nest::vid_t * p_indices) {
    hornets_nest::triangle_t ret = 0;
    for (hornets_nest::vid_t src = 0; src < num_vertices; ++src) {
        hornets_nest::degree_t src_degree = p_offsets[src + 1] - p_offsets[src];
        for (auto nbr_index = p_offsets[src]; nbr_index < p_offsets[src + 1]; ++nbr_index) {
            hornets_nest::vid_t dst = p_indices[nbr_index];
            hornets_nest::degree_t dst_degree = p_offsets[dst + 1] - p_offsets[dst];
            ret += hostSingleIntersection (p_indices + p_offsets[src], p_indices + p_offsets[src] + src_degree, p_indices + p_offsets[dst], p_indices + p_offsets[dst] + dst_degree);
        }
    }
    return ret;
}

void exec(const int argc, char* argv[]) {
    graph::GraphStd<hornets_nest::vid_t, hornets_nest::eoff_t> graph(graph::structure_prop::UNDIRECTED);
    graph.read(argv[1], graph::parsing_prop::DIRECTED_BY_DEGREE | graph::parsing_prop::PRINT_INFO | graph::parsing_prop::SORT);

    int work_factor;
    if (argc >= 3) {
        work_factor = atoi(argv[2]);
    } else {
        work_factor = 1;
    }

    auto host_count = hostCountTriangles(graph.nV(), graph.nE(),graph.csr_out_offsets(), graph.csr_out_edges());

    hornets_nest::HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());
    hornets_nest::gpu::Hornet<std::tuple<>, std::tuple<>> hornet_graph(hornet_init);

    hornet_graph.check_sorted_adjs();

    hornets_nest::TriangleCounting tc(hornet_graph);

    tc.init();

    auto start = std::chrono::high_resolution_clock::now();

    tc.run(work_factor);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "Computation time: " << diff.count() * 1000/* s to ms */ << " ms" << std::endl;

    auto device_count = tc.countTriangles();
    tc.release();

    std::cout << "host_count=" << host_count << " device_count=" << device_count << std::endl;
    ASSERT_EQ(host_count, device_count);

    return;
}

}

class TriangleCountingTest : public HornetTest {
public:
    static int argc;
    static char** argv;

protected:
};

int TriangleCountingTest::argc = 0;
char** TriangleCountingTest::argv = nullptr;

TEST_F(TriangleCountingTest, TriangleCountingTest) {
    ASSERT_TRUE(TriangleCountingTest::argc >= 2);
    exec(TriangleCountingTest::argc, TriangleCountingTest::argv);
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    TriangleCountingTest::argc = argc;
    TriangleCountingTest::argv = argv;
    return RUN_ALL_TESTS();
}

