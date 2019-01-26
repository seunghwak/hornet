#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>

#include <gtest/gtest.h>

#include "Static/TriangleCounting/triangle.cuh"
#include "Static/TriangleCounting/triangle2.cuh"
#include <StandardAPI.hpp>
#include <Device/Util/Timer.cuh>

#include <Graph/GraphStd.hpp>

#include "../../hornet/tests/hornet_test_fixtures.h"

using namespace std;
using namespace timer;
using namespace hornets_nest;

triangle_t hostSingleIntersection (const vid_t* p_start0, const vid_t* p_end0, const vid_t* p_start1, const vid_t* p_end1) {//assume sorted index lists
    if ((p_start0 == p_end0) || (p_start1 == p_end1) || (*(p_end0 - 1) < *p_start1) || (*(p_end1 - 1) < *p_start0)) {
        return 0;
    }
    else {
        triangle_t ret = 0;
        const vid_t* p_cur0 = p_start0;
        const vid_t* p_cur1 = p_start1;

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

triangle_t hostCountTriangles (const vid_t num_vertices, const vid_t num_edges, const eoff_t * p_offsets, const vid_t * p_indices) {
    triangle_t ret = 0;
    for (vid_t src = 0; src < num_vertices; ++src) {
        degree_t src_degree = p_offsets[src + 1] - p_offsets[src];
        for (auto nbr_index = p_offsets[src]; nbr_index < p_offsets[src + 1]; ++nbr_index) {
            vid_t dst = p_indices[nbr_index];
            degree_t dst_degree = p_offsets[dst + 1] - p_offsets[dst];
            ret += hostSingleIntersection (p_indices + p_offsets[src], p_indices + p_offsets[src] + src_degree, p_indices + p_offsets[dst], p_indices + p_offsets[dst] + dst_degree);
        }
    }
    return ret;
}

void execOrg(const int argc, char *argv[]) {//use TriangleCounting and based on the old TriangleCountingTest.cu file.
    graph::GraphStd<vid_t, eoff_t> graph(graph::structure_prop::UNDIRECTED);
    graph.read(argv[1], graph::parsing_prop::DIRECTED_BY_DEGREE | graph::parsing_prop::SORT | graph::parsing_prop::PRINT_INFO);

    auto host_count = hostCountTriangles(graph.nV(), graph.nE(),graph.csr_out_offsets(), graph.csr_out_edges());
    cout << "host_count=" << host_count << std::endl;

    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());
    hornets_nest::gpu::Hornet<EMPTY,EMPTY> hornet_graph(hornet_init);

    hornet_graph.check_sorted_adjs();

    constexpr int blocks = 96000;
    constexpr int sps = 192;
    constexpr int tsp = 16;
    constexpr int shifter = 3;

    const std::vector<int> v_cutoff = { -1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700 };//this is bad... currently, cutoff part is disabled in triangle.cu... also, host_count and device_count do not always match... not sure this is worth debugging... we may just replace TriangleCounting with TriangleCounting2

    for (auto cutoff : v_cutoff) {
        TriangleCounting tc(hornet_graph);

        tc.setInitParameters(blocks,sps,tsp);
        tc.init();
        tc.reset();

        auto start = std::chrono::high_resolution_clock::now();

        tc.run(cutoff);

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> diff = end - start;
        std::cout << "Computation time:" << diff.count() * 1000/* s to ms */ << " ms" << std::endl;

        auto device_count = tc.countTriangles();
        tc.release();

        std::cout << "cutoff=" << cutoff << " blocks=" << blocks << " sps=" << sps << " tsp=" << tsp << " shifter=" << shifter << " device_count=" << device_count << std::endl;
    }

    return;
}

void exec2(const int argc, char* argv[]) {//use TriangleCounting2 and based on the old TriangleCountingTest2.cu file.
    graph::GraphStd<vid_t, eoff_t> graph(graph::structure_prop::UNDIRECTED);
    graph.read(argv[1], graph::parsing_prop::DIRECTED_BY_DEGREE | graph::parsing_prop::PRINT_INFO | graph::parsing_prop::SORT);

    int work_factor;
    if (argc >= 3) {
        work_factor = atoi(argv[2]);
    } else {
        work_factor = 1;
    }

    auto host_count = hostCountTriangles(graph.nV(), graph.nE(),graph.csr_out_offsets(), graph.csr_out_edges());

    HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());
    hornets_nest::gpu::Hornet<EMPTY,EMPTY> hornet_graph(hornet_init);

    hornet_graph.check_sorted_adjs();

    TriangleCounting2 tc2(hornet_graph);

    tc2.init();

    auto start = std::chrono::high_resolution_clock::now();

    tc2.run(work_factor);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "Computation time:" << diff.count() * 1000/* s to ms */ << " ms" << std::endl;

    auto device_count = tc2.countTriangles();
    tc2.release();

    std::cout << "host_count=" << host_count << " device_count=" << device_count << std::endl;
    ASSERT_EQ(host_count, device_count);

    return;
}

class TriangleCountingTest : public HornetTest {
public:
    static int argc;
    static char** argv;

protected:
};

int TriangleCountingTest::argc = 0;
char** TriangleCountingTest::argv = nullptr;

TEST_F(TriangleCountingTest, TriangleCountingOrgTest) {
    ASSERT_TRUE(TriangleCountingTest::argc >= 2);
    execOrg(TriangleCountingTest::argc, TriangleCountingTest::argv);
}

TEST_F(TriangleCountingTest, TriangleCounting2Test) {
    ASSERT_TRUE(TriangleCountingTest::argc >= 2);
    exec2(TriangleCountingTest::argc, TriangleCountingTest::argv);
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    TriangleCountingTest::argc = argc;
    TriangleCountingTest::argv = argv;
    return RUN_ALL_TESTS();
}

