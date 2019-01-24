#include <chrono>
#include <string>

#include <gtest/gtest.h>

#include "StandardAPI.hpp"

#include "Hornet.hpp"
#include "Core/GPUHornet/BatchUpdate.cuh"
#include "Util/BatchFunctions.hpp"

#include "../hornet_test_fixtures.h"

namespace {

void exec(const bool insertion/* insertion if true, deletion if false */, int argc, char* argv[]) {
    std::string op_str;

    if (insertion == true) {
        op_str = "insertion";
    }
    else {
        op_str = "deletion";
    }

    graph::GraphStd<hornets_nest::vid_t, hornets_nest::eoff_t> graph;
    graph.read(argv[1]);

    int batch_size = std::stoi(argv[2]);

    hornets_nest::HornetInit hornet_init(graph.nV(), graph.nE(), graph.csr_out_offsets(), graph.csr_out_edges());
    hornets_nest::gpu::Hornet<std::tuple<>,std::tuple<>> hornet(hornet_init);

    std::cout << std::string(80, '-') << std::endl;

    auto my_new = [](const size_t size) { hornets_nest::vid_t* p_elems = nullptr; hornets_nest::host::allocatePageLocked(p_elems, size); return p_elems; };
    auto my_del = [](hornets_nest::vid_t* p_elems) { hornets_nest::host::freePageLocked(p_elems); };
    std::unique_ptr<hornets_nest::vid_t[], decltype(my_del)> p_batch_src(my_new(batch_size), my_del);
    std::unique_ptr<hornets_nest::vid_t[], decltype(my_del)> p_batch_dst(my_new(batch_size), my_del);

    if (insertion == true) {
        hornets_nest::generateBatch(graph, batch_size, p_batch_src.get(), p_batch_dst.get(), hornets_nest::BatchGenType::INSERT, hornets_nest::batch_gen_property::UNIQUE);//old HornetTest.cu
        //hornets_nest::generateBatch(graph, batch_size, p_batch_src.get(), p_batch_dst.get(), hornets_nest::BatchGenType::INSERT);//old HornetInsertTest.cu
    }
    else {
        hornets_nest::generateBatch(graph, batch_size, p_batch_src.get(), p_batch_dst.get(), hornets_nest::BatchGenType::INSERT);//old HornetDeleteTest.cu
    }

    hornets_nest::gpu::BatchUpdate batch_update(p_batch_src.get(), p_batch_dst.get(), batch_size);

    if (insertion == true) {
        hornet.reserveBatchOpResource(batch_size);
    }
    else {
        hornet.reserveBatchOpResource(batch_size, hornets_nest::gpu::batch_property::IN_PLACE | hornets_nest::gpu::batch_property::REMOVE_BATCH_DUPLICATE);
    }

    std::cout << "BEFORE batch " << op_str << std::endl;
    hornet.print();
    std::cout << "# edges:" << hornet.nE() << std::endl;

    std::cout << std::string(80, '-') << std::endl;

    batch_update.print();

    auto start = std::chrono::high_resolution_clock::now();

    if (insertion == true) {
        hornet.insertEdgeBatch(batch_update);
    }
    else {
        hornet.deleteEdgeBatch(batch_update);
    }

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "batch " << op_str << " execution time:" << diff.count() * 1000 << " ms" << std::endl;

    std::cout << std::string(80, '-') << std::endl;

    std::cout << "AFTER batch " << op_str << std::endl;
    hornet.print();
    std::cout << "# edges:" << hornet.nE() << std::endl;

    return;
}

}

class InsertionDeletionTest : public HornetTest {
public:
    static int argc;
    static char** argv;

protected:
};

int InsertionDeletionTest::argc = 0;
char** InsertionDeletionTest::argv = nullptr;

TEST_F(InsertionDeletionTest, InsertionDeletionInsertionTest) {
    ASSERT_TRUE(InsertionDeletionTest::argc >= 3);
    exec(true, InsertionDeletionTest::argc, InsertionDeletionTest::argv);
}

TEST_F(InsertionDeletionTest, InsertionDeletionDeletionTest) {
    ASSERT_TRUE(InsertionDeletionTest::argc >= 3);
    exec(false, InsertionDeletionTest::argc, InsertionDeletionTest::argv);
}

int main(int argc, char* argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    InsertionDeletionTest::argc = argc;
    InsertionDeletionTest::argv = argv;
    return RUN_ALL_TESTS();
}

