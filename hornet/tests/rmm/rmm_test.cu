#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include <gtest/gtest.h>

#include <Host/Basic.hpp>//xlib::byte_t
#include <Host/PrintExt.hpp>//xlib::human_readable

#include <StandardAPI.hpp>

#include "../hornet_test_fixtures.h"

void exec() {
#if defined(RMM_WRAPPER)
    constexpr size_t repeat_cnt = 1;
    constexpr size_t min_size = 1024;//1KB
    size_t round = 0;

    std::vector<double> v_alloc_time_host_cpp;//new and delete
    std::vector<double> v_alloc_time_host_cuda;//cudaMallocHost and cudaFreeHost
    std::vector<double> v_alloc_time_device_cuda;//cudaMalloc and cudaFree
    std::vector<double> v_alloc_time_device_rmm;//RMM_ALLOC and RMM_FREE

    std::cout << "Computing (repeat count=" << repeat_cnt << ", RMM alloc mode=pool)";

    while (true) {
        size_t size = min_size << round;
        bool success = true;

        std::cout << "." << std::flush;

        //host malloc/free

        if (success == true) {
            auto start = std::chrono::high_resolution_clock::now();
            for (std::size_t i = 0; i < repeat_cnt; ++i) {
                std::unique_ptr<xlib::byte_t[]> h_p_cpp(new (std::nothrow) xlib::byte_t[size]);//no initialization, should not use std::make_unique here as this enforces initialization and is slower.
                if (h_p_cpp == nullptr) {
                    std::cout << std::endl;
                    std::cout << "new failed (size=" << xlib::human_readable(size) << "), this is normal if the size exceeds available host memory." << std::endl;
                    success = false;
                    break;
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            v_alloc_time_host_cpp.push_back(diff.count() * 1000.0/* s to ms */);
        }

        //host malloc/free (page-locked)

        if (success == true ) {
            auto start = std::chrono::high_resolution_clock::now();
            for (std::size_t i = 0; i < repeat_cnt; ++i) {
                auto my_new = [](const size_t size) { xlib::byte_t* h_p_cuda; auto result = cudaMallocHost(&h_p_cuda, size); if (result == cudaSuccess) { return static_cast<xlib::byte_t*>(h_p_cuda); } else { return static_cast<xlib::byte_t*>(nullptr); } };
                auto my_del = [](xlib::byte_t* h_p_cuda) { SAFE_CALL(cudaFreeHost(h_p_cuda)); };
                std::unique_ptr<xlib::byte_t[], decltype(my_del)> h_p_cuda(my_new(size), my_del);
                if (h_p_cuda == nullptr) {
                    std::cout << std::endl;
                    std::cout << "cudaMallocHost failed (size=" << xlib::human_readable(size) << "), this is normal if the size exceeds available host memory (that can be page-locked)." << std::endl;
                    success = false;
                    break;
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            v_alloc_time_host_cuda.push_back(diff.count() * 1000.0/* s to ms */);
        }

        //device malloc/free

        if (success == true ) {
            auto start = std::chrono::high_resolution_clock::now();
            for (std::size_t i = 0; i < repeat_cnt; ++i) {
                auto my_new = [](const size_t size) { xlib::byte_t* d_p_cuda; auto result = cudaMalloc(&d_p_cuda, size); if (result == cudaSuccess) { return static_cast<xlib::byte_t*>(d_p_cuda); } else { return static_cast<xlib::byte_t*>(nullptr); } };
                auto my_del = [](xlib::byte_t* d_p_cuda) { SAFE_CALL(cudaFree(d_p_cuda)); };
                std::unique_ptr<xlib::byte_t[], decltype(my_del)> d_p_cuda(my_new(size), my_del);
                if (d_p_cuda == nullptr) {
                    std::cout << std::endl;
                    std::cout << "cudaMalloc failed (size=" << xlib::human_readable(size) << "), this is normal if the size exceeds available device memory." << std::endl;
                    success = false;
                    break;
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            v_alloc_time_device_cuda.push_back(diff.count() * 1000.0/* s to ms */);
        }

        //device malloc/free (page-locked)

        if (success == true ) {
            auto start = std::chrono::high_resolution_clock::now();
            for (std::size_t i = 0; i < repeat_cnt; ++i) {
                auto my_new = [](const size_t size) { xlib::byte_t* d_p_rmm; auto result = RMM_ALLOC(&d_p_rmm, size, 0);/* by default, use the default stream, RMM_ALLOC instead of hornets_nest::gpu::allocate to test return value, gpu::allocate calls std::exit on error */ if (result == RMM_SUCCESS) { return static_cast<xlib::byte_t*>(d_p_rmm); } else { return static_cast<xlib::byte_t*>(nullptr); } };
                auto my_del = [](xlib::byte_t* d_p_rmm) { hornets_nest::gpu::free(d_p_rmm); };
                std::unique_ptr<xlib::byte_t[], decltype(my_del)> d_p_rmm(my_new(size), my_del);
                if (d_p_rmm == nullptr) {
                    std::cout << std::endl;
                    std::cout << "RMM_ALLOC failed (size=" << xlib::human_readable(size) << "), this is normal if the size exceeds available device memory (accessible to RMM)." << std::endl;
                    success = false;
                    break;
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            v_alloc_time_device_rmm.push_back(diff.count() * 1000.0/* s to ms */);
        }

        if (success == true ) {
            round++;
        }
        else {
            v_alloc_time_host_cpp.resize(round);
            v_alloc_time_host_cuda.resize(round);
            v_alloc_time_device_cuda.resize(round);
            v_alloc_time_device_rmm.resize(round);
            break;
        }
    }

    std::cout << "RESULT:" << std::endl;
    std::cout << std::setprecision(2) << std::right << std::fixed
              << std::setw(8)  << "SIZE"
              << std::setw(16) << "malloc"
              << std::setw(16) << "cudaMallocHost"
              << std::setw(16) << "cudaMalloc"
              << std::setw(16) << "rmmAlloc" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (size_t i = 0; i < round; ++i) {
        std::cout << std::setw(8)  << xlib::human_readable(min_size << i)
                  << std::setw(16) << v_alloc_time_host_cpp[i]
                  << std::setw(16) << v_alloc_time_host_cuda[i]
                  << std::setw(16) << v_alloc_time_device_cuda[i]
                  << std::setw(16) << v_alloc_time_device_rmm[i] << std::endl;
    }

    std::cout << "* unit: ms, measured time includes both memory allocation and deallocation." << std::endl;
#else
    std::cout << "RMM_WRAPPER should be defined to benchmark RMM." << std::endl;
#endif

    return;
}

class RMMTest : public HornetTest {
protected:
};


TEST_F(RMMTest, RMMBenchmarkTest) {
    exec();
}

