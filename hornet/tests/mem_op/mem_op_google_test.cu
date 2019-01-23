#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include <Host/Basic.hpp>//xlib::byte_t
#include <Host/PrintExt.hpp>//xlib::human_readable

#include <StandardAPI.hpp>

#include "../hornet_test_fixtures.h"

void exec() {
    constexpr size_t min_size = 1024;
    size_t round = 0;

    std::vector<double> v_alloc_time_host_cpp;//new and delete
    std::vector<double> v_alloc_time_host_cuda;//cudaMallocHost and cudaFreeHost
    std::vector<double> v_alloc_time_device_cuda;//cudaMalloc and cudaFree
    std::vector<double> v_memset_time_host;
    std::vector<double> v_memset_time_host_page_locked;
    std::vector<double> v_memset_time_device;
    std::vector<double> v_copy_time_host_to_device;
    std::vector<double> v_copy_time_host_page_locked_to_device;
    std::vector<double> v_copy_time_device_to_device;

    auto my_host_cuda_del = [](xlib::byte_t* h_p_cuda) { SAFE_CALL(cudaFreeHost(h_p_cuda)); };
    auto my_device_cuda_del = [](xlib::byte_t* d_p_cuda) { SAFE_CALL(cudaFree(d_p_cuda)); };

    cudaEvent_t start;//for asynchronous (with respect to the host) cuda calls
    cudaEvent_t stop;

/*
    Source: https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html
    CUDA Toolkit Documentation - Last updated October 30, 2018

    2. API synchronization behavior

    The API provides memcpy/memset functions in both synchronous and asynchronous forms, the latter having an "Async" suffix. This is a misnomer as each function may exhibit synchronous or asynchronous behavior depending on the arguments passed to the function.

    Memcpy

    In the reference documentation, each memcpy function is categorized as synchronous or asynchronous, corresponding to the definitions below.

    Synchronous

    All transfers involving Unified Memory regions are fully synchronous with respect to the host.

    For transfers from pageable host memory to device memory, a stream sync is performed before the copy is initiated. The function will return once the pageable buffer has been copied to the staging memory for DMA transfer to device memory, but the DMA to final destination may not have completed.

    For transfers from pinned host memory to device memory, the function is synchronous with respect to the host.

    For transfers from device to either pageable or pinned host memory, the function returns only once the copy has completed.

    For transfers from device memory to device memory, no host-side synchronization is performed.

    For transfers from any host memory to any host memory, the function is fully synchronous with respect to the host.

    Asynchronous

    For transfers from device memory to pageable host memory, the function will return only once the copy has completed.

    For transfers from any host memory to any host memory, the function is fully synchronous with respect to the host.

    For all other transfers, the function is fully asynchronous. If pageable memory must first be staged to pinned memory, this will be handled asynchronously with a worker thread.

    Memset

    The synchronous memset functions are asynchronous with respect to the host except when the target is pinned host memory or a Unified Memory region, in which case they are fully synchronous. The Async versions are always asynchronous with respect to the host.
*/

    std::cout << "Computing";

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    while (true) {
        size_t size = min_size << round;
        bool success = true;

        std::cout << "." << std::flush;

        std::unique_ptr<xlib::byte_t[]> h_p_cpp;
        std::unique_ptr<xlib::byte_t[], decltype(my_host_cuda_del)> h_p_cuda(nullptr, my_host_cuda_del);//page-locked
        std::unique_ptr<xlib::byte_t[], decltype(my_device_cuda_del)> d_p_cuda0(nullptr, my_device_cuda_del);
        std::unique_ptr<xlib::byte_t[], decltype(my_device_cuda_del)> d_p_cuda1(nullptr, my_device_cuda_del);//for device to device memcpy

        //host malloc

        if (success == true) {
            auto start = std::chrono::high_resolution_clock::now();
            h_p_cpp.reset(new (std::nothrow) xlib::byte_t[size]);
            if (h_p_cpp == nullptr) {
                std::cout << std::endl;
                std::cout << "new failed (size=" << xlib::human_readable(size) << "), this is normal if the size exceeds available host memory." << std::endl;
                success = false;
                break;
            }
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            v_alloc_time_host_cpp.push_back(diff.count() * 1000.0/* s to ms */);
        }

        //host malloc (page-locked)

        if (success == true) {
            auto start = std::chrono::high_resolution_clock::now();
            xlib::byte_t* p_tmp = nullptr;
            auto result = cudaMallocHost(&p_tmp, size);
            if (result == cudaSuccess) {
                h_p_cuda.reset(p_tmp);
            }
            else {
                h_p_cuda.reset(nullptr);
            }
            if (h_p_cuda == nullptr) {
                std::cout << std::endl;
                std::cout << "cudaMallocHost failed (size=" << xlib::human_readable(size) << "), this is normal if the size exceeds available host memory (that can be page-locked)." << std::endl;
                success = false;
                break;
            }
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            v_alloc_time_host_cuda.push_back(diff.count() * 1000.0/* s to ms */);
        }

        //cuda malloc (d_p_cuda0)

        if (success == true) {
            auto start = std::chrono::high_resolution_clock::now();
            xlib::byte_t* p_tmp = nullptr;
            auto result = cudaMalloc(&p_tmp, size);
            if (result == cudaSuccess) {
                d_p_cuda0.reset(p_tmp);
            }
            else  {
                d_p_cuda0.reset(nullptr);
            }
            if (d_p_cuda0 == nullptr) {
                std::cout << std::endl;
                std::cout << "cudaMalloc failed (size=" << xlib::human_readable(size) << "), this is normal if the size exceeds available device memory." << std::endl;
                success = false;
                break;
            }
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            v_alloc_time_device_cuda.push_back(diff.count() * 1000.0/* s to ms */);
        }

        //cuda malloc (d_p_cuda1)

        if (success == true) {
            xlib::byte_t* p_tmp = nullptr;
            auto result = cudaMalloc(&p_tmp, size);
            if (result == cudaSuccess) {
                d_p_cuda1.reset(p_tmp);
            }
            else  {
                d_p_cuda1.reset(nullptr);
            }
        }

        //memset host memory

        if (success == true) {
            auto start = std::chrono::high_resolution_clock::now();
            hornets_nest::host::memsetZero(h_p_cpp.get(), size);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            v_memset_time_host.push_back(diff.count() * 1000.0/* s to ms */);
        }

        //memset host (page-locked) memory

        if (success == true) {
            auto start = std::chrono::high_resolution_clock::now();
            hornets_nest::host::memsetZero(h_p_cuda.get(), size);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            v_memset_time_host_page_locked.push_back(diff.count() * 1000.0/* s to ms */);
        }

        //memset device memory

        if (success == true) {
            float elapsed_gpu_time = 0.0;
            ASSERT_EQ(cudaEventRecord(start), cudaSuccess);
            hornets_nest::gpu::memsetZero(d_p_cuda0.get(), size);
            ASSERT_EQ(cudaEventRecord(stop), cudaSuccess);
            ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);
            ASSERT_EQ(cudaEventElapsedTime(&elapsed_gpu_time, start, stop), cudaSuccess);
            v_memset_time_device.push_back(elapsed_gpu_time);
        }

        //copy from host to device

        if (success == true) {
            float elapsed_gpu_time = 0.0;
            ASSERT_EQ(cudaEventRecord(start), cudaSuccess);
            hornets_nest::host::copyToDevice(h_p_cpp.get(), size, d_p_cuda0.get());
            ASSERT_EQ(cudaEventRecord(stop), cudaSuccess);
            ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);
            ASSERT_EQ(cudaEventElapsedTime(&elapsed_gpu_time, start, stop), cudaSuccess);
            v_copy_time_host_to_device.push_back(elapsed_gpu_time);
        }

        //copy from host (page-locked) to devcie

        if (success == true) {
            auto start = std::chrono::high_resolution_clock::now();
            hornets_nest::host::copyToDevice(h_p_cuda.get(), size, d_p_cuda0.get());
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            v_copy_time_host_page_locked_to_device.push_back(diff.count() * 1000.0/* s to ms */);
        }

        //copy from device to device

        if (success == true) {
            if (d_p_cuda1 != nullptr) {
                float elapsed_gpu_time = 0.0;
                ASSERT_EQ(cudaEventRecord(start), cudaSuccess);
                hornets_nest::gpu::copyToDevice(d_p_cuda0.get(), size, d_p_cuda1.get());
                ASSERT_EQ(cudaEventRecord(stop), cudaSuccess);
                ASSERT_EQ(cudaEventSynchronize(stop), cudaSuccess);
                ASSERT_EQ(cudaEventElapsedTime(&elapsed_gpu_time, start, stop), cudaSuccess);
                v_copy_time_device_to_device.push_back(elapsed_gpu_time);
            }
            else {
                v_copy_time_device_to_device.push_back(std::nan(""));
            }
        }

        if (success == true) {
            round++;
        }
        else {
            v_alloc_time_host_cpp.resize(round);
            v_alloc_time_host_cuda.resize(round);
            v_alloc_time_device_cuda.resize(round);
            v_memset_time_host.resize(round);
            v_memset_time_host_page_locked.resize(round);
            v_memset_time_device.resize(round);
            v_copy_time_host_to_device.resize(round);
            v_copy_time_host_page_locked_to_device.resize(round);
            v_copy_time_device_to_device.resize(round);
        }
    }

    std::cout << "RESULT:" << std::endl;
    std::cout << "Memory Allocation" << std::endl;
    std::cout << std::setprecision(2) << std::right << std::fixed
              << std::setw(8)  << "SIZE"
              << std::setw(20) << "host"
              << std::setw(20) << "host (pinned)"
              << std::setw(20) << "device" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (size_t i = 0; i < round; i++) {
        std::cout << std::setw(8)  << xlib::human_readable(min_size << i)
                  << std::setw(20) << v_alloc_time_host_cpp[i]
                  << std::setw(20) << v_alloc_time_host_cuda[i]
                  << std::setw(20) << v_alloc_time_device_cuda[i] << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Memset" << std::endl;
    std::cout << std::setprecision(2) << std::right << std::fixed
              << std::setw(8)  << "SIZE"
              << std::setw(20) << "host"
              << std::setw(20) << "host (pinned)"
              << std::setw(20) << "device" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (size_t i = 0; i < round; i++) {
        std::cout << std::setw(8)  << xlib::human_readable(min_size << i)
                  << std::setw(20)  << v_memset_time_host[i]
                  << std::setw(20)  << v_memset_time_host_page_locked[i]
                  << std::setw(20)  << v_memset_time_device[i] << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Copy" << std::endl;
    std::cout << std::setprecision(2) << std::right << std::fixed
              << std::setw(8)  << "SIZE"
              << std::setw(20) << "h=>d"
              << std::setw(20) << "h (pinned)=>d"
              << std::setw(20) << "d=>d" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (size_t i = 0; i < round; i++) {
        std::cout << std::setw(8)  << xlib::human_readable(min_size << i)
                  << std::setw(20) << v_copy_time_host_to_device[i]
                  << std::setw(20) << v_copy_time_host_page_locked_to_device[i]
                  << std::setw(20) << v_copy_time_device_to_device[i] << std::endl;
    }
    std::cout << std::endl;

    constexpr size_t stack_heap_comparison_size = 4 * 1024 * 1024;//4 MB, should not be set to a too large value as this will leads to stack overflow.

    {
        std::array<xlib::byte_t, stack_heap_comparison_size> h_a_stack;
        std::unique_ptr<xlib::byte_t[]> h_p_cpp(nullptr);
        std::unique_ptr<xlib::byte_t[], decltype(my_host_cuda_del)> h_p_cuda(nullptr, my_host_cuda_del);//page-locked
        std::unique_ptr<xlib::byte_t[], decltype(my_device_cuda_del)> d_p_cuda(nullptr, my_device_cuda_del);

        h_p_cpp.reset(new (std::nothrow) xlib::byte_t[stack_heap_comparison_size]);
        ASSERT_NE(h_p_cpp.get(), nullptr);

        xlib::byte_t* p_tmp0 = nullptr;
        hornets_nest::host::allocatePageLocked(p_tmp0, stack_heap_comparison_size);
        h_p_cuda.reset(p_tmp0);

        xlib::byte_t* p_tmp1 = nullptr;
        hornets_nest::gpu::allocate(p_tmp1, stack_heap_comparison_size);
        d_p_cuda.reset(p_tmp1);

        std::cout << "Copy (stack vs heap)" << std::endl;
        std::cout << std::setprecision(2) << std::right << std::fixed
                  << std::setw(8)  << "SIZE"
                  << std::setw(20) << "d=>h (stack)"
                  << std::setw(20) << "d=>h (heap)"
                  << std::setw(20) << "d=>h (heap, pinned)" << std::endl;
        std::cout << std::string(80, '-') << std::endl;

        std::cout << std::setw(8) << xlib::human_readable(stack_heap_comparison_size);

        {
            auto start = std::chrono::high_resolution_clock::now();
            hornets_nest::gpu::copyToHost(d_p_cuda.get(), stack_heap_comparison_size, h_a_stack.data());
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            std::cout << std::setw(20) << diff.count() * 1000.0/* s to ms */;
        }

        {
            auto start = std::chrono::high_resolution_clock::now();
            hornets_nest::gpu::copyToHost(d_p_cuda.get(), stack_heap_comparison_size, h_p_cpp.get());
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            std::cout << std::setw(20) << diff.count() * 1000.0/* s to ms */;
        }

        {
            auto start = std::chrono::high_resolution_clock::now();
            hornets_nest::gpu::copyToHost(d_p_cuda.get(), stack_heap_comparison_size, h_p_cuda.get());
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            std::cout << std::setw(20) << diff.count() * 1000.0/* s to ms */;
        }

        std::cout << std::endl;
    }

    return;
}

class MemOpTest : public HornetTest {
protected:
};


TEST_F(MemOpTest, MemOpBenchmarkTest) {
    exec();
}

