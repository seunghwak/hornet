#pragma once

#include <gtest/gtest.h>

#if defined(RMM_WRAPPER)
#include <rmm.h>
#endif

// Base class fixture for Hornet google tests that initializes / finalizes the memory manager
class HornetTest : public ::testing::Test {
protected:
    static void SetUpTestCase() {
#if defined(RMM_WRAPPER)
        hornets_nest::gpu::initializeRMMPoolAllocation();//update initPoolSize if you know your memory requirement and memory availability in your system, if initial pool size is set to 0 (default value), RMM currently assigns half the device memory.
#endif
        return;
    }

    static void TearDownTestCase() {
#if defined(RMM_WRAPPER)
        hornets_nest::gpu::finalizeRMMPoolAllocation();
#endif
        return;
    }
};

// Base class fixture for Hornet google tests that initializes / finalizes the memory manager
template <typename T>
class HornetTestWithParam : public ::testing::TestWithParam<T> {
public://needs to be public to be used in TEST_P
    static void SetUpTestCase() {
#if defined(RMM_WRAPPER)
        hornets_nest::gpu::initializeRMMPoolAllocation();//update initPoolSize if you know your memory requirement and memory availability in your system, if initial pool size is set to 0 (default value), RMM currently assigns half the device memory.
#endif
        return;
    }

    static void TearDownTestCase() {
#if defined(RMM_WRAPPER)
        hornets_nest::gpu::finalizeRMMPoolAllocation();
#endif
        return;
    }
};

