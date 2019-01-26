###################################################################################################
# - set build options -----------------------------------------------------------------------------

option(CUB_LIBRARY "Configure CMake to use CUB" ON) # if set to off, some test cases using CUB will not compile.
option(RMM_LIBRARY "Configure CMake to use RMM" OFF) # if set to ON, gpu::allocate invokes RMM_ALLOC, if set to OFF, gpu::allocate invokes cuMalloc (which in turn, invokes cudaMalloc)
option(BUILD_TESTS "Configure CMake to build tests" ON)

###################################################################################################
# - cmake modules ---------------------------------------------------------------------------------

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../cmake/Modules/" ${CMAKE_MODULE_PATH})

include(FeatureSummary)
include(CheckIncludeFiles)
include(CheckLibraryExists)

###################################################################################################
# - compiler function -----------------------------------------------------------------------------

function(ConfigureTest CMAKE_TEST_NAME CMAKE_TEST_SRC CMAKE_TEST_LIB)
    add_executable(${CMAKE_TEST_NAME} ${CMAKE_TEST_SRC})
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${CMAKE_TEST_NAME} gmock gtest gmock_main gtest_main pthread ${CMAKE_TEST_LIB})
    set_target_properties(${CMAKE_TEST_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/gtests")
    add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME})
endfunction(ConfigureTest)

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) # no other RAPIDS projects use this

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # if on, save output of compile commands to compile_command.json

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0") # for compatibility with Arrow
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0") # for compatibility with Arrow
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0") # for compatibility with Arrow

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

# set warnings as errors
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")
endif(CMAKE_COMPILER_IS_GNUCXX)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Werror cross-execution-space-call,deprecated-declarations -Xcompiler -Wall,-Werror") # want to add -Werror reorder as well but this generates errors in including cub, and we may better not modify external library

# set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CUB_LIBRARY)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DCUB_WRAPPER")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUB_WRAPPER")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCUB_WRAPPER")
endif()

if(RMM_LIBRARY)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DRMM_WRAPPER")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DRMM_WRAPPER")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DRMM_WRAPPER")
endif()

###################################################################################################
# - Find and add different modules and supporting repos -------------------------------------------

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

###################################################################################################
# - add gtest -------------------------------------------------------------------------------------

if(BUILD_TESTS)
    include(CTest)
    include(ConfigureGoogleTest)

    if(GTEST_FOUND)
        message(STATUS "Google C++ Testing Framework (Google Test) found in ${GTEST_ROOT}")
        include_directories("${GTEST_INCLUDE_DIR}")
        link_directories("${GTEST_LIBRARY_DIR}")
    else()
        message(AUTHOR_WARNING "Google C++ Testing Framework (Google Test) not found: automated tests are disabled.")
    endif(GTEST_FOUND)
endif(BUILD_TESTS)

###################################################################################################
# - include paths ---------------------------------------------------------------------------------

include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
include_directories("${CMAKE_SOURCE_DIR}/../externals/cpp-btree-master")
include_directories("${CMAKE_SOURCE_DIR}/../xlib/include")
include_directories("${CMAKE_SOURCE_DIR}/../primitives")
include_directories("${CMAKE_SOURCE_DIR}/../hornet/include")
include_directories("${CMAKE_SOURCE_DIR}/../hornetsnest/include")

if(CUB_LIBRARY)
    include_directories("${CMAKE_SOURCE_DIR}/../externals/cub-1.8.0")
endif()

if(RMM_LIBRARY)
    include_directories("${CMAKE_SOURCE_DIR}/../externals/rmm/include")
    include_directories("${CMAKE_SOURCE_DIR}/../externals/rmm/src")
    include_directories("${CMAKE_SOURCE_DIR}/../externals/rmm/thirdparty/cnmem/include")
endif()

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}") # CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc

###################################################################################################
# - library targets -------------------------------------------------------------------------------

###################################################################################################
# - build options ---------------------------------------------------------------------------------

option(USE_NVTX "Build with NVTX support" ON)
if(USE_NVTX)
    message(STATUS "Using Nvidia Tools Extension")
    find_library(NVTX_LIBRARY nvToolsExt PATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --define-macro USE_NVTX")
endif(USE_NVTX)

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
endif(CMAKE_BUILD_TYPE MATCHES Debug)

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

###################################################################################################
# - custom targets --------------------------------------------------------------------------------

