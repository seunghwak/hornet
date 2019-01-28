cmake_minimum_required(VERSION 3.12 FATAL_ERROR) #language level CUDA support was added in CMake 3.8 and the current implementation of Google test requires 3.12+

set(PROJECT "HornetAlg")
project(${PROJECT} LANGUAGES C CXX CUDA)

include(../compiler-util/CMakeLists.txt)

add_subdirectory(../hornet build)

###################################################################################################
# - library targets -------------------------------------------------------------------------------

file(GLOB_RECURSE CU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/*)
file(GLOB_RECURSE CPP_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/externals/xlib/src/*)

add_library(hornetAlg ${CU_SRCS})

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

target_link_libraries(hornetAlg hornet)

###################################################################################################
# - add executables -------------------------------------------------------------------------------

if(BUILD_TESTS)
    add_subdirectory("./tests")
endif(BUILD_TESTS)

