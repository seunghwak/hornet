cmake_minimum_required(VERSION 3.8 FATAL_ERROR) #language level CUDA support was added in CMake 3.8

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
# - add executables -------------------------------------------------------------------------------

if(BUILD_TESTS)
    add_subdirectory("./tests")
endif(BUILD_TESTS)

