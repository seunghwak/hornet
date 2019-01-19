cmake_minimum_required(VERSION 3.8 FATAL_ERROR) #language level CUDA support was added in CMake 3.8

set(PROJECT "Hornet")
project(${PROJECT} LANGUAGES C CXX CUDA)

include(../compiler-util/CMakeLists.txt)

###################################################################################################
# - library targets -------------------------------------------------------------------------------

if(RMM_LIBRARY)
    add_library(rmm SHARED IMPORTED)
    set_target_properties(rmm PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../externals/rmm/build/librmm.so)
endif()

file(GLOB_RECURSE CPP_SRCS ${CMAKE_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE CU_SRCS  ${CMAKE_SOURCE_DIR}/src/*.cu)
file(GLOB_RECURSE SCPP_SRCS ${CMAKE_SOURCE_DIR}/../xlib/src/*.cpp)
file(GLOB_RECURSE SCU_SRCS  ${CMAKE_SOURCE_DIR}/../xlib/src/*.cu)

add_library(hornet ${CPP_SRCS} ${CU_SRCS} ${SCPP_SRCS} ${SCU_SRCS})

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

if(RMM_LIBRARY)
target_link_libraries(hornet rmm)
endif()
if(USE_NVTX)
target_link_libraries(hornet ${NVTX_LIBRARIES})
endif()

###################################################################################################
# - add executables -------------------------------------------------------------------------------

if(BUILD_TESTS)
    add_subdirectory("./tests")
endif(BUILD_TESTS)

