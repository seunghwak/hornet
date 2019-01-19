cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(HORNET_TESTS LANGUAGES C CXX CUDA)

include(../../compiler-util/CMakeLists.txt)

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - warp (scan & reduce) tests --------------------------------------------------------------------

set(WARP_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/warp_test.cu")

ConfigureTest(WARP_TEST "${WARP_TEST_SRC}")

#add_executable(lb_test            test/BinarySearchTest.cu)
#add_executable(ptxtest            test/ptxtest.cu)
#add_executable(hornet_test        test/HornetTest.cu)
#add_executable(hornet_insert_test test/HornetInsertTest.cu)
#add_executable(hornet_delete_test test/HornetDeleteTest.cu)
#add_executable(mem_benchmark      test/MemBenchmark.cu)
#if(RMM_LIBRARY)
#add_executable(rmm_benchmark      test/RMMBenchmark.cu)
#endif()

#target_link_libraries(lb_test            hornet)
#target_link_libraries(ptxtest            hornet)
#target_link_libraries(hornet_test        hornet)
#target_link_libraries(hornet_insert_test hornet)
#target_link_libraries(hornet_delete_test hornet)
#target_link_libraries(mem_benchmark      hornet)
#if(RMM_LIBRARY)
#target_link_libraries(rmm_benchmark      hornet)
#endif()

###################################################################################################
### enable testing ################################################################################
###################################################################################################

enable_testing()

