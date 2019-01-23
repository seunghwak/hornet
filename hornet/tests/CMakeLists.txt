cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(HORNET_TESTS LANGUAGES C CXX CUDA)

include(../../compiler-util/CMakeLists.txt)

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - warp (scan & reduce) tests --------------------------------------------------------------------

set(WARP_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/warp/warp_google_test.cu")
set(MEM_OP_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/mem_op/mem_op_google_test.cu")
#if(RMM_LIBRARY)
set(RMM_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/rmm/rmm_google_test.cu")
#endif

ConfigureTest(WARP_TEST "${WARP_TEST_SRC}")
ConfigureTest(MEM_OP_TEST "${MEM_OP_TEST_SRC}")
#if(RMM_LIBRARY)
ConfigureTest(RMM_TEST "${RMM_TEST_SRC}")
#endif

add_executable(warp_test "${CMAKE_CURRENT_SOURCE_DIR}/warp/warp_cmdline_test.cu")
add_executable(mem_op_test "${CMAKE_CURRENT_SOURCE_DIR}/mem_op/mem_op_cmdline_test.cu")
#if(RMM_LIBRARY)
add_executable(rmm_test "${CMAKE_CURRENT_SOURCE_DIR}/rmm/rmm_cmdline_test.cu")
#endif

target_link_libraries(warp_test hornet)
target_link_libraries(mem_op_test hornet)
#if(RMM_LIBRARY)
target_link_libraries(rmm_test hornet)
#endif

#add_executable(lb_test            test/BinarySearchTest.cu)
#add_executable(hornet_test        test/HornetTest.cu)
#add_executable(hornet_insert_test test/HornetInsertTest.cu)
#add_executable(hornet_delete_test test/HornetDeleteTest.cu)
#add_executable(mem_benchmark      test/MemBenchmark.cu)

#target_link_libraries(lb_test            hornet)
#target_link_libraries(hornet_test        hornet)
#target_link_libraries(hornet_insert_test hornet)
#target_link_libraries(hornet_delete_test hornet)
#target_link_libraries(mem_benchmark      hornet)

###################################################################################################
### enable testing ################################################################################
###################################################################################################

enable_testing()

