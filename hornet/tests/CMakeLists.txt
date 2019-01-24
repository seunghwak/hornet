cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(HORNET_TESTS LANGUAGES C CXX CUDA)

include(../../compiler-util/CMakeLists.txt)

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - warp (scan & reduce) tests --------------------------------------------------------------------

set(WARP_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/warp/warp_test.cu")
#if(RMM_LIBRARY)
set(RMM_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/rmm/rmm_test.cu")
#endif
set(MEM_OP_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/mem_op/mem_op_test.cu")
set(LOAD_BALANCE_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/load_balance/load_balance_test.cu")

ConfigureTest(WARP_TEST "${WARP_TEST_SRC}" hornet)
#if(RMM_LIBRARY)
ConfigureTest(RMM_TEST "${RMM_TEST_SRC}" hornet)
#endif
ConfigureTest(MEM_OP_TEST "${MEM_OP_TEST_SRC}" hornet)
ConfigureTest(LOAD_BALANCE_TEST "${LOAD_BALANCE_TEST_SRC}" hornet)

#add_executable(hornet_test        test/HornetTest.cu)
#add_executable(hornet_insert_test test/HornetInsertTest.cu)
#add_executable(hornet_delete_test test/HornetDeleteTest.cu)
#add_executable(mem_benchmark      test/MemBenchmark.cu)

#target_link_libraries(hornet_test        hornet)
#target_link_libraries(hornet_insert_test hornet)
#target_link_libraries(hornet_delete_test hornet)
#target_link_libraries(mem_benchmark      hornet)

###################################################################################################
### enable testing ################################################################################
###################################################################################################

enable_testing()

