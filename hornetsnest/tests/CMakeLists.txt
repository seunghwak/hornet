cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(HORNET_TESTS LANGUAGES C CXX CUDA)

include(../../compiler-util/CMakeLists.txt)

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - breadth-first-search tests --------------------------------------------------------------------

set(BFS_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/bfs/bfs_test.cu")
set(PAGE_RANK_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/page_rank/page_rank_test.cu")
set(SPMV_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/spmv/spmv_test.cu")
set(SSSP_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/sssp/sssp_test.cu")
set(TRIANGLE_COUNTING_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/triangle_counting/triangle_counting_test.cu")

ConfigureTest(BFS_TEST "${BFS_TEST_SRC}" hornetAlg)
ConfigureTest(PAGE_RANK_TEST "${PAGE_RANK_TEST_SRC}" hornetAlg)
ConfigureTest(SPMV_TEST "${SPMV_TEST_SRC}" hornetAlg)
ConfigureTest(SSSP_TEST "${SSSP_TEST_SRC}" hornetAlg)
ConfigureTest(TRIANGLE_COUNTING_TEST "${TRIANGLE_COUNTING_TEST_SRC}" hornetAlg)

#add_executable(con-comp   test/CCTest.cu)
#add_executable(spmv       test/SpMVTest.cu)
#add_executable(sssp       test/SSSPTest.cu)
#add_executable(katz       test/KatzTest.cu)
#add_executable(dyn-katz   test/KatzDynamicTest.cu)
#add_executable(clus-coeff test/ClusCoeffTest.cu)
#add_executable(pr         test/PageRankTest.cu)

#target_link_libraries(con-comp   hornetAlg hornet)
#target_link_libraries(spmv       hornetAlg hornet)
#target_link_libraries(sssp       hornetAlg hornet)
#target_link_libraries(katz       hornetAlg hornet)
#target_link_libraries(dyn-katz   hornetAlg hornet)
#target_link_libraries(clus-coeff hornetAlg hornet)
#target_link_libraries(pr         hornetAlg hornet)

###################################################################################################
### enable testing ################################################################################
###################################################################################################

enable_testing()

