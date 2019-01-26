cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(HORNET_TESTS LANGUAGES C CXX CUDA)

include(../../compiler-util/CMakeLists.txt)

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - breadth-first-search tests --------------------------------------------------------------------

set(BFS_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/bfs/bfs_test.cu")
set(CLUSTERING_COEFFICIENT_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/clustering_coefficient/clustering_coefficient_test.cu")
set(CONNECTED_COMPONENT_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/connected_component/connected_component_test.cu")
set(KATZ_CENTRALITY_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/katz_centrality/katz_centrality_test.cu")
set(PAGE_RANK_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/page_rank/page_rank_test.cu")
set(SPMV_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/spmv/spmv_test.cu")
set(SSSP_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/sssp/sssp_test.cu")
set(TRIANGLE_COUNTING_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/triangle_counting/triangle_counting_test.cu")

ConfigureTest(BFS_TEST "${BFS_TEST_SRC}" hornetAlg)
ConfigureTest(CLUSTERING_COEFFICIENT_TEST "${CLUSTERING_COEFFICIENT_TEST_SRC}" hornetAlg)
ConfigureTest(CONNECTED_COMPONENT_TEST "${CONNECTED_COMPONENT_TEST_SRC}" hornetAlg)
ConfigureTest(KATZ_CENTRALITY_TEST "${KATZ_CENTRALITY_TEST_SRC}" hornetAlg)
ConfigureTest(PAGE_RANK_TEST "${PAGE_RANK_TEST_SRC}" hornetAlg)
ConfigureTest(SPMV_TEST "${SPMV_TEST_SRC}" hornetAlg)
ConfigureTest(SSSP_TEST "${SSSP_TEST_SRC}" hornetAlg)
ConfigureTest(TRIANGLE_COUNTING_TEST "${TRIANGLE_COUNTING_TEST_SRC}" hornetAlg)

#add_executable(con-comp   test/CCTest.cu)

#target_link_libraries(con-comp   hornetAlg hornet)

###################################################################################################
### enable testing ################################################################################
###################################################################################################

enable_testing()

