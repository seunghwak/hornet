cmake_minimum_required(VERSION 3.12 FATAL_ERROR) #language level CUDA support was added in CMake 3.8 and the current implementation of Google test requires 3.12+

project(HORNET_TESTS LANGUAGES C CXX CUDA)

include(../../compiler-util/CMakeLists.txt)

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - breadth-first-search tests --------------------------------------------------------------------

set(BC_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/bc/bc_test.cu")
ConfigureTest(BC_TEST "${BC_TEST_SRC}" hornetAlg)

###################################################################################################
# - betweenness-centrality tests ------------------------------------------------------------------

set(BFS_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/bfs/bfs_test.cu")
ConfigureTest(BFS_TEST "${BFS_TEST_SRC}" hornetAlg)

###################################################################################################
# - clustering-coefficient tests ------------------------------------------------------------------

set(CLUSTERING_COEFFICIENT_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/clustering_coefficient/clustering_coefficient_test.cu")
ConfigureTest(CLUSTERING_COEFFICIENT_TEST "${CLUSTERING_COEFFICIENT_TEST_SRC}" hornetAlg)

###################################################################################################
# - connected-component tests ---------------------------------------------------------------------

set(CONNECTED_COMPONENT_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/connected_component/connected_component_test.cu")
ConfigureTest(CONNECTED_COMPONENT_TEST "${CONNECTED_COMPONENT_TEST_SRC}" hornetAlg)

###################################################################################################
# - katz-centrality tests -------------------------------------------------------------------------

set(KATZ_CENTRALITY_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/katz_centrality/katz_centrality_test.cu")
ConfigureTest(KATZ_CENTRALITY_TEST "${KATZ_CENTRALITY_TEST_SRC}" hornetAlg)

###################################################################################################
# - page-rank tests -------------------------------------------------------------------------------

set(PAGE_RANK_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/page_rank/page_rank_test.cu")
ConfigureTest(PAGE_RANK_TEST "${PAGE_RANK_TEST_SRC}" hornetAlg)

###################################################################################################
# - sparse-matrix-vector-multiplication tests -----------------------------------------------------

set(SPMV_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/spmv/spmv_test.cu")
ConfigureTest(SPMV_TEST "${SPMV_TEST_SRC}" hornetAlg)

###################################################################################################
# - single-source-shortest-path tests -------------------------------------------------------------

set(SSSP_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/sssp/sssp_test.cu")
ConfigureTest(SSSP_TEST "${SSSP_TEST_SRC}" hornetAlg)

###################################################################################################
# - triangle-counting tests -----------------------------------------------------

set(TRIANGLE_COUNTING_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/triangle_counting/triangle_counting_test.cu")
ConfigureTest(TRIANGLE_COUNTING_TEST "${TRIANGLE_COUNTING_TEST_SRC}" hornetAlg)

###################################################################################################
### enable testing ################################################################################
###################################################################################################

enable_testing()

