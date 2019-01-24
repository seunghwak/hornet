cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(HORNET_TESTS LANGUAGES C CXX CUDA)

include(../../compiler-util/CMakeLists.txt)

###################################################################################################
### test sources ##################################################################################
###################################################################################################

###################################################################################################
# - breadth-first-search tests --------------------------------------------------------------------

set(BFS_TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/bfs/bfs_test.cu")

ConfigureTest(BFS_TEST "${BFS_TEST_SRC}" hornetAlg)

#add_executable(bfs        test/BFSTest.cu)
#add_executable(bfs2       test/BFSTest2.cu)
#add_executable(con-comp   test/CCTest.cu)
#add_executable(spmv       test/SpMVTest.cu)
#add_executable(sssp       test/SSSPTest.cu)
#add_executable(katz       test/KatzTest.cu)
#add_executable(dyn-katz   test/KatzDynamicTest.cu)
#add_executable(triangle   test/TriangleTest.cu)
#add_executable(triangle2  test/TriangleTest2.cu)
#add_executable(clus-coeff test/ClusCoeffTest.cu)
#add_executable(pr         test/PageRankTest.cu)

#target_link_libraries(bfs        hornetAlg hornet)
#target_link_libraries(bfs2       hornetAlg hornet)
#target_link_libraries(con-comp   hornetAlg hornet)
#target_link_libraries(spmv       hornetAlg hornet)
#target_link_libraries(sssp       hornetAlg hornet)
#target_link_libraries(katz       hornetAlg hornet)
#target_link_libraries(dyn-katz   hornetAlg hornet)
#target_link_libraries(triangle   hornetAlg hornet)
#target_link_libraries(triangle2  hornetAlg hornet)
#target_link_libraries(clus-coeff hornetAlg hornet)
#target_link_libraries(pr         hornetAlg hornet)

###################################################################################################
### enable testing ################################################################################
###################################################################################################

enable_testing()

