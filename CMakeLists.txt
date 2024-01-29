cmake_minimum_required(VERSION 3.5)
project (sshe-worker)

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -pthread -Wall -Wextra -Wl,--no-as-needed -funroll-loops -march=native -flto -pedantic -std=c++17 -fopenmp")

find_package(SCI REQUIRED PATHS "/home/zzh/project/test-GCN/MPC/SCI-SilentOT/build/install" NO_DEFAULT_PATH)
if (NOT SCI_FOUND)
    message(STATUS "SCI was not found")
endif()
# include_directories()

find_package(Eigen3 3.3.3 REQUIRED PATHS "/home/zzh/project/test-GCN/MPC/SCI-SilentOT/deps/build" NO_DEFAULT_PATH)

find_package(troy REQUIRED PATHS "/home/zzh/project/test-GCN/MPC/troy/build/install" NO_DEFAULT_PATH)
if (NOT TROY_FOUND)
    message(STATUS "troy was not found")
endif()

# Paillier encryption dependency
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake/OPHE/cmake")
find_package(NTL REQUIRED)
find_package(GMP REQUIRED)
find_package(GMPXX REQUIRED)
find_package(OPHELib REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP)

# emp two-party GC dependency
find_path(CMAKE_FOLDER NAMES cmake/emp-tool-config.cmake)
include(${CMAKE_FOLDER}/cmake/emp-base.cmake)
include(${CMAKE_FOLDER}/cmake/enable_rdseed.cmake)
# include(${CMAKE_FOLDER}/cmake/enable_float.cmake)

find_package(emp-sh2pc REQUIRED)
include_directories(${EMP-SH2PC_INCLUDE_DIRS})

# include_directories("${CMAKE_SOURCE_DIR}/CoGNN/include/")

include_directories(include)
FILE(GLOB SRCS src/*.cu src/*.cpp)
# set_source_files_properties(${SRCS} PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)
add_library(TaskHandler ${SRCS})
# set_target_properties(TaskHandler PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
# target_compile_definitions(TaskHandler PUBLIC -DSSHEBACKEND -DUSE_FHE)
target_compile_definitions(TaskHandler PUBLIC -DSSHEBACKEND -DUSE_CHEETAH -DSCI_OT)
target_link_libraries(TaskHandler TROY::troy libOTe cryptoTools relic_s ${Boost_LIBRARIES} ${EMP-OT_LIBRARIES} OPHELib::ophelib SCI::SCI-FloatingPoint)

# set(BOOST_ROOT "${CMAKE_SOURCE_DIR}/thirdparty/linux/boost/")
# find_package (Boost COMPONENTS thread system REQUIRED)

# FILE(GLOB_RECURSE TEST_COMM_SRCS ${CMAKE_SOURCE_DIR}/SSHE-Worker/test/*.cpp)
# add_executable(test-comm ${TEST_COMM_SRCS})
# target_compile_definitions(test-comm PRIVATE -DSSHEBACKEND)
# target_include_directories(test-comm PRIVATE "${CMAKE_SOURCE_DIR}/SSHE-Worker/include/")
# target_include_directories(test-comm PRIVATE "${Boost_INCLUDE_DIRS}")
# target_link_libraries(test-comm PRIVATE ${Boost_LIBRARIES} cryptoTools TaskHandler ${EMP-OT_LIBRARIES} OPHELib::ophelib)

FILE(GLOB_RECURSE TEST_COMM_SRCS ${CMAKE_SOURCE_DIR}/SSHE-Worker/test/test.cpp)
add_executable(test-comm ${TEST_COMM_SRCS})
target_compile_definitions(test-comm PUBLIC -DSSHEBACKEND)
include_directories("${CMAKE_SOURCE_DIR}/SSHE-Worker/include/")
target_link_libraries(test-comm TaskHandler)

FILE(GLOB_RECURSE TEST_CLIENT_SRC ${CMAKE_SOURCE_DIR}/SSHE-Worker/test/client_test.cpp)
add_executable(test-client ${TEST_CLIENT_SRC})
target_compile_definitions(test-client PUBLIC -DSSHEBACKEND)
include_directories("${CMAKE_SOURCE_DIR}/SSHE-Worker/include/")
target_link_libraries(test-client TaskHandler)

FILE(GLOB_RECURSE TEST_SERVER_SRC ${CMAKE_SOURCE_DIR}/SSHE-Worker/test/server_test.cpp)
add_executable(test-server ${TEST_SERVER_SRC})
target_compile_definitions(test-server PUBLIC -DSSHEBACKEND)
include_directories("${CMAKE_SOURCE_DIR}/SSHE-Worker/include/")
target_link_libraries(test-server TaskHandler)

# FILE(GLOB_RECURSE TEST_TASK_SRC ${CMAKE_SOURCE_DIR}/SSHE-Worker/test/task_test.cpp)
# add_executable(test-task ${TEST_TASK_SRC})
# target_compile_definitions(test-task PUBLIC -DSSHEBACKEND)
# include_directories("${CMAKE_SOURCE_DIR}/SSHE-Worker/include/")
# target_link_libraries(test-task TaskHandler)

FILE(GLOB_RECURSE TEST_2PC_SRC ${CMAKE_SOURCE_DIR}/SSHE-Worker/test/2PC_test.cpp)
add_executable(test-2PC ${TEST_2PC_SRC})
target_compile_definitions(test-2PC PUBLIC -DSSHEBACKEND)
include_directories("${CMAKE_SOURCE_DIR}/SSHE-Worker/include/")
target_link_libraries(test-2PC TaskHandler)

FILE(GLOB_RECURSE TEST_PLAINTEXT_GCN_SRC ${CMAKE_SOURCE_DIR}/SSHE-Worker/test/plaintext_gcn.cpp)
add_executable(test-plaintext-gcn ${TEST_PLAINTEXT_GCN_SRC})

FILE(GLOB_RECURSE TEST_FED_GCN_SRC ${CMAKE_SOURCE_DIR}/SSHE-Worker/test/fed_gcn.cpp)
add_executable(test-fed-gcn ${TEST_FED_GCN_SRC})

FILE(GLOB_RECURSE TEST_PLAINTEXT_GCN_SMALL_SRC ${CMAKE_SOURCE_DIR}/SSHE-Worker/test/plaintext_gcn_small.cpp)
add_executable(test-plaintext-gcn-small ${TEST_PLAINTEXT_GCN_SMALL_SRC})

FILE(GLOB_RECURSE TEST_WRAPPER_SRC ${CMAKE_SOURCE_DIR}/SSHE-Worker/test/fhe_wrapper_test.cpp)
add_executable(test-fhe-wrapper ${TEST_WRAPPER_SRC})
target_compile_definitions(test-fhe-wrapper PUBLIC -DSSHEBACKEND)
include_directories("${CMAKE_SOURCE_DIR}/SSHE-Worker/include/")
target_link_libraries(test-fhe-wrapper TaskHandler)

FILE(GLOB_RECURSE TEST_GRAPHSC ${CMAKE_SOURCE_DIR}/SSHE-Worker/test/GraphAnalysis/*.cpp ${CMAKE_SOURCE_DIR}/SSHE-Worker/test/graphsc_test.cpp)
add_executable(test-graphsc ${TEST_GRAPHSC})
target_compile_definitions(test-graphsc PUBLIC -DSSHEBACKEND)
include_directories("${CMAKE_SOURCE_DIR}/SSHE-Worker/include/")
include_directories("${CMAKE_SOURCE_DIR}/SSHE-Worker/test/GraphAnalysis/")
target_link_libraries(test-graphsc TaskHandler)