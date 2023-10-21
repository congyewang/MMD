cmake_minimum_required(VERSION 3.0)
project(MMD)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

add_definitions(-DEIGEN_NO_DEBUG)

include_directories(${CMAKE_SOURCE_DIR}/include/Eigen)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mkl")
endif()

add_executable(MMD src/single.cpp)

target_link_libraries(MMD mkl_intel_lp64 mkl_sequential mkl_core)
