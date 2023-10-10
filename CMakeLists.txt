cmake_minimum_required(VERSION 3.0)
project(MMD)

set(CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_SOURCE_DIR}/include/Eigen)

add_executable(MMD src/main.cpp)
