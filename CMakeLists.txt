cmake_minimum_required(VERSION 3.20)
project(edge_lstm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-only template library placeholder.
add_library(edge_lstm INTERFACE)
target_include_directories(edge_lstm INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
