cmake_minimum_required(VERSION 3.20)
project(edgelstm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Header-only template library placeholder.
add_library(edgelstm INTERFACE)
target_include_directories(edgelstm INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
