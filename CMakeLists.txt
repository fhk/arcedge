cmake_minimum_required(VERSION 3.16)
project(arcedge CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

add_library(arcedge_core
  src/arcedge/instance.cpp
  src/arcedge/graph.cpp
  src/arcedge/dijkstra.cpp
  src/arcedge/generator.cpp
  src/arcedge/primal.cpp
  src/arcedge/lagrangian.cpp
)
target_include_directories(arcedge_core PUBLIC src)
target_link_libraries(arcedge_core PUBLIC Threads::Threads)
target_compile_options(arcedge_core PRIVATE -Wall -Wextra)

add_executable(arcedge src/main.cpp)
target_link_libraries(arcedge PRIVATE arcedge_core)

enable_testing()
add_executable(arcedge_tests tests/test_core.cpp)
target_link_libraries(arcedge_tests PRIVATE arcedge_core)
add_test(NAME core COMMAND arcedge_tests)
