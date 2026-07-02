cmake_minimum_required(VERSION 3.16)
project(arcedge CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

# GPU batched-SSSP backend (Stage 2). OFF by default: this repo must build on
# CPU-only machines; turn on where a CUDA toolkit + device exist (e.g. Colab):
#   cmake -B build -DARCEDGE_CUDA=ON
option(ARCEDGE_CUDA "Build the CUDA batched shortest-path backend" OFF)

add_library(arcedge_core
  src/arcedge/instance.cpp
  src/arcedge/graph.cpp
  src/arcedge/dijkstra.cpp
  src/arcedge/generator.cpp
  src/arcedge/street_graph.cpp
  src/arcedge/design.cpp
  src/arcedge/primal.cpp
  src/arcedge/lagrangian.cpp
)
target_include_directories(arcedge_core PUBLIC src)
target_link_libraries(arcedge_core PUBLIC Threads::Threads)
target_compile_options(arcedge_core PRIVATE -Wall -Wextra)

if(ARCEDGE_CUDA)
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # T4 (Colab) = 75, RTX 3090 = 86, A100 = 80.
    set_property(TARGET arcedge_core PROPERTY CUDA_ARCHITECTURES 75 80 86)
  endif()
  target_sources(arcedge_core PRIVATE src/arcedge/cuda_sssp.cu)
  target_compile_definitions(arcedge_core PUBLIC ARCEDGE_CUDA)
endif()

add_executable(arcedge src/main.cpp)
target_link_libraries(arcedge PRIVATE arcedge_core)

enable_testing()
add_executable(arcedge_tests tests/test_core.cpp)
target_link_libraries(arcedge_tests PRIVATE arcedge_core)
add_test(NAME core COMMAND arcedge_tests)
