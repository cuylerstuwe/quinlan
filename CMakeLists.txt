cmake_minimum_required(VERSION 3.18)
project(quinlan LANGUAGES C CXX CUDA)

# The vendored engine has no FetchContent, ExternalProject, or submodule step.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
add_subdirectory(src/quinlan/_vendor/qwentin engine)
