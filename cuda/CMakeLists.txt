cmake_minimum_required(VERSION 3.12.0)
project(gpu)

find_package(CUDA)

SET(CUDA_NVCC_FLAGS -O3;-G;)

FILE(GLOB_RECURSE CURRENT_HEADERS *.h *.cuh *.hpp)
FILE(GLOB CURRENT_SOURCES *.cpp *.cu)

source_group("Include" FILES ${CURRENT_HEADERS})
source_group("Source" FILES ${CURRENT_SOURCES})
cuda_add_library(gpu STATIC ${CURRENT_HEADERS} ${CURRENT_SOURCES})

