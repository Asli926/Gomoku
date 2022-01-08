cmake_minimum_required(VERSION 3.12.0)
project(Gomoku)

set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_FLAGS_DEBUG "-O3")

include_directories(include)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(Gomoku
        include/Board.h
        src/Board.cpp
        test/test_board.cpp test/test_search.cpp src/Player.cpp test/test_game.cpp test/test_main.cpp src/RandomStrategy.cpp src/HumanStrategy.cpp src/MinMaxStrategy.cpp test/test_score.cpp src/HeuristicMinMaxStrategy.cpp utils/pattern_search.cpp src/ParallelHeuristicMinMax.cpp)
