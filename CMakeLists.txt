cmake_minimum_required(VERSION 3.17)
project(Gomoku)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)

add_executable(Gomoku
        include/Board.h
        src/Board.cpp
        test/test_board.cpp src/Player.cpp src/Strategy.cpp test/test_game.cpp test/test_main.cpp)
