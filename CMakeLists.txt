cmake_minimum_required(VERSION 3.12.0)
project(Gomoku)

SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_FLAGS_DEBUG "-O3")

include_directories(include)
add_subdirectory(cuda)
SET(EXTRA_LIBS ${EXTRA_LIBS} gpu)

add_executable(Gomoku
        include/Board.h
        src/Board.cpp
        test/test_board.cpp test/test_search.cpp src/Player.cpp test/test_game.cpp test/test_main.cpp src/RandomStrategy.cpp src/HumanStrategy.cpp src/MinMaxStrategy.cpp test/test_score.cpp src/HeuristicMinMaxStrategy.cpp utils/pattern_search.cpp)

target_link_libraries(Gomoku ${EXTRA_LIBS})