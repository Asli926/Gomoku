#include "../include/Player.h"
#include "../include/Board.h"
#include "../include/RandomStrategy.h"
#include "../include/HumanStrategy.h"
#include "../include/MinMaxStrategy.h"
#include "../include/HeuristicMinMaxStrategy.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>

int test_game() {
    std::cout << "LOADING..." << std::endl;
    int board_size = 15;
    HumanStrategy hmStrategy{};
    HeuristicMinMaxStrategy hmmStrategy{};

    Board board{board_size};


    int finished;

    Player player1{1, &board, &hmStrategy};
    Player player2{2, &board, &hmmStrategy};

    std::cout << "START!" << std::endl;
    while (true) {
        if (player1.NextChess() != 0) printf("player1 x or y wrong!\n");
        board.PrintBoard();
        finished = board.IsFinish();
        if (finished == 1) {
            printf("player 1 win!\n");
            break;
        } else if (finished == 2) {
            printf("the board is full, teams are tying!\n");
            break;
        }
        if (player2.NextChess() != 0) printf("player2 x or y wrong!\n");
        board.PrintBoard();
        finished = board.IsFinish();
        if (finished == 1) {
            printf("player 2 win!\n");
            break;
        } else if (finished == 2) {
            printf("the board is full, teams are tying!\n");
            break;
        }
    }
    return 0;
}