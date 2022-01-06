#include "../include/Player.h"
#include "../include/Board.h"
#include "../include/RandomStrategy.h"
#include "../include/HumanStrategy.h"
#include "../include/MinMaxStrategy.h"
#include "../include/HeuristicMinMaxStrategy.h"
#include <cstdio>
#include <cstdlib>

int test_game() {
    int board_size = 10;
    RandomStrategy strategy{};
    HumanStrategy hmStrategy{};
    MinMaxStrategy mmStrategy{};
    HeuristicMinMaxStrategy hmmStrategy{};

    Board board{board_size};


    int finished;

    Player player1{1, &board, &hmStrategy};
    Player player2{2, &board, &hmmStrategy};
//    board.PlaceChess(1, 5, 5);
//    board.PlaceChess(2, 7, 7);

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