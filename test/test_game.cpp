#include "../include/Player.h"
#include "../include/Board.h"
#include "../include/RandomStrategy.h"
#include "../include/HumanStrategy.h"
#include "../include/MinMaxStrategy.h"
#include "../include/HeuristicMinMaxStrategy.h"
#include "../include/ParallelHeuristicMinMax.h"
#include <cstdio>
#include <cstdlib>
#include <omp.h>

int test_game() {
    int board_size = 15;
    RandomStrategy strategy{};
    HumanStrategy hmStrategy{};
    MinMaxStrategy mmStrategy{};
    HeuristicMinMaxStrategy hmmStrategy{};
    ParallelHeuristicMinMax phmmStrategy{};

    Board board{board_size};


    int finished;

    Player player1{1, &board, &hmStrategy};
    Player player2{2, &board, &phmmStrategy};
    omp_set_nested(1);

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
        double start = omp_get_wtime();
        if (player2.NextChess() != 0) printf("player2 x or y wrong!\n");
        double end = omp_get_wtime();
        double time = end - start;
        printf("time: %.5f \n", time);

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