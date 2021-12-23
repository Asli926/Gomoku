#include "../include/Player.h"
#include "../include/Board.h"
#include <cstdio>

int test_game() {
    Board board{10};
    Strategy strategy{1};
    Player player1{true, 1, &board, nullptr};
    Player player2{false, 2, &board, &strategy};

    while (true) {
        player1.NextChess();
        board.PrintBoard();
        if (board.IsFinish()) {
            printf("player 1 win!\n");
            break;
        }
        player2.NextChess();
        board.PrintBoard();
        if (board.IsFinish()) {
            printf("player 2 win!\n");
            break;
        }
    }
    return 0;
}