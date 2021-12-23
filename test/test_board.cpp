#include <cassert>
#include "../include/Board.h"

int test_board() {
    Board board{10};
    board.PrintBoard();
    board.PlaceChess(1, 4,5);
    board.PrintBoard();
    board.PlaceChess(1, 4,6);
    board.PrintBoard();
    board.PlaceChess(1, 4,7);
    board.PrintBoard();
    board.PlaceChess(1, 4,8);
    board.PrintBoard();
    board.PlaceChess(1, 4,9);
    board.PrintBoard();
    assert(board.IsFinish() == 1);
    return 0;
}