#include <iostream>
#include "../include/Board.h"
#include "../include/MinMaxStrategy.h"

int test_score() {
    /* test ok */
    /*
    Board board{10};
    MinMaxStrategy mmStrategy{};
    board.PlaceChess(1, 5, 5);
    board.PlaceChess(2, 7, 7);
    board.PlaceChess(1, 9, 9);
    board.PrintBoard();
    int score = mmStrategy.EvalLevelPoints(board, 1);
    std::cout << score << std::endl;
    */

    /* test ok */
    /*
    Board board2{10};
    MinMaxStrategy mmStrategy2{};
    board2.PlaceChess(1, 5, 5);
    board2.PlaceChess(2, 7, 7);
    board2.PlaceChess(1, 5, 3);
    board2.PlaceChess(2, 7, 8);
    board2.PlaceChess(1, 4, 3);
    board2.PrintBoard();
    int score2 = mmStrategy2.EvalLevelPoints(board2, 1);
    std::cout << score2 << std::endl;
    */

    /* test ok */
    /*
    Board board2{10};
    MinMaxStrategy mmStrategy2{};
    board2.PlaceChess(1, 5, 5);
    board2.PlaceChess(2, 5, 6);
    board2.PlaceChess(1, 5, 3);
    board2.PlaceChess(2, 7, 8);
    board2.PlaceChess(1, 4, 3);
    board2.PlaceChess(1, 5, 4);
    board2.PlaceChess(1, 6, 3);
    board2.PrintBoard();
    int score2 = mmStrategy2.EvalLevelPoints(board2, 1);
    std::cout << score2 << std::endl;
     */

    /* test ok */
//    Board board2{10};
//    MinMaxStrategy mmStrategy2{};
//    board2.PlaceChess(2, 7, 7);
//    board2.PlaceChess(2, 9, 8);
//    board2.PlaceChess(1, 5, 4);
//    board2.PlaceChess(1, 5, 5);
//    board2.PrintBoard();
//    int score2 = mmStrategy2.EvalLevelPoints(board2, 2);
//    std::cout << score2 << std::endl;

    Board board{10};
    MinMaxStrategy mmStrategy{};
    board.PlaceChess(2, 7, 7);
    board.PlaceChess(2, 3, 2);
    board.PlaceChess(1, 5, 4);
    board.PlaceChess(1, 5, 5);
    board.PrintBoard();
    int score = mmStrategy.EvalLevelPoints(board, 2);
    std::cout << score << std::endl;
    return 0;
}

int test_reg() {
    auto pattern = std::regex{"010"};
    auto s = std::string{"01010"};
    auto words_begin = std::sregex_iterator(s.begin(), s.end(), pattern);
    auto words_end = std::sregex_iterator();
    auto level_points = (int)std::distance(words_begin, words_end);
    return 0;
}