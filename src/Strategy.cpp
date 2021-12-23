#include "../include/Strategy.h"
#include <random>

Strategy::Strategy(int strategy_mode) {
    strategy_mode_ = strategy_mode;
}

bool Strategy::GetStrategy(Board *board, int player_num, int *px, int *py) {
    switch (strategy_mode_) {
        case 1:
            Strategy1_(board, player_num, px, py);
        default:
            return false;
    }
    return true;
}

bool Strategy::Strategy1_(Board *board, int player_num, int *px, int *py) {
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distrib(0, board->GetSize() - 1);
    int x, y;
    int max_try = 200, i = 0;
    while(i ++ < max_try) {
        x = distrib(gen);
        y = distrib(gen);
        if (board->GetChess(x, y) == 0) {
            *px = x; *py = y;
            return true;
        }
    }
    return false;
}