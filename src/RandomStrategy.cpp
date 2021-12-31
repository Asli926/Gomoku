#include "../include/RandomStrategy.h"
#include <random>

bool RandomStrategy::GetStrategy(Board *board, int player_num, int *px, int *py) {
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