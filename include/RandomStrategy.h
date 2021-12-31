#ifndef GOMOKU_RANDOM_STRATEGY_H
#define GOMOKU_RANDOM_STRATEGY_H

#include "../include/Strategy.h"

class RandomStrategy : public Strategy{
public:
    bool GetStrategy(Board *board, int player_num, int *px, int *py);
};

#endif //GOMOKU_STRATEGY_H
