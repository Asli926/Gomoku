#ifndef GOMOKU_STRATEGY_H
#define GOMOKU_STRATEGY_H

#include "../include/Board.h"

class Strategy {
public:
    virtual bool GetStrategy(Board *board, int player_num, int *px, int *py) = 0;
};

#endif //GOMOKU_STRATEGY_H
