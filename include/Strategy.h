#ifndef GOMOKU_STRATEGY_H
#define GOMOKU_STRATEGY_H

#include "../include/Board.h"

class Strategy {
private:
    int strategy_mode_;
    bool Strategy1_(Board *board, int player_num, int *px, int *py);
public:
    Strategy(int strategy_mode);
    bool GetStrategy(Board *board, int player_num, int *px, int *py);
};

#endif //GOMOKU_STRATEGY_H
