//
// Created by 李艾珊 on 2021/12/24.
//

#ifndef GOMOKU_HUMANSTRATEGY_H
#define GOMOKU_HUMANSTRATEGY_H

#include "../include/Strategy.h"

class HumanStrategy : public Strategy{
public:
    bool GetStrategy(Board *board, int player_num, int *px, int *py);
};

#endif //GOMOKU_HUMANSTRATEGY_H
