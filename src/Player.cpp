#include <cstdio>
#include <random>
#include "../include/Player.h"

Player::Player(bool is_human, int player_num, Board *board, Strategy *strategy) {
    is_human_ = is_human;
    player_num_ = player_num;
    pboard_ = board;
    strategy_ = strategy;
}

int Player::NextChess() {
    int x, y;
    int state;

    if (is_human_) {
        while (true) {
            printf("Please input the valid coordinates of next chess. (row col)\n");
            scanf("%d %d", &x, &y);
            state = pboard_->PlaceChess(player_num_, x, y);
            if (state == 0) break;
        }
    } else {
        strategy_->GetStrategy(pboard_, player_num_, &x, &y);
        pboard_->PlaceChess(player_num_, x, y);
    }

    return 0;
}