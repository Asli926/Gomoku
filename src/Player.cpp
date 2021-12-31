#include <cstdio>
#include <random>
#include "../include/Player.h"

Player::Player(int player_num, Board *board, Strategy *strategy) {
    player_num_ = player_num;
    pboard_ = board;
    strategy_ = strategy;
}

int Player::NextChess() {
    int x, y;

    strategy_->GetStrategy(pboard_, player_num_, &x, &y);

    return pboard_->PlaceChess(player_num_, x, y);
}