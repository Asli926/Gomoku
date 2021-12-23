#ifndef GOMOKU_PLAYER_H
#define GOMOKU_PLAYER_H

#include "../include/Board.h"
#include "../include/Strategy.h"

class Player {
private:
    bool is_human_;
    int player_num_;
    Board *pboard_;
    Strategy *strategy_;
public:
    Player(bool is_human, int player_num, Board *board, Strategy *strategy);
    int NextChess();
};

#endif //GOMOKU_PLAYER_H
