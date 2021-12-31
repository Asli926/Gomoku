
#ifndef GOMOKU_MINMAXSTRATEGY_H
#define GOMOKU_MINMAXSTRATEGY_H

#include "../include/Strategy.h"
#include <unordered_map>
#include <vector>
#include <regex>

enum chess_pattern_t {ONE = 0, TWO, THREE, FOUR, FIVE,
            BLOCKED_TWO, BLOCKED_THREE, BLOCKED_FOUR};
using needle_map_t = std::unordered_map<chess_pattern_t, std::regex>;

class MinMaxStrategy : public Strategy{
private:
//    int EvalLevelPoints(Board& board, int player_num);
    std::pair<int, int> EvalTotalPoints(Board board, int player_num, int cur_depth, int alpha, int beta);
    static char Int2Char(int);
    static int Diagonal(Board& board, const int& x, const int& y, std::string& s, const int& boardSize);
    static int AntiDiagonal(Board& board, const int& x, const int& y, std::string& s, const int& boardSize);
    int CountPoints(const needle_map_t& player_needle_map, const std::string& s, int& level_points);
    needle_map_t player1_needle_map;
    needle_map_t player2_needle_map;
    std::unordered_map<chess_pattern_t, int> score_map;
    int total_depth;
public:
    int EvalLevelPoints(Board& board, int player_num);
    bool GetStrategy(Board *board, int player_num, int *px, int *py);
    MinMaxStrategy(int _total_depth=4);
};

#endif //GOMOKU_MINMAXSTRATEGY_H
