#ifndef GOMOKU_HEURISTICMINMAXSTRATEGY_H
#define GOMOKU_HEURISTICMINMAXSTRATEGY_H

#include <unordered_map>
// #include <regex>
#include <boost/regex.hpp>
#include "../include/Strategy.h"


using needle_list_t = std::vector<boost::regex>;

class HeuristicMinMaxStrategy : public Strategy{
private:
    needle_list_t player1_needle_list;
    needle_list_t player2_needle_list;
    std::vector<needle_list_t> player_needle_lists;
    std::vector<int> score_map;
    int total_depth;

    // random table for hash
    std::vector<unsigned int> player1_rd_tb;
    std::vector<unsigned int> player2_rd_tb;
    std::unordered_map<unsigned int, int> board_score;

    // int UpdateScore(Board& board, int player_num, int score, int r, int c);
    std::array<std::string, 4> GetLinesByChess(Board& board, int r, int c);
    std::pair<int, int> EvalTotalPoints(Board board, int player_num, int cur_depth, int alpha, int beta, int score);
    int EvaluateChessByLines(const std::array<std::string, 4>&, int);
    int EvaluateBoard(Board& board, int player_num);
    std::vector<std::pair<int, int>> HeuristicNextMoves(Board& board, int player_num, bool max_layer);
    static char Int2Char(int);
    static int Diagonal(Board& board, const int& x, const int& y, std::string& s, const int& boardSize);
    static int AntiDiagonal(Board& board, const int& x, const int& y, std::string& s, const int& boardSize);
    int CountPoints(const needle_list_t & player_needle_list, const std::string& s, int& level_points);

    int PlaceWrapper(Board& board, int player_num, int r, int c);
    int RevertWrapper(Board& board, int r, int c);

public:
    bool GetStrategy(Board *board, int player_num, int *px, int *py);
    int SetBoardSize(int);
    HeuristicMinMaxStrategy(int _total_depth=6);
};

#endif //GOMOKU_HEURISTICMINMAXSTRATEGY_H
