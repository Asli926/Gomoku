
#ifndef GOMOKU_PARALLELHEURISTICMINMAX_H
#define GOMOKU_PARALLELHEURISTICMINMAX_H

#include <unordered_map>
#include "../include/Strategy.h"


using needle_list_t = std::vector<std::string>;
using dfa_t = std::vector<int>;
using dfa_list_t = std::vector<dfa_t>;

class ParallelHeuristicMinMax : public Strategy{
private:
    needle_list_t player1_needle_list;
    needle_list_t player2_needle_list;
    dfa_list_t player1_dfa_list;
    dfa_list_t player2_dfa_list;
    std::vector<needle_list_t> player_needle_lists;
    std::vector<dfa_list_t> player_dfa_lists;
    std::vector<int> score_map;
    int total_depth;

    // int UpdateScore(Board& board, int player_num, int score, int r, int c);
    std::array<std::string, 4> GetLinesByChess(Board& board, int r, int c);
    std::pair<int, int> EvalTotalPoints(Board board, int player_num, int cur_depth, int alpha, int beta, int score);
    int EvaluateChessByLines(const std::array<std::string, 4>&, int);
    int EvaluateBoard(Board& board, int player_num);
    std::vector<std::pair<int, int>> HeuristicNextMoves(Board& board, int player_num, int cur_depth);
    static char Int2Char(int);
    static int Diagonal(Board& board, const int& x, const int& y, std::string& s, const int& boardSize);
    static int AntiDiagonal(Board& board, const int& x, const int& y, std::string& s, const int& boardSize);
    int CountPoints(int player_num, const std::string& s, int& level_points);

    int PlaceWrapper(Board& board, int player_num, int r, int c);
    int RevertWrapper(Board& board, int r, int c);

public:
    bool GetStrategy(Board *board, int player_num, int *px, int *py);
    ParallelHeuristicMinMax(int _total_depth=7);
};



#endif //GOMOKU_PARALLELHEURISTICMINMAX_H
