#ifndef GOMOKU_HEURISTICMINMAXSTRATEGY_H
#define GOMOKU_HEURISTICMINMAXSTRATEGY_H

#include <unordered_map>
// #include <regex>
//#include <boost/regex.hpp>
#include "../include/Strategy.h"


using needle_list_t = std::vector<std::string>;
using dfa_t = int*;
using dfa_list_t = dfa_t*;

class HeuristicMinMaxStrategy : public Strategy{
private:
    int* needle_size_list;
    needle_list_t player1_needle_list;
    needle_list_t player2_needle_list;
    dfa_list_t player1_dfa_list;
    dfa_list_t player2_dfa_list;
    std::vector<needle_list_t> player_needle_lists;
    std::vector<dfa_list_t> player_dfa_lists;
    int* score_map;
    int total_depth;

    char** c_needle_list_two;
    int** c_dfas_two;

    // int UpdateScore(Board& board, int player_num, int score, int r, int c);
    int GetLinesByChess(Board& board, int r, int c, char* lines, int* line_sizes);
    std::pair<int, int> EvalTotalPoints(Board board, int player_num, int cur_depth, int alpha, int beta, int score);
    int EvaluateChessScoreByLinesGPU(char* c_lines, int* c_line_size, int);
    int EvaluateChessByLines(const std::array<std::string, 4>&, int);
    int EvaluateBoard(Board& board, int player_num);
    std::vector<std::pair<int, int>> HeuristicNextMoves(Board& board, int player_num, bool max_layer);
    static char Int2Char(int);
    static int Diagonal(Board& board, const int& x, const int& y, std::string& s, const int& boardSize);
    static int AntiDiagonal(Board& board, const int& x, const int& y, std::string& s, const int& boardSize);
    int CountPoints(int player_num, const std::string& s, int& level_points);

public:
    bool GetStrategy(Board *board, int player_num, int *px, int *py);
    ~HeuristicMinMaxStrategy();
    HeuristicMinMaxStrategy(int _total_depth=4);
};

#endif //GOMOKU_HEURISTICMINMAXSTRATEGY_H
