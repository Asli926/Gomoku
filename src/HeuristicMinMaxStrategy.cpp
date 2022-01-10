#include "../include/HeuristicMinMaxStrategy.h"
#include "../include/pattern_search.h"
#include <string>
#include <array>
#include <functional>
#include <climits>
#include <algorithm>
#include <iostream>
#include <cstring>
#include "../cuda/gpu_match.cuh"

// extern "C"
// int match_count_multiple(char** lines, char** patterns, int** dfas, int* pattern_size, int* line_size, int* score_map);

HeuristicMinMaxStrategy::HeuristicMinMaxStrategy(int _total_depth) {
    std::cout << "HeuristicMinMaxStrategy: Construction" << std::endl;
    needle_size_list = new int[16];
    needle_size_list[0] = 5; needle_size_list[1] = 6; needle_size_list[2] = 6; needle_size_list[3] = 6; needle_size_list[4] = 6;
    needle_size_list[5] = 6; needle_size_list[6] = 5; needle_size_list[7] = 5; needle_size_list[8] = 5; needle_size_list[9] = 5;
    needle_size_list[10] = 5; needle_size_list[11] = 6; needle_size_list[12] = 6; needle_size_list[13] = 6; needle_size_list[14] = 6;
    needle_size_list[15] = 6;

    player1_needle_list = needle_list_t{
        "11111", "011110", "011100", "001110", "011010",
        "010110", "11110", "01111", "11011", "10111",
        "11101", "001100", "001010", "010100", "000100", "001000"
    };

    player1_dfa_list = new dfa_t[16];
    for (int k = 0; k < 16; k ++) player1_dfa_list[k] = construct_nxt(player1_needle_list[k]);

    player2_needle_list = needle_list_t{
            "22222", "022220", "022200", "002220", "012020",
            "020220", "22220", "02222", "22022", "20222",
            "22202", "002200", "002020", "020200", "000200", "002000"
    };

    player2_dfa_list = new dfa_t[16];
    for (int k = 0; k < 16; k ++) player2_dfa_list[k] = construct_nxt(player2_needle_list[k]);

    player_needle_lists = std::vector<needle_list_t>{player1_needle_list, player2_needle_list};
    player_dfa_lists = std::vector<dfa_list_t>{player1_dfa_list, player2_dfa_list};

    score_map = new int[16];
    score_map[0] = 50000; score_map[1] = 4320; score_map[2] = 720; score_map[3] = 720; score_map[4] = 720;
    score_map[5] = 720; score_map[6] = 720; score_map[7] = 720; score_map[8] = 720; score_map[9] = 720;
    score_map[10] = 720; score_map[11] = 120; score_map[12] = 120; score_map[13] = 120; score_map[14] = 20;
    score_map[15] = 20;

    total_depth = _total_depth;
}

/* only translate 0 - 9 to its char type */
inline char HeuristicMinMaxStrategy::Int2Char(int num) {
    return (char) (num - 0 + (int)('0'));
}

std::array<std::string, 4> HeuristicMinMaxStrategy::GetLinesByChess(Board& board, int r, int c) {
    int board_size = board.GetSize();
    std::array<std::string, 4> lines;
    for (int i = 0; i < 4; i ++) lines[i].reserve(board_size);


    for (int i = 0; i < board_size; i ++) {
        lines[0] += Int2Char(board.GetChess(i, c));
    }

    for (int j = 0; j < board_size; j ++) {
        lines[1] += Int2Char(board.GetChess(r, j));
    }

    int min_rc = std::min(r, c);
    for (int i = r - min_rc, j = c - min_rc;
    i < board_size && j < board_size;
    i ++, j ++) {
        lines[2] += Int2Char(board.GetChess(i, j));
    }

    int dr = std::min(r, board_size - 1 - c);
    for (int i = r - dr, j = c + dr; i < board_size && j >= 0; i ++, j --) {
        lines[3] += Int2Char(board.GetChess(i, j));
    }

    return lines;
}

int HeuristicMinMaxStrategy::EvaluateChessByLinesGPU(const std::array<std::string, 4>& lines, int player_num) {
    std::cout << "EvaluateChessByLinesGPU: cuda function wrapper" << std::endl;
    char c_lines[4 * 20];
    char c_needle_list[16 * 6];
    int c_line_size[4];
    int c_dfas[16 * 7];

    for (int k = 0; k < 4; k ++) {
        memcpy((void*) &c_lines[20 * k], lines[k].c_str(), lines[k].size());
        c_line_size[k] = lines[k].size();
    }

    for (int k = 0; k < 16; k ++) {
        memcpy((void*) &c_needle_list[6 * k], player_needle_lists[player_num - 1][k].c_str(),
               player_needle_lists[player_num - 1][k].size());
        memcpy((void*) &c_dfas[7 * k],  player_dfa_lists[player_num - 1][k], needle_size_list[k] + 1);
    }

    return match_count_multiple(c_lines, c_needle_list, c_dfas, needle_size_list, c_line_size, score_map);
}

int HeuristicMinMaxStrategy::EvaluateChessByLines(const std::array<std::string, 4>& lines, int player_num) {
    int res = 0;
    for (const auto& li: lines) {
        for (int nx = 0; nx < player_needle_lists[player_num - 1].size(); nx ++) {
            res += match_count(li, player_needle_lists[player_num - 1][nx],
                               player_dfa_lists[player_num - 1][nx]) * score_map[nx];
        }
    }

    return res;
}

int HeuristicMinMaxStrategy::CountPoints(int player_num, const std::string& s, int& level_points) {
    for (int i = 0; i < player1_needle_list.size(); i ++) {
        level_points += match_count(s, player_needle_lists[player_num - 1][i],
                                    player_dfa_lists[player_num - 1][i]) * score_map[i];
    }
    return 0;
}

int HeuristicMinMaxStrategy::Diagonal(Board& board, const int& x, const int& y, std::string& s, const int& boardSize) {
    int c = y, r = x;
    while (r < boardSize && c < boardSize) {
        s += Int2Char(board.GetChess(r, c));
        c ++;
        r ++;
    }
    return 0;
}

int HeuristicMinMaxStrategy::AntiDiagonal(Board& board, const int& x, const int& y, std::string& s, const int& boardSize) {
    int r = x, c = y;
    while (r < boardSize && c >= 0) {
        s += Int2Char(board.GetChess(r, c));
        r ++;
        c --;
    }
    return 0;
}

int HeuristicMinMaxStrategy::EvaluateBoard(Board &board, int player_num) {
    int boardSize = board.GetSize();
    int level_points = 0;

    std::string s;
    s.reserve(boardSize);

    // From up to down.
    for (int r = 0; r < boardSize; r ++) {
        s = "";
        for (int c = 0; c < boardSize; c ++) {
            s += Int2Char(board.GetChess(r, c));
        }
        // regular exp (enumerate each needle of myself)
        CountPoints(player_num, s, level_points);
    }

    // From left to right.
    for (int c = 0; c < boardSize; c ++) {
        s = "";
        for (int r = 0; r < boardSize; r ++) {
            s += Int2Char(board.GetChess(r, c));
        }
        // regular exp (enumerate each needle of myself)
        CountPoints(player_num, s, level_points);
    }

    // Diagonal.
    for (int r = 0; r < boardSize; r ++) {
        s = "";
        Diagonal(board, r, 0, s, boardSize);
        CountPoints(player_num, s, level_points);
    }
    for (int c = 1; c < boardSize; c ++) {
        s = "";
        Diagonal(board, 0, c, s, boardSize);
        CountPoints(player_num, s, level_points);
    }

    // Anti-diagonal.
    for (int c = 0; c < boardSize; c ++) {
        s = "";
        AntiDiagonal(board, 0, c, s, boardSize);
        CountPoints(player_num, s, level_points);
    }
    for (int r = 1; r < boardSize; r ++) {
        s = "";
        AntiDiagonal(board, r, boardSize - 1, s, boardSize);
        CountPoints(player_num, s, level_points);
    }

    return level_points;
}

std::vector<std::pair<int, int>> HeuristicMinMaxStrategy::HeuristicNextMoves(Board& board, int player_num, bool max_layer) {
    std::cout << "HeuristicNextMoves: entering" << std::endl;
    Board tmp_board{board};
    int board_size = board.GetSize();
    int opp_player_num = 3 - player_num;
    int old_chess_score, new_chess_score, delta_chess_score;
    std::vector<std::pair<int, int>> possible_moves; // (score, location)
    std::vector<std::pair<int, int>> res;
    std::array<std::string, 4> lines;

    for (int move : board.AvailableChildren(1)) {
        int mr = move / board_size;
        int mc = move % board_size;

        lines = GetLinesByChess(tmp_board, mr, mc);
        old_chess_score = EvaluateChessByLinesGPU(lines, player_num) -
                EvaluateChessByLinesGPU(lines, opp_player_num);

        if (max_layer)
            tmp_board.PlaceChess(player_num, mr, mc);
        else
            tmp_board.PlaceChess(opp_player_num, mr, mc);

        lines = GetLinesByChess(tmp_board, mr, mc);
        new_chess_score = EvaluateChessByLinesGPU(lines, player_num) -
                EvaluateChessByLinesGPU(lines, opp_player_num);

        RevertWrapper(tmp_board, mr, mc);
        delta_chess_score = new_chess_score - old_chess_score;
        possible_moves.emplace_back(std::make_pair(delta_chess_score, move));
    }

    if (max_layer)
        std::sort(possible_moves.begin(), possible_moves.end(), std::greater<>());
    else
        std::sort(possible_moves.begin(), possible_moves.end(), std::less<>());

    for (int i = 0; i < std::min(board_size, (int)possible_moves.size()); i ++) {
        res.emplace_back(possible_moves[i]);
    }
    return res;
}

std::pair<int, int> HeuristicMinMaxStrategy::EvalTotalPoints(
        Board board, int player_num, int cur_depth, int alpha, int beta, int score
        ) {
    std::cout << "EvalTotalPoints: entering" << std::endl;
    int opp_player_num = 3 - player_num;
    int location, val, result;
    int boardSize = board.GetSize();

    if (board.IsFinish() || cur_depth == total_depth) {
        return std::pair<int, int>{0, score};
    }

    /* IF AI's turn */
    if (cur_depth % 2 == 0) {
        result = INT_MIN;
        for (const auto& p : HeuristicNextMoves(board, player_num, true)) {
            int dscore = p.first, loc = p.second;
            int r = loc / boardSize, c = loc % boardSize;
            Board b{board};
            if (cur_depth % 2 == 0) PlaceWrapper(b, player_num, r, c);
            else PlaceWrapper(b, opp_player_num, r, c);

            val = EvalTotalPoints(b, player_num, cur_depth + 1, alpha, beta, score + dscore).second;
            if (val > result) {
                location = r * boardSize + c;
                result = val;
            }

            if (result >= beta) return std::pair<int, int>{location, result};
            alpha = std::max(alpha, val);
        }
        return std::pair<int, int>{location, result};
    } else {
        result = INT_MAX;
        for (const auto& p : HeuristicNextMoves(board, player_num, false)) {
            int dscore = p.first, loc = p.second;
            int r = loc / boardSize, c = loc % boardSize;
            Board b{board};

            if (cur_depth % 2 == 0) PlaceWrapper(b,player_num, r, c);
            else PlaceWrapper(b, opp_player_num, r, c);

            val = EvalTotalPoints(b, player_num, cur_depth + 1, alpha, beta, score + dscore).second;
            if (val < result) {
                location = r * boardSize + c;
                result = val;
            }

            if (result <= alpha) return std::pair<int, int>{location, result};
            beta = std::min(beta, val);
        }
        return std::pair<int, int>{location, result};
    }
}

bool HeuristicMinMaxStrategy::GetStrategy(Board* board, int player_num, int *px, int *py) {
    std::cout << "GetStrategy: entering" << std::endl;
    int boardSize = board->GetSize();

    int score = EvaluateBoard(*board, player_num) - EvaluateBoard(*board, 3 - player_num);
    std::pair<int, int> nxt_step = EvalTotalPoints(*board, player_num, 0, INT_MIN, INT_MAX, score);

    *px = nxt_step.first / boardSize;
    *py = nxt_step.first % boardSize;

    return true;
}

int HeuristicMinMaxStrategy::PlaceWrapper(Board &board, int player_num, int r, int c) {
    board.PlaceChess(player_num, r, c);
    return 0;
}

int HeuristicMinMaxStrategy::RevertWrapper(Board &board, int r, int c) {
    board.Revert(r, c);
    return 0;
}