#include "../include/HeuristicMinMaxStrategy.h"
#include "../include/pattern_search.h"
#include <cstring>
#include <array>
#include <functional>
#include <climits>
#include <algorithm>
#include <iostream>
#include <utility>
#include <ctime>
#include "../cuda/gpu_match.cuh"
#include "../cuda/heuristic_moves.cuh"

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

    c_needle_list_two = new char*[2];
    c_needle_list_two[0] = new char[16 * 6];
    c_needle_list_two[1] = new char[16 * 6];

    for (int k = 0; k < 16; k ++) {
        memcpy((void*) &c_needle_list_two[0][6 * k], player_needle_lists[0][k].c_str(),
               player_needle_lists[0][k].size());
        memcpy((void*) &c_needle_list_two[1][6 * k], player_needle_lists[1][k].c_str(),
               player_needle_lists[1][k].size());
    }

    c_dfas_two = new int*[2];
    c_dfas_two[0] = new int[16 * 7];
    c_dfas_two[1] = new int[16 * 7];

    for (int u = 0; u < 16; u ++) {
        for (int w = 0; w < 7; w ++) {
            c_dfas_two[0][u * 7 + w] = player_dfa_lists[0][u][w];
            c_dfas_two[1][u * 7 + w] = player_dfa_lists[1][u][w];
        }
    }

    // copy constant variable to the gpu
//    setPatternRelatedInfo(c_needle_list_two[0], c_needle_list_two[1],
//                          c_dfas_two[0], c_dfas_two[1], needle_size_list, score_map);

    // Test GPU function
    Board board_test{15};
    board_test.PlaceChess(2, 7, 7);
    auto mov_paris = HeuristicNextMoves(board_test, 1, true);
    if (!(mov_paris[0].first < 2000 && mov_paris[0].first > -2000)) {
        std::cout << "GPU TEST NOT PASSED!" << std::endl;
        exit(-1);
    }
}

HeuristicMinMaxStrategy::~HeuristicMinMaxStrategy() {
    delete[] needle_size_list;
    delete[] player1_dfa_list;
    delete[] player2_dfa_list;
    delete[] score_map;
}

/* only translate 0 - 9 to its char type */
inline char HeuristicMinMaxStrategy::Int2Char(int num) {
    return (char) (num - 0 + (int)('0'));
}

int HeuristicMinMaxStrategy::GetLinesByChess(Board& board, int r, int c, char* lines, int* line_sizes) {
    int board_size = board.GetSize();
    int pLines = 0;

    for (int i = 0; i < board_size; i ++) {
        lines[pLines++] = Int2Char(board.GetChess(i, c));
    }
    line_sizes[0] = pLines; pLines = 20;

    for (int j = 0; j < board_size; j ++) {
        lines[pLines++] = Int2Char(board.GetChess(r, j));
    }
    line_sizes[1] = pLines; pLines = 40;

    int min_rc = std::min(r, c);
    for (int i = r - min_rc, j = c - min_rc;
    i < board_size && j < board_size;
    i ++, j ++) {
        lines[pLines++] = Int2Char(board.GetChess(i, j));
    }
    line_sizes[2] = pLines; pLines = 60;

    int dr = std::min(r, board_size - 1 - c);
    for (int i = r - dr, j = c + dr; i < board_size && j >= 0; i ++, j --) {
        lines[pLines++] = Int2Char(board.GetChess(i, j));
    }
    line_sizes[3] = pLines;

    return 0;
}

//int HeuristicMinMaxStrategy::EvaluateChessScoreByLinesGPU(char* c_lines, int* c_line_size, int player_num) {
//    // std::cout << "EvaluateChessByLinesGPU: cuda function wrapper" << std::endl;
//    return match_count_multiple(c_lines, c_line_size, player_num);
//}

//int HeuristicMinMaxStrategy::EvaluateChessByLines(const std::array<std::string, 4>& lines, int player_num) {
//    int res = 0;
//    for (const auto& li: lines) {
//        for (int nx = 0; nx < player_needle_lists[player_num - 1].size(); nx ++) {
//            res += match_count(li, player_needle_lists[player_num - 1][nx],
//                               player_dfa_lists[player_num - 1][nx]) * score_map[nx];
//        }
//    }
//
//    return res;
//}

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
    int *scores = (int*) malloc(sizeof(int) * 60);
    int *locations = (int*) malloc(sizeof(int) * 60);
    int moves_count = 0;
    clock_t start = clock();
    heuristic_moves_cpu(scores, locations, &moves_count, board.GetRawBoard(), player_num,
                        c_needle_list_two[0], c_needle_list_two[1],
                        c_dfas_two[0], c_dfas_two[1],
                        needle_size_list, score_map, max_layer ? 1 : 0);
    printf("gpu call time: %.4f (s)\n", (clock() - start) / (double) CLOCKS_PER_SEC);
    std::vector<std::pair<int, int>> possible_moves; // (score, location)
    std::vector<std::pair<int, int>> res; // (score, location)
    possible_moves.reserve(moves_count);
    for (int i = 0; i < moves_count; i ++) {
        possible_moves.emplace_back(std::make_pair(scores[i], locations[i]));
    }

    if (max_layer)
        std::sort(possible_moves.begin(), possible_moves.end(), std::greater<>());
    else
        std::sort(possible_moves.begin(), possible_moves.end(), std::less<>());

    for (int i = 0; i < std::min(board.GetSize(), (int)possible_moves.size()); i ++) {
        res.emplace_back(possible_moves[i]);
    }
    return res;
}

std::pair<int, int> HeuristicMinMaxStrategy::EvalTotalPoints(
        Board board, int player_num, int cur_depth, int alpha, int beta, int score
        ) {
    // std::cout << "EvalTotalPoints: entering" << std::endl;
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
            if (cur_depth % 2 == 0) b.PlaceChess(player_num, r, c);
            else b.PlaceChess(opp_player_num, r, c);

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

            if (cur_depth % 2 == 0) b.PlaceChess(player_num, r, c);
            else b.PlaceChess(opp_player_num, r, c);

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
