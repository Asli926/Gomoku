#include "../include/MinMaxStrategy.h"
#include <string>
#include <limits>
#include <iostream>


MinMaxStrategy::MinMaxStrategy(int _total_depth) {
    player1_needle_map = needle_map_t{};
    player1_needle_map[ONE] = std::regex{"10"};
    player1_needle_map[TWO] = std::regex{"0110"};
    player1_needle_map[THREE] = std::regex{"01110"};
    player1_needle_map[FOUR] = std::regex{"011110"};
    player1_needle_map[FIVE] = std::regex{"11111"};
    player1_needle_map[BLOCKED_TWO] = std::regex{"2110|0112"};
    player1_needle_map[BLOCKED_THREE] = std::regex{"21110|01112"};
    player1_needle_map[BLOCKED_FOUR] = std::regex{"211110|011112"};

    player2_needle_map = needle_map_t{};
    player2_needle_map[ONE] = std::regex{"20"};
    player2_needle_map[TWO] = std::regex{"0220"};
    player2_needle_map[THREE] = std::regex{"02220"};
    player2_needle_map[FOUR] = std::regex{"022220"};
    player2_needle_map[FIVE] = std::regex{"22222"};
    player2_needle_map[BLOCKED_TWO] = std::regex{"1220|0221"};
    player2_needle_map[BLOCKED_THREE] = std::regex{"12220|02221"};
    player2_needle_map[BLOCKED_FOUR] = std::regex{"122220|022221"};


    score_map.insert({{ONE, 10}, {TWO, 100}, {THREE, 1000}, {FOUR, 100000},
                      {FIVE, 100000000}, {BLOCKED_TWO, 10},
                      {BLOCKED_THREE, 100}, {BLOCKED_FOUR, 10000}});

    total_depth = _total_depth;
}

/* only translate 0 - 9 to its char type */
inline char MinMaxStrategy::Int2Char(int num) {
    return (char) (num - 0 + (int)('0'));
}

int MinMaxStrategy::Diagonal(Board& board, const int& x, const int& y, std::string& s, const int& boardSize) {
    int c = y, r = x;
    while (r < boardSize && c < boardSize) {
        s += Int2Char(board.GetChess(r, c));
        c ++;
        r ++;
    }
    return 0;
}

int MinMaxStrategy::AntiDiagonal(Board& board, const int& x, const int& y, std::string& s, const int& boardSize) {
    int r = x, c = y;
    while (r < boardSize && c >= 0) {
        s += Int2Char(board.GetChess(r, c));
        r ++;
        c --;
    }
    return 0;
}

int MinMaxStrategy::CountPoints(const needle_map_t& player_needle_map, const std::string& s, int& level_points) {
    for (const auto& n : player_needle_map) {
        auto words_begin = std::sregex_iterator(s.begin(), s.end(), n.second);
        auto words_end = std::sregex_iterator();
        level_points += (int)std::distance(words_begin, words_end) * score_map[n.first];
    }
    return 0;
}


int MinMaxStrategy::EvalLevelPoints(Board& board, int player_num) {
    int opp_player_num;
    int boardSize = board.GetSize();
    int level_points = 0;

    needle_map_t player_needle_map = player2_needle_map;
    if (player_num == 1) {
        player_needle_map = player1_needle_map;
        opp_player_num = 2;
    } else {
        opp_player_num = 1;
    }

    std::string s(Int2Char(opp_player_num), boardSize);
    // From up to down.
    for (int r = 0; r < boardSize; r ++) {
        s = Int2Char(opp_player_num);
        for (int c = 0; c < boardSize; c ++) {
            s += Int2Char(board.GetChess(r, c));
        }
        s += Int2Char(opp_player_num);
        // regular exp (enumerate each needle of myself)
        CountPoints(player_needle_map, s, level_points);
    }

    // From left to right.
    for (int c = 0; c < boardSize; c ++) {
        s = Int2Char(opp_player_num);
        for (int r = 0; r < boardSize; r ++) {
            s += Int2Char(board.GetChess(r, c));
        }
        s += Int2Char(opp_player_num);
        // regular exp (enumerate each needle of myself)
        CountPoints(player_needle_map, s, level_points);
    }

    // Diagonal.
    for (int r = 0; r < boardSize; r ++) {
        s = Int2Char(opp_player_num);
        Diagonal(board, r, 0, s, boardSize);
        s += Int2Char(opp_player_num);
        CountPoints(player_needle_map, s, level_points);
    }
    for (int c = 1; c < boardSize; c ++) {
        s = Int2Char(opp_player_num);
        Diagonal(board, 0, c, s, boardSize);
        s += Int2Char(opp_player_num);
        CountPoints(player_needle_map, s, level_points);
    }

    // Anti-diagonal.
    for (int c = 0; c < boardSize; c ++) {
        s = Int2Char(opp_player_num);
        AntiDiagonal(board, 0, c, s, boardSize);
        s += Int2Char(opp_player_num);
        CountPoints(player_needle_map, s, level_points);
    }
    for (int r = 1; r < boardSize; r ++) {
        s = Int2Char(opp_player_num);
        AntiDiagonal(board, r, boardSize - 1, s, boardSize);
        s += Int2Char(opp_player_num);
        CountPoints(player_needle_map, s, level_points);
    }
    return level_points;
}



std::pair<int, int> MinMaxStrategy::EvalTotalPoints(Board board, int player_num, int cur_depth) {
    int opp_player_num = 1;
    if (player_num == 1) opp_player_num = 2;
    int score, location;
    int boardSize = board.GetSize();
    // std::cout << total_depth << '\n';
    if (board.IsFinish() || cur_depth == total_depth) {
        return std::pair<int, int>{0, EvalLevelPoints(board, player_num) - EvalLevelPoints(board, opp_player_num)};
    }

    std::unordered_map<int, int> loc_score_map;
    for (int i = 0; i < boardSize; i ++) {
        for (int j = 0; j < boardSize; j ++) {

            // Most-likely new chess puts around the existing ones.
            if (board.GetChess(i, j) != 0) {
                 for (int r = std::max(i - 2, 0); r <= std::min(i + 2, boardSize - 1); r ++) {
                    for (int c = std::max(j - 2, 0); c <= std::min(j + 2, boardSize - 1); c ++) {
                        // if not visited, store it.
                        if (board.GetChess(r, c) == 0 && loc_score_map.find(boardSize * r + c) == loc_score_map.end()) {
                            Board b{board};
                            if (cur_depth % 2 == 0) b.PlaceChess(player_num, r, c);
                            else                    b.PlaceChess(opp_player_num, r, c);
                            loc_score_map[r * boardSize + c] = EvalTotalPoints(b, player_num, cur_depth + 1).second;
                        }
                    }
                }
            }

        }
    }
                                                    // MAX layer
    if (cur_depth % 2 == 0) {
        score = INT_MIN;
        for (const auto& n : loc_score_map) {
            if (n.second > score) {
                location = n.first;
                score = n.second;
            }
        }
                                                    // MIN layer
    } else {
        score = INT_MAX;
        for (const auto& n : loc_score_map) {
            if (n.second < score) {
                location = n.first;
                score = n.second;
            }
        }
    }

    return std::pair<int, int>{location, score};
}


bool MinMaxStrategy::GetStrategy(Board* board, int player_num, int *px, int *py) {
    int boardSize = board->GetSize();
    // int total_depth = 2;
    std::pair<int, int> nxt_step = EvalTotalPoints(*board, player_num, 0);

    *px = nxt_step.first / boardSize;
    *py = nxt_step.first % boardSize;

    return true;
}