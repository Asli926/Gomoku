#include "../include/HeuristicMinMaxStrategy.h"
#include "../include/pattern_search.h"
#include <string>
#include <array>
#include <functional>

HeuristicMinMaxStrategy::HeuristicMinMaxStrategy(int _total_depth) {
    player1_needle_list = needle_list_t{
        "11111", "011110", "011100", "001110", "011010",
        "010110", "11110", "01111", "11011", "10111",
        "11101", "001100", "001010", "010100", "000100", "001000"
    };

    player1_dfa_list = dfa_list_t{};
    for (const auto& needle : player1_needle_list) {
        player1_dfa_list.emplace_back(construct_nxt(needle));
    }

    player2_needle_list = needle_list_t{
            "22222", "022220", "022200", "002220", "012020",
            "020220", "22220", "02222", "22022", "20222",
            "22202", "002200", "002020", "020200", "000200", "002000"
    };

    player2_dfa_list = dfa_list_t{};
    for (const auto& needle : player2_needle_list) {
        player2_dfa_list.emplace_back(construct_nxt(needle));
    }

    player_needle_lists = std::vector<needle_list_t>{player1_needle_list, player2_needle_list};
    player_dfa_lists = std::vector<dfa_list_t>{player1_dfa_list, player2_dfa_list};

    score_map = std::vector<int>{50000, 4320, 720, 720, 720, 720, 720, 720, 720,
                                 720, 720, 120, 120, 120, 20, 20};

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

int HeuristicMinMaxStrategy::EvaluateChessByLines(const std::array<std::string, 4>& lines, int player_num) {
    int res = 0;
    for (const auto& li: lines) {
        for (int nx = 0; nx < player_needle_lists[player_num - 1].size(); nx ++) {
            res += match_count(li, player_needle_lists[player_num - 1][nx],
                               player_dfa_lists[player_num - 1][nx]) * score_map[nx];

//            const auto& n = player_needle_lists[player_num - 1][nx];
//            auto words_begin = boost::sregex_iterator(li.begin(), li.end(), n);
//            auto words_end = boost::sregex_iterator();
//            res += (int)std::distance(words_begin, words_end) * score_map[nx];
        }
    }

    return res;
}

int HeuristicMinMaxStrategy::CountPoints(int player_num, const std::string& s, int& level_points) {
    for (int i = 0; i < player1_needle_list.size(); i ++) {
//        const auto& n = player_needle_list[i];
//        auto words_begin = boost::sregex_iterator(s.begin(), s.end(), n);
//        auto words_end = boost::sregex_iterator();
//        level_points += (int)std::distance(words_begin, words_end) * score_map[i];
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
        old_chess_score = EvaluateChessByLines(lines, player_num) -
                            EvaluateChessByLines(lines, opp_player_num);

        if (max_layer)
            tmp_board.PlaceChess(player_num, mr, mc);
        else
            tmp_board.PlaceChess(opp_player_num, mr, mc);

        lines = GetLinesByChess(tmp_board, mr, mc);
        new_chess_score = EvaluateChessByLines(lines, player_num) -
                          EvaluateChessByLines(lines, opp_player_num);

        RevertWrapper(tmp_board, mr, mc);
        delta_chess_score = new_chess_score - old_chess_score;
        possible_moves.emplace_back(std::make_pair(delta_chess_score, move));
    }

    if (max_layer)
        std::sort(possible_moves.begin(), possible_moves.end(), std::greater<>());
    else
        std::sort(possible_moves.begin(), possible_moves.end(), std::less<>());

    for (int i = 0; i < std::min(10, (int)possible_moves.size()); i ++) {
        res.emplace_back(possible_moves[i]);
    }
    return res;
}

std::pair<int, int> HeuristicMinMaxStrategy::EvalTotalPoints(
        Board board, int player_num, int cur_depth, int alpha, int beta, int score
        ) {
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