#include "../include/HeuristicMinMaxStrategy.h"
#include <string>
#include <functional>

unsigned int randu32() {
    return (unsigned int)(rand() | (rand() << 15));
}

HeuristicMinMaxStrategy::HeuristicMinMaxStrategy(int _total_depth) {
    player1_needle_list = needle_list_t{
        boost::regex{"11111"}, boost::regex{"011110"}, boost::regex{"011100"},
        boost::regex{"001110"}, boost::regex{"011010"}, boost::regex{"010110"},
        boost::regex{"11110"}, boost::regex{"01111"}, boost::regex{"11011"},
        boost::regex{"10111"}, boost::regex{"11101"}, boost::regex{"001100"},
        boost::regex{"001010"}, boost::regex{"010100"}, boost::regex{"000100"},
        boost::regex{"001000"}
    };

    player2_needle_list = needle_list_t{
            boost::regex{"22222"}, boost::regex{"022220"}, boost::regex{"022200"},
            boost::regex{"002220"}, boost::regex{"012020"}, boost::regex{"020220"},
            boost::regex{"22220"}, boost::regex{"02222"}, boost::regex{"22022"},
            boost::regex{"20222"}, boost::regex{"22202"}, boost::regex{"002200"},
            boost::regex{"002020"}, boost::regex{"020200"}, boost::regex{"000200"},
            boost::regex{"002000"}
    };

    player_needle_lists = std::vector<needle_list_t>{player1_needle_list, player2_needle_list};

    score_map = std::vector<int>{50000, 4320, 720, 720, 720, 720, 720, 720, 720,
                                 720, 720, 120, 120, 120, 20, 20};

    total_depth = _total_depth;
}

int HeuristicMinMaxStrategy::SetBoardSize(int board_size) {
    for (int i = 0; i < board_size * board_size; i ++) {
        player1_rd_tb.emplace_back(randu32());
        player2_rd_tb.emplace_back(randu32());
    }
    return 0;
}

/* only translate 0 - 9 to its char type */
inline char HeuristicMinMaxStrategy::Int2Char(int num) {
    return (char) (num - 0 + (int)('0'));
}

int HeuristicMinMaxStrategy::EvaluateChess(Board& board, int player_num, int r, int c) {
    int board_size = board.GetSize();
    int res = 0;
    std::string line[4];

    for (int i = std::max(0, r - 5); i < std::max(board_size - 1, r + 6); i ++) {
        line[0] += Int2Char(board.GetChess(i, c));
    }

    for (int j = std::max(0, c - 5); j < std::max(board_size - 1, c + 6); j ++) {
        line[1] += Int2Char(board.GetChess(r, j));
    }

    int min_rc = std::min(r, c);
    for (int i = r - std::min(min_rc, 5), j = c - std::min(min_rc, 5);
        i < std::max(board_size - 1, r + 6) && j < std::max(board_size - 1, c + 6);
        i ++, j ++) {
        line[2] += Int2Char(board.GetChess(i, j));
    }

    for (int i = std::max(board_size - 1, r + 5), j = std::max(0, c - 5);
        i >= std::max(0, r - 5) && j < std::max(board_size - 1, c + 6);
        i --, j ++) {
        if (board.IsValid(i, j)) {
            line[3] += Int2Char(board.GetChess(i, j));
        }
    }


    for (const auto& li: line) {
        for (int nx = 0; nx < player_needle_lists[player_num - 1].size(); nx ++) {
            const auto& n = player_needle_lists[player_num - 1][nx];
            auto words_begin = boost::sregex_iterator(li.begin(), li.end(), n);
            auto words_end = boost::sregex_iterator();
            res += (int)std::distance(words_begin, words_end) * score_map[nx];
        }
    }

    return res;
}

int HeuristicMinMaxStrategy::CountPoints(const needle_list_t & player_needle_list, const std::string& s, int& level_points) {
    for (int i = 0; i < player_needle_list.size(); i ++) {
        const auto& n = player_needle_list[i];
        auto words_begin = boost::sregex_iterator(s.begin(), s.end(), n);
        auto words_end = boost::sregex_iterator();
        level_points += (int)std::distance(words_begin, words_end) * score_map[i];
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
    if (board_score.find(board.GetHash(player_num)) != board_score.end()) {
        // printf("cache hit\n");
        return board_score[board.GetHash(player_num)];
    }

    // int opp_player_num = 3 - player_num;
    int boardSize = board.GetSize();
    int level_points = 0;

    std::string s;
    // From up to down.
    for (int r = 0; r < boardSize; r ++) {
        s = "";
        for (int c = 0; c < boardSize; c ++) {
            s += Int2Char(board.GetChess(r, c));
        }
        // regular exp (enumerate each needle of myself)
        CountPoints(player_needle_lists[player_num - 1], s, level_points);
    }

    // From left to right.
    for (int c = 0; c < boardSize; c ++) {
        s = "";
        for (int r = 0; r < boardSize; r ++) {
            s += Int2Char(board.GetChess(r, c));
        }
        // regular exp (enumerate each needle of myself)
        CountPoints(player_needle_lists[player_num - 1], s, level_points);
    }

    // Diagonal.
    for (int r = 0; r < boardSize; r ++) {
        s = "";
        Diagonal(board, r, 0, s, boardSize);
        CountPoints(player_needle_lists[player_num - 1], s, level_points);
    }
    for (int c = 1; c < boardSize; c ++) {
        s = "";
        Diagonal(board, 0, c, s, boardSize);
        CountPoints(player_needle_lists[player_num - 1], s, level_points);
    }

    // Anti-diagonal.
    for (int c = 0; c < boardSize; c ++) {
        s = "";
        AntiDiagonal(board, 0, c, s, boardSize);
        CountPoints(player_needle_lists[player_num - 1], s, level_points);
    }
    for (int r = 1; r < boardSize; r ++) {
        s = "";
        AntiDiagonal(board, r, boardSize - 1, s, boardSize);
        CountPoints(player_needle_lists[player_num - 1], s, level_points);
    }

    board_score[board.GetHash(player_num)] = level_points;
    return level_points;
}

std::vector<int> HeuristicMinMaxStrategy::HeuristicNextMoves(Board& board, int player_num, int max_num) {
    Board tmp_board{board};
    int board_size = board.GetSize();
    int op_player_num = 3 - player_num;
    int scoreMe, scoreOp;
    std::vector<std::pair<int, int>> possible_moves; // (score, location)
    std::vector<int> res;

    for (int move : board.AvailableChildren(1)) {
        int mr = move / board_size;
        int mc = move % board_size;

        tmp_board.PlaceChess(player_num, mr, mc);
        scoreMe = EvaluateChess(tmp_board, player_num, mr, mc);
        scoreOp = EvaluateChess(tmp_board, op_player_num, mr, mc);
        tmp_board.Revert(mr, mc);

        possible_moves.emplace_back(std::make_pair(scoreMe - scoreOp, move));
    }

    std::sort(possible_moves.begin(), possible_moves.end(), std::greater<>());
    for (int i = 0; i < std::min(max_num, (int)possible_moves.size()); i ++) {
        res.emplace_back(possible_moves[i].second);
    }

    return res;
}

std::pair<int, int> HeuristicMinMaxStrategy::EvalTotalPoints(
        Board board, int player_num, int cur_depth, int alpha, int beta
        ) {
    int opp_player_num = 1;
    if (player_num == 1) opp_player_num = 2;
    int location, val, result;
    int boardSize = board.GetSize();

    if (board.IsFinish() || cur_depth == total_depth) {
        return std::pair<int, int>{0, EvaluateBoard(board, player_num) - EvaluateBoard(board, opp_player_num)};
    }

    /* IF AI's turn */
    if (cur_depth % 2 == 0) {
        result = INT_MIN;
        for (const auto &loc : board.AvailableChildren(1)) {
            int r = loc / boardSize, c = loc % boardSize;
            Board b{board};
            if (cur_depth % 2 == 0) PlaceWrapper(b, player_num, r, c);
            else PlaceWrapper(b, opp_player_num, r, c);
            val = EvalTotalPoints(b, player_num, cur_depth + 1, alpha, beta).second;
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
        for (const auto &loc : board.AvailableChildren(1)) {
            int r = loc / boardSize, c = loc % boardSize;
            Board b{board};
            if (cur_depth % 2 == 0) PlaceWrapper(b,player_num, r, c);
            else PlaceWrapper(b, opp_player_num, r, c);
            val = EvalTotalPoints(b, player_num, cur_depth + 1, alpha, beta).second;
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
    board_score.clear();
    std::pair<int, int> nxt_step = EvalTotalPoints(*board, player_num, 0, INT_MIN, INT_MAX);

    *px = nxt_step.first / boardSize;
    *py = nxt_step.first % boardSize;

    return true;
}

int HeuristicMinMaxStrategy::PlaceWrapper(Board &board, int player_num, int r, int c) {
    board.PlaceChess(player_num, r, c);
    int loc = r * board.GetSize() + c;
    if (player_num == 1)
        board.SetHash(player_num, board.GetHash(player_num) ^ player1_rd_tb[loc]);
    else
        board.SetHash(player_num, board.GetHash(player_num) ^ player2_rd_tb[loc]);

    return 0;
}

int HeuristicMinMaxStrategy::RevertWrapper(Board &board, int r, int c) {
    int loc = r * board.GetSize() + c;
    int player_num = board.GetChess(r, c);
    if (player_num == 1)
        board.SetHash(player_num, board.GetHash(player_num) ^ player1_rd_tb[loc]);
    else
        board.SetHash(player_num, board.GetHash(player_num) ^ player2_rd_tb[loc]);


    board.Revert(r, c);
    return 0;
}