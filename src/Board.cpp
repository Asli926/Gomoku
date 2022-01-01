#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "../include/Board.h"

unsigned int brandu32() {
    return (unsigned int)(rand() | (rand() << 15));
}

Board::Board(int size) {
    size_ = size;
    board_ = new int[size * size];
    Initialize();
}

Board::Board(const Board& board) {
    int size_sqr = board.size_ * board.size_;
    size_ = board.size_;
    board_ = new int[size_sqr];
    std::memcpy(board_, board.board_, size_sqr * sizeof(int));
    last_chess_row_ = board.last_chess_row_;
    last_chess_col_ = board.last_chess_col_;
    chess_count = board.chess_count;
    hash_value[0] = board.hash_value[0]; hash_value[1] = board.hash_value[1];
}

Board::~Board() {
    delete[] board_;
}

int Board::GetSize() {
    return size_;
}

unsigned int Board::GetHash(int player_num) {
    return hash_value[player_num - 1];
}

int Board::SetHash(int player_num, unsigned int new_hash) {
    hash_value[player_num - 1] = new_hash;
    return 0;
}

int Board::GetChess(int row, int col) {
    return board_[size_ * row + col];
}

int Board::Initialize() {
    last_chess_row_ = last_chess_col_ = -1;
    hash_value[0] = brandu32(); hash_value[1] = brandu32();
    chess_count = 0;

    for (int i = 0; i < size_ * size_; i ++) {
        board_[i] = 0;
    }
    return 0;
}

int Board::PlaceChess(int player_num, int x, int y) {
    if (!IsValid(x, y)) return -1;
    int location = size_ * x + y;

    if (board_[location] != 0) { return -1; }

    board_[location] = player_num;
    last_chess_row_ = x; last_chess_col_ = y;
    chess_count += 1;

    return 0;
}

bool Board::IsValid(int x, int y) {
    return x >= 0 && x < size_ && y >= 0 && y < size_;
}

int Board::CountByDirection_(int dx, int dy, int player) {
    int count = 1;

    for (int step = 1; step < 5; step ++) {
        if (IsValid(last_chess_row_ + step * dx, last_chess_col_ + step * dy)) {
            if(board_[size_ * (last_chess_row_ + step * dx) + last_chess_col_ + step * dy] == player)
                count++;
            else
                break;
        } else {
            break;
        }
    }

    for (int step = 1; step < 5; step ++) {
        if (IsValid(last_chess_row_ - step * dx, last_chess_col_ - step * dy)) {
            if(board_[size_ * (last_chess_row_ - step * dx) + last_chess_col_ - step * dy] == player)
                count++;
            else
                break;
        } else {
            break;
        }
    }

    return count;
}

/* 0: not finished
 * 1: someone win
 * 2: board is full, draw */
int Board::IsFinish() {
    int count;
    if (last_chess_row_ == -1) return 0;

    int player = board_[size_ * last_chess_row_ + last_chess_col_];

    count = CountByDirection_(1, 0, player);
    if (count >= 5) return 1;

    count = CountByDirection_(0, 1, player);
    if (count >= 5) return 1;

    count = CountByDirection_(1, 1, player);
    if (count >= 5) return 1;

    count = CountByDirection_(1, -1, player);
    if (count >= 5) return 1;

    if (chess_count == size_ * size_) return 2;

    return 0;
}

int Board::PrintBoard() {
    printf("============= Board =============\n");
    for (int i = 0; i < size_; i ++) {
        for (int j = 0; j < size_; j ++) {
            printf("%d ", board_[i * size_ + j]);
        }
        printf("\n");
    }
    return 0;
}

std::unordered_set<int> Board::AvailableChildren(int dist) {
    std::unordered_set<int> availChildren;
    
    for (int i = 0; i < size_; i ++) {
        for (int j = 0; j < size_; j ++) {

            // Most-likely new chess puts around the existing ones.
            if (GetChess(i, j) != 0) {
                for (int r = std::max(i - dist, 0); r <= std::min(i + dist, size_ - 1); r ++) {
                    for (int c = std::max(j - dist, 0); c <= std::min(j + dist, size_ - 1); c ++) {
                        if (GetChess(r, c) == 0) availChildren.insert(r * size_ + c);
                    }
                }
            }

        }
    }
    return availChildren;
}

int Board::Revert(int x, int y) {
    int player = GetChess(x, y);
    if (player == 0) return -1;

    int loc = x * size_ + y;

    board_[loc] = 0;
    return 0;
}


