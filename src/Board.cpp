#include <cstdio>
#include "../include/Board.h"

Board::Board(int size) {
    size_ = size;
    board_ = new int[size * size];
    Initialize();
}

Board::~Board() {
    delete[] board_;
}

int Board::GetSize() {
    return size_;
}

int Board::GetChess(int row, int col) {
    return board_[size_ * row + col];
}

int Board::Initialize() {
    last_chess_row_ = last_chess_col_ = -1;
    for (int i = 0; i < size_ * size_; i ++) {
        board_[i] = 0;
    }
}

int Board::PlaceChess(int player_num, int x, int y) {
    if (!IsValid(x, y)) return -1;
    int location = size_ * x + y;

    if (board_[location] != 0) { return -1; }

    board_[size_ * x + y] = player_num;
    last_chess_row_ = x; last_chess_col_ = y;

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

bool Board::IsFinish() {
    int count;
    if (last_chess_row_ == -1) return false;

    int player = board_[size_ * last_chess_row_ + last_chess_col_];

    count = CountByDirection_(1, 0, player);
    if (count >= 5) return true;

    count = CountByDirection_(0, 1, player);
    if (count >= 5) return true;

    count = CountByDirection_(1, 1, player);
    if (count >= 5) return true;

    count = CountByDirection_(1, -1, player);
    if (count >= 5) return true;

    return false;
}

int Board::PrintBoard() {
    printf("============= Board =============\n");
    for (int i = 0; i < size_; i ++) {
        for (int j = 0; j < size_; j ++) {
            printf("%d ", board_[i * size_ + j]);
        }
        printf("\n");
    }
}



