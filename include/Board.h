#ifndef GOMOKU_BOARD_H
#define GOMOKU_BOARD_H

#include <vector>
#include <unordered_set>

struct int_hash {
    inline std::size_t operator()(const int & v) const {
        return v;
    }
};

class Board {
private:
    int size_;               /* width == height == size */
    int *board_;             /* chess board: 1d array */
    int last_chess_row_;     /* row of last chess */
    int last_chess_col_;     /* column of last chess */
    int chess_count;         /* # of placed chess */

    unsigned int hash_value[2]; /* hash value of two players */

    int CountByDirection_(int dx, int dy, int player);
public:
    Board(int size);

    Board(const Board&);

    ~Board();

    /* Get size of the board */
    int GetSize();

    /* Get the chess of (row, col): 0, 1, 2 */
    int GetChess(int row, int col);

    /* set each location to 0 */
    int Initialize();

    /* place chess to (x, y). return 1 if success else 0 */
    int PlaceChess(int player_num, int x, int y);

    /* return true if someone win. notice that this function must be called after each PlaceChess */
    int IsFinish();

    /* Is (x, y) within the boundary. */
    bool IsValid(int x, int y);

    /* print the chess board. */
    int PrintBoard();

    /* Get next possible moves */
    std::unordered_set<int, int_hash> AvailableChildren(int dist=1);

    /* revert one chess (dangerous method: do not call this method of the game board)*/
    int Revert(int x, int y);
};

#endif //GOMOKU_BOARD_H
